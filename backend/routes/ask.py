"""Contract-based /ask routes on the existing FastAPI application."""
import asyncio
import json
from pathlib import Path
import threading
from time import monotonic
from typing import Annotated

import httpx

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, StringConstraints

from backend.ask_service import AskService, memory
from backend.fresh_scenes import (FreshRequest, ProviderError, SceneUnavailable,
                                  backup_for, fetch_chip, offline, resolve_point)
from backend.services import InvalidImageUpload, SceneNotFound, health_capabilities
from core.registry import capability_state, CAPABILITY_TOOLS
from core.schema import Answer
from core.schema_version import SCHEMA_VERSION
from orchestrator.trace import TraceIntegrityError

router=APIRouter(tags=['ask'])
_INIT_LOCK=threading.Lock()
_LIVE_READS = threading.BoundedSemaphore(4)
Text=Annotated[str,StringConstraints(strip_whitespace=True,min_length=1,max_length=2000)]


class PackRef(BaseModel):
    model_config=ConfigDict(extra='forbid')
    pack_id: Text
    patch_id: Text


SceneRef = PackRef | Annotated[str,StringConstraints(pattern=r'^upload_[0-9a-f]{32}$')]


class AskRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    question: Text
    scene_ref: SceneRef
    second_scene_ref: SceneRef | None = None


def service(request):
    with _INIT_LOCK:
        if not hasattr(request.app.state,'ask_service'):
            request.app.state.ask_service=AskService()
    return request.app.state.ask_service


@router.post('/ask',response_model=Answer)
async def ask(body:AskRequest,request:Request):
    try:
        return await service(request).ask(body)
    except SceneNotFound as error: raise HTTPException(404,str(error)) from error
    except (InvalidImageUpload,ValueError) as error: raise HTTPException(422,str(error)) from error
    except (TraceIntegrityError,OSError) as error:
        raise HTTPException(503,'Persistent execution trace or local scene storage is unavailable') from error


def _acquire_scene(body, bounds, deadline):
    if not _LIVE_READS.acquire(timeout=max(0, deadline - monotonic())):
        raise TimeoutError
    try:
        if monotonic() >= deadline:
            raise TimeoutError
        with httpx.Client(timeout=10) as client:
            point, fallback_reason = resolve_point(client, body)
            if not _in_bounds(point, bounds) or fallback_reason:
                return point, fallback_reason, None
            if monotonic() >= deadline:
                raise TimeoutError
            try:
                return point, None, fetch_chip(client, body, point, deadline=deadline)
            except ProviderError as error:
                return point, str(error), None
    finally:
        _LIVE_READS.release()


def _in_bounds(point, bounds):
    west, south, east, north = bounds
    lon, lat = point
    return west <= lon <= east and south <= lat <= north


@router.post('/scenes/fetch')
async def fresh_scene(body: FreshRequest, request: Request):
    runtime = service(request)
    bounds = runtime.thresholds.india_bounds
    point = (body.lon, body.lat) if body.place is None else None
    if point is not None and not _in_bounds(point, bounds):
        return {'refusal': runtime.outside_region()}
    try:
        if offline():
            backup = backup_for(body, point)
            reason = 'Offline mode is enabled.'
        else:
            timeout = runtime.config.live_fetch_timeout
            try:
                # GDAL can finish its current read after cancellation. Only the caller publishes a chip.
                resolved, reason, pixels = await asyncio.wait_for(
                    asyncio.to_thread(_acquire_scene, body, bounds, monotonic() + timeout), timeout)
            except TimeoutError:
                reason = f'Live fetch timed out after {timeout:g}s'
                try:
                    backup = backup_for(body, point)
                    if point is not None and not _in_bounds(point, backup['bbox']):
                        raise SceneUnavailable
                except SceneUnavailable:
                    raise HTTPException(504, f'{reason}. No cached scene is available for this location.') from None
                reason += f" — showing cached scene from {backup['date']}"
            else:
                if not _in_bounds(resolved, bounds):
                    return {'refusal': runtime.outside_region()}
                if pixels is not None:
                    chip = runtime.scenes.raster(pixels, fresh_mode='live')
                    ref = PackRef(pack_id='fresh', patch_id=runtime.scenes.save_fresh(pixels))
                    return {'scene_ref': ref, **chip.preview(), 'fallback_reason': None}
                backup = backup_for(body, resolved)
        ref = PackRef(pack_id='fresh-backups', patch_id=backup['id'])
        return {'scene_ref': ref, **runtime.scenes.load(ref).preview(),
                'fallback_reason': reason, 'backup_place': backup['place']}
    except (SceneUnavailable, SceneNotFound) as error:
        raise HTTPException(404, str(error)) from error
    except (OSError, ValueError, TraceIntegrityError) as error:
        raise HTTPException(503, 'Scene storage or execution trace is unavailable.') from error


class PreviewRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    scene_ref: SceneRef


@router.post('/scenes/preview')
def scene_preview(body: PreviewRequest, request: Request):
    try:
        return service(request).scenes.load(body.scene_ref).preview()
    except SceneNotFound as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@router.get('/scenes/demo/{fixture}')
def demo_fixture(fixture: str):
    if fixture not in {'punjab-s2.tif', 'mumbai-s2-cloudy.tif', 'mumbai-s2-cloudy-v2.tif'}:
        raise HTTPException(404, 'Unknown demo fixture')
    path = Path(__file__).resolve().parents[2] / 'tests' / 'fixtures' / fixture
    if not path.is_file():
        raise HTTPException(404, 'Demo fixture is not available on this installation')
    return FileResponse(path, media_type='image/tiff', filename=fixture)


@router.post('/scenes',status_code=201)
async def upload(request:Request,file:UploadFile=File(...)):
    runtime=service(request)
    chunks=[];size=0
    try:
        async with runtime.upload_lock:
            while chunk:=await file.read(1024*1024):
                size+=len(chunk)
                if size>runtime.config.upload_limit: raise HTTPException(413,'GeoTIFF exceeds the 200 MiB upload limit')
                chunks.append(chunk)
            loop=asyncio.get_running_loop()
            return await loop.run_in_executor(runtime.executor,runtime.scenes.upload,b''.join(chunks))
    except InvalidImageUpload as error: raise HTTPException(422,str(error)) from error
    finally:
        await file.close()


@router.get('/capabilities')
def capabilities(request:Request):
    runtime=service(request)
    evidence = Path(__file__).resolve().parents[2] / "docs/evidence/capabilities.json"
    return {'evidence': json.loads(evidence.read_text(encoding="utf-8")),
            'capabilities':[capability_state(name,registry=runtime.registry) for name in CAPABILITY_TOOLS],
            'tools':[t.status.model_dump(mode='json') for t in runtime.registry.list()]}


@router.get('/trace')
@router.get('/trace/{trace_id}')
def get_trace(request:Request,trace_id:str|None=None):
    try:
        result=service(request).verified_trace(trace_id)
    except (TraceIntegrityError,OSError):
        return {'events':[],'verified':False,'message':'Persisted trace integrity could not be verified','chain_scope':'entire persisted chain'}
    if trace_id and not result['events'] and result['verified']: raise HTTPException(404,'Unknown trace id')
    return result


@router.get('/health')
def health(request:Request):
    runtime=service(request)
    checks, disabled, host_limits = health_capabilities()
    loaded = runtime.loaded.copy()
    if runtime.vqa is not None:
        reason = runtime.vqa.unavailable_reason
        checks['single_image_vqa'] = {'available': not reason, 'reason_code': reason}
        host_limits.pop('qwen2.5vl-3b', None)
        if reason:
            loaded.pop(runtime.vqa.name, None)
    return {'status':'ready' if all(check['available'] for check in checks.values()) else 'degraded',
            'disabled_by_decision':disabled,'unavailable_on_host':host_limits,'capability_checks':checks,
            'schema_version':SCHEMA_VERSION,'code_commit':runtime.commit,'tools_loaded':loaded,
            'tools_registered':list(runtime.registry.names()),'timeout_s':runtime.config.timeout,
            'packs':['worldcover','sen1floods11-hand-v1'],**memory()}
