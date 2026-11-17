"""Contract-based /ask routes on the existing FastAPI application."""
import asyncio
import threading
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict, StringConstraints

from backend.ask_service import AskService, memory
from backend.services import InvalidImageUpload, SceneNotFound
from core.registry import capability_state, CAPABILITY_TOOLS
from core.schema import Answer
from core.schema_version import SCHEMA_VERSION
from orchestrator.trace import TraceIntegrityError

router=APIRouter(tags=['ask'])
_INIT_LOCK=threading.Lock()
Text=Annotated[str,StringConstraints(strip_whitespace=True,min_length=1,max_length=2000)]


class PackRef(BaseModel):
    model_config=ConfigDict(extra='forbid')
    pack_id: Text
    patch_id: Text


class AskRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    question: Text
    scene_ref: PackRef | Annotated[str,StringConstraints(pattern=r'^upload_[0-9a-f]{32}$')]
    second_scene_ref: PackRef | Annotated[str,StringConstraints(pattern=r'^upload_[0-9a-f]{32}$')] | None = None


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
    return {'capabilities':[capability_state(name,registry=runtime.registry) for name in CAPABILITY_TOOLS],
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
    return {'schema_version':SCHEMA_VERSION,'code_commit':runtime.commit,'tools_loaded':runtime.loaded,
            'tools_registered':list(runtime.registry.names()),'timeout_s':runtime.config.timeout,
            'packs':['worldcover','sen1floods11-hand-v1'],**memory()}
