from dataclasses import replace
import json
from pathlib import Path
import time
from typing import get_args

from fastapi.testclient import TestClient
import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from backend.main import app
from backend.ask_service import AskService
from backend.config import AskConfig
from backend.routes.ask import PackRef
from core.gate import GateDecision, Thresholds
from core.registry import ToolNeeds, ToolRegistry
from core.schema import Evidence, RefusalReason, ToolResult
from orchestrator import trace


class FakeTool:
    name='flood-otsu';version='fake-v1';needs=ToolNeeds(sensors=('S1',),bands=('VH',))
    def __init__(self): self.calls=0
    def run(self,scene,query,pixels):
        self.calls+=1
        return ToolResult(tool=self.name,tool_version=self.version,value={'answer':'Fake result'},latency_ms=1)


def tiff(*,bands=('VV','VH'),units='dB',crs='EPSG:32643',cloud=None,nodata=-9999,all_nodata=False,size=120):
    with MemoryFile() as memory:
        with memory.open(driver='GTiff',width=size,height=size,count=len(bands),dtype='float32',
                         crs=crs,transform=from_origin(500000,2400000,10,10),nodata=nodata) as dst:
            data=np.full((len(bands),size,size),-20,dtype=np.float32)
            if all_nodata: data[:]=nodata
            dst.write(data)
            for i,name in enumerate(bands,1):
                if name is not None: dst.set_band_description(i,name)
                unit='dB' if name in ('VV','VH') else 'reflectance_x1e4'
                if units is not None: dst.update_tags(i,units=unit if units=='mixed' else units)
            if cloud is not None: dst.update_tags(cloud_pct=str(cloud))
        return memory.read()


@pytest.fixture
def api(tmp_path,monkeypatch):
    status=tmp_path/'status.json';evidence=tmp_path/'evidence.json';evidence.write_text('{}')
    status.write_text(json.dumps({'schema_version':1,'tools':{'flood-otsu':{
        'status':'validated','evidence_refs':[{'commit':'1'*40,'results':'evidence.json'}]}}}))
    reg=ToolRegistry(status,tmp_path);tool=reg.register(FakeTool())
    config=AskConfig(data_dir=tmp_path/'data',trace_dir=tmp_path/'trace',timeout=2)
    service=AskService(config,reg)
    monkeypatch.setattr(app.state,'ask_service',service,raising=False)
    with TestClient(app) as client:
        yield client,service,tool
    service.close()


def upload(client,data=None):
    response=client.post('/scenes',files={'file':('scene.tif',data if data is not None else tiff(),'image/tiff')})
    assert response.status_code==201,response.text
    return response.json()['upload_id']


def ask(client,ref,question='Map flood water',**extra):
    return client.post('/ask',json={'question':question,'scene_ref':ref,**extra})


def test_answered_and_trace_and_health(api):
    client,runtime,tool=api
    ref=upload(client);response=ask(client,ref)
    assert response.status_code==200,response.text
    a=response.json();assert a['status']=='answered' and a['confidence'] is None
    t=client.get('/trace/'+a['trace_id']).json()
    assert t['verified'] and t['events'] and all(e['trace_id']==a['trace_id'] for e in t['events'])
    assert client.get('/trace').json()['verified']
    h=client.get('/health').json()
    assert h['schema_version']=='1.3.0' and len(h['code_commit'])==40
    assert h['tools_loaded']=={'flood-otsu':['ready']} and tool.calls==1
    assert client.get('/capabilities').json()['tools'][0]['status']=='validated'


def test_experimental_always_low_confidence(api):
    client,runtime,tool=api
    tool.status=tool.status.model_copy(update={'status':'experimental_unvalidated'})
    a=ask(client,upload(client)).json()
    assert a['status']=='low_confidence' and a['confidence'] is None
    assert a['evidence'][-1]['summary']['label']=='experimental_unvalidated'


@pytest.mark.parametrize('reason',get_args(RefusalReason))
def test_all_refusal_reasons_cross_api_boundary(api,monkeypatch,reason):
    client,runtime,tool=api
    def gate(query,scene,choice,registry,**kwargs):
        return GateDecision(None,'S1',reason,(Evidence(kind='text',summary={'rule':reason}),))
    monkeypatch.setattr('core.answer.evaluate',gate)
    a=ask(client,upload(client)).json()
    assert a['status']=='refused' and a['refusal_reason']==reason and a['confidence'] is None
    assert a['primary'] is None and tool.calls==0


@pytest.mark.parametrize('question,reason',[('What is the price?','not_answerable'),
                                         ('What changed since last year?','not_answerable')])
def test_actual_gate_refusals(api,question,reason):
    client,runtime,tool=api
    response=ask(client,upload(client),question)
    assert response.json()['refusal_reason']==reason and tool.calls==0


def test_tool_error_is_answer_and_only_class_is_traced(api):
    client,runtime,tool=api
    def fail(*args): raise RuntimeError('secret stack contents')
    tool.run=fail
    a=ask(client,upload(client)).json()
    assert a['refusal_reason']=='tool_error' and a['status']=='refused'
    assert 'secret' not in json.dumps(a)
    chain=client.get('/trace/'+a['trace_id']).json()
    assert any(e.get('error_type')=='RuntimeError' for e in chain['events'])
    assert 'secret' not in json.dumps(chain)


def test_adapter_error_result_becomes_tool_error(api):
    client,runtime,tool=api
    tool.run=lambda *args:ToolResult(tool=tool.name,tool_version=tool.version,
                                    value={'error':'internal secret','error_type':'ValueError'},latency_ms=0)
    a=ask(client,upload(client)).json()
    assert a['refusal_reason']=='tool_error' and a['primary'] is None
    assert any(e.get('error_type')=='ValueError' for e in client.get('/trace/'+a['trace_id']).json()['events'])


def test_timeout_keeps_lock_and_suppresses_late_answer(api):
    client,runtime,tool=api
    runtime.config=replace(runtime.config,timeout=0.03)
    active=0;maximum=0
    def slow(*args):
        nonlocal active,maximum
        active+=1;maximum=max(maximum,active)
        time.sleep(0.15)
        active-=1
        return ToolResult(tool=tool.name,tool_version=tool.version,value={'answer':'late'},latency_ms=150)
    tool.run=slow;ref=upload(client)
    first=ask(client,ref).json();second=ask(client,ref).json()
    assert first['refusal_reason']==second['refusal_reason']=='timeout'
    time.sleep(0.18)
    assert maximum==1
    for a in (first,second):
        events=client.get('/trace/'+a['trace_id']).json()['events']
        assert not any(e.get('status')=='answered' for e in events)


def test_trace_tamper_reverified_on_read_and_blocks_execution(api):
    client,runtime,tool=api
    ref=upload(client);a=ask(client,ref).json()
    path=runtime.config.trace_dir/'trace.jsonl'
    lines=path.read_text().splitlines();line=json.loads(lines[0]);line['source']='tampered'
    lines[0]=json.dumps(line);path.write_text('\n'.join(lines)+'\n')
    assert client.get('/trace/'+a['trace_id']).json()['verified'] is False
    assert ask(client,ref).status_code==503
    assert tool.calls==1


@pytest.mark.parametrize('data,reason',[(b'not a raster','GeoTIFF'),
    (tiff(crs=None),'CRS'),(tiff(bands=(None,None)),'band'),
    (tiff(units=None),'units'),(tiff(bands=('banana','VH')),'band'),
    (tiff(all_nodata=True),'valid pixels')])
def test_upload_rejects(api,data,reason):
    client,_,_=api
    response=client.post('/scenes',files={'file':('bad.tif',data,'image/tiff')})
    assert response.status_code==422 and reason.lower() in response.text.lower()


def test_upload_size_and_decode_limits(api):
    client,runtime,_=api
    runtime.config=replace(runtime.config,upload_limit=4)
    runtime.scenes.config=runtime.config
    assert client.post('/scenes',files={'file':('large.tif',b'12345')}).status_code==413
    runtime.config=replace(runtime.config,upload_limit=200*1024**2,decoded_limit=4)
    runtime.scenes.config=runtime.config
    assert client.post('/scenes',files={'file':('large.tif',tiff())}).status_code==422


def test_invalid_and_unknown_refs(api):
    client,_,_=api
    for body in ({'question':' ','scene_ref':'bad'}, {'question':'Map water','scene_ref':{'pack_id':'x'}},
                 {'question':'Map water','scene_ref':'../../etc/passwd'}):
        assert client.post('/ask',json=body).status_code==422
    assert ask(client,'upload_'+'f'*32).status_code==404
    assert ask(client,{'pack_id':'unknown','patch_id':'0'}).status_code==404
    assert client.get('/trace/nonexistent').status_code==404


@pytest.mark.parametrize('question', ['What changed?', 'Map flood water'])
def test_pair_abstains_and_advertises_change_unavailability(api, question):
    client, _, tool = api
    ref = upload(client)
    response = ask(client, ref, question, second_scene_ref=ref)
    assert response.status_code == 200
    answer = response.json()
    assert answer['status'] == 'refused' and answer['refusal_reason'] == 'not_answerable'
    assert 'T18' in answer['text'] and tool.calls == 0
    capabilities = {row['capability']: row for row in client.get('/capabilities').json()['capabilities']}
    assert capabilities['change_vqa']['state'] == 'UNAVAILABLE'
    assert not capabilities['change_vqa']['available'] and 'T18' in capabilities['change_vqa']['reason']
    assert capabilities['flood']['available']


def test_temporal_refusal_does_not_require_second_scene_pixels(api):
    client, _, tool = api
    response = ask(client, upload(client), 'What changed?', second_scene_ref='upload_'+'f'*32)
    assert response.status_code == 200
    assert response.json()['refusal_reason'] == 'not_answerable'
    assert tool.calls == 0


def test_threshold_file_changes_gate_and_adapter_consistently(api):
    client,runtime,_=api
    runtime.thresholds=Thresholds(cloud_pct=20)
    # Count requires optical and has no authorized radar alternative.
    ref=upload(client,tiff(bands=('B02','B03','B04','B08'),units='reflectance_x1e4',cloud=30))
    a=ask(client,ref,'Count cars').json()
    assert a['refusal_reason']=='cloud' and a['evidence'][-1]['needs_sensor']==['S1']


@pytest.mark.parametrize('bad',[{'cloud_pct':101},{'ood_std':float('inf')},{'india_bounds':[98,38,68,6]}])
def test_invalid_thresholds_fail_closed(bad):
    with pytest.raises(ValueError): Thresholds.model_validate(bad)


@pytest.mark.parametrize('pack,patch,question',[('worldcover','0','What is the land cover?'),
                            ('sen1floods11-hand-v1','India_900498','Map flood water')])
def test_real_cpu_end_to_end(tmp_path,monkeypatch,pack,patch,question):
    config=AskConfig(data_dir=tmp_path/'data',trace_dir=tmp_path/'trace')
    directory=config.worldcover_dir if pack=='worldcover' else config.flood_dir
    if not (directory/'MANIFEST.json').is_file(): pytest.skip(f'{pack} data absent: {directory}')
    if pack=='worldcover' and not (Path.home()/'ben-work/transfer-run-1/ben-landcover-s1s2-checkpoint.pt').is_file():
        pytest.skip('BEN s1s2 checkpoint absent')
    runtime=AskService(config)
    monkeypatch.setattr(app.state,'ask_service',runtime,raising=False)
    try:
        with TestClient(app) as client:
            response=ask(client,{'pack_id':pack,'patch_id':patch},question)
            assert response.status_code==200,response.text
            a=response.json()
            assert a['status']=='low_confidence' and a['confidence'] is None,a
            assert a['primary']['tool']==('landcover' if pack=='worldcover' else 'flood-otsu')
            assert client.get('/trace/'+a['trace_id']).json()['verified']
    finally: runtime.close()


def test_real_mumbai_s2_cloud_metadata_refuses_without_inference(tmp_path, monkeypatch):
    fixture = Path(__file__).resolve().parents[1] / 'tests/fixtures/mumbai-s2-cloudy.tif'
    runtime = AskService(AskConfig(data_dir=tmp_path/'data', trace_dir=tmp_path/'trace'))
    monkeypatch.setattr(app.state, 'ask_service', runtime, raising=False)
    with TestClient(app) as client:
        uploaded = client.post('/scenes', files={'file': (fixture.name, fixture.read_bytes(), 'image/tiff')})
        assert uploaded.status_code == 201, uploaded.text
        data = uploaded.json()
        assert data['scene']['sensor'] == 'S2' and data['scene']['cloud_pct'] == 99.980801
        response = ask(client, data['upload_id'], 'What is the land cover?')
        assert response.status_code == 200, response.text
        answer = response.json()
        assert answer['status'] == 'refused' and answer['refusal_reason'] == 'cloud'
        assert answer['primary'] is None and any(e['needs_sensor'] == ['S1'] for e in answer['evidence'])
        chain = client.get('/trace/' + answer['trace_id']).json()
        assert chain['verified'] and not any(e['phase'] == 'execution' for e in chain['events'])


def test_error_after_timeout_is_audited_without_late_answer(api):
    client,runtime,tool=api
    runtime.config=replace(runtime.config,timeout=0.03)
    def slow_error(*args):
        time.sleep(0.08)
        raise ArithmeticError('private traceback contents')
    tool.run=slow_error
    a=ask(client,upload(client)).json()
    assert a['refusal_reason']=='timeout'
    time.sleep(0.1)
    events=client.get('/trace/'+a['trace_id']).json()['events']
    assert any(e.get('error_type')=='ArithmeticError' for e in events)
    assert not any(e.get('status')=='answered' for e in events)
    assert 'private' not in json.dumps(events)


def test_validity_mask_affects_scene_identity_and_preserves_band(api):
    client,runtime,_=api
    arrays=[]
    for col in (0,1):
        with MemoryFile() as memory:
            with memory.open(driver='GTiff',width=2,height=2,count=3,dtype='float32',crs='EPSG:32643',
                             transform=from_origin(500000,2400000,10,10)) as dst:
                data=np.full((3,2,2),-20,dtype=np.float32);data[2]=1;data[2,0,col]=0
                dst.write(data)
                for i,name in enumerate(('VV','VH','dataMask'),1):
                    dst.set_band_description(i,name);dst.update_tags(i,units='dB')
            arrays.append(runtime.scenes.raster(memory.read()).scene)
    assert arrays[0].nodata_pct==arrays[1].nodata_pct==25
    assert arrays[0].bands==('VV','VH','dataMask')
    assert arrays[0].scene_id!=arrays[1].scene_id


def test_actual_nodata_gate_with_partial_upload(api):
    client,runtime,tool=api
    runtime.thresholds=Thresholds(nodata_pct=10)
    ref=upload(client)
    original=runtime.scenes.load
    def load(reference):
        item=original(reference)
        item.scene=item.scene.model_copy(update={'nodata_pct':20})
        return item
    runtime.scenes.load=load
    a=ask(client,ref).json()
    assert a['refusal_reason']=='nodata' and tool.calls==0


def test_fresh_outside_india_refuses_before_stac(api, monkeypatch):
    client, _, tool = api
    monkeypatch.setenv('SATQUERY_OFFLINE', '0')

    def unexpected_fetch(*args):
        pytest.fail('An outside-India point must never reach STAC')

    monkeypatch.setattr('backend.routes.ask.fetch_chip', unexpected_fetch)
    response = client.post('/scenes/fetch', json={'lat': 48.45, 'lon': 16.15, 'start_date': '2025-08-29'})
    assert response.status_code == 200
    refusal = response.json()['refusal']
    assert refusal['refusal_reason'] == 'not_answerable'
    history = client.get('/trace/' + refusal['trace_id']).json()
    assert history['verified']
    assert any(event.get('policy') == 'out_of_region' for event in history['events'])
    assert tool.calls == 0


@pytest.mark.parametrize('offline_mode', [True, False])
def test_fresh_fallback_has_provenance_and_uses_the_same_gate(api, monkeypatch, offline_mode):
    from backend.fresh_scenes import ProviderError

    client, _, tool = api
    monkeypatch.setenv('SATQUERY_OFFLINE', '1' if offline_mode else '0')

    def failed_fetch(*args):
        assert not offline_mode, 'Offline mode must not call STAC'
        raise ProviderError('Earth Search unavailable.')

    monkeypatch.setattr('backend.routes.ask.fetch_chip', failed_fetch)
    response = client.post('/scenes/fetch', json={'lat': 19.025, 'lon': 72.835, 'start_date': '2025-08-29'})
    assert response.status_code == 200
    scene = response.json()
    assert scene['backup_place'] == 'Mumbai'
    assert scene['provenance']['mode'] == 'cached'
    assert scene['provenance']['label'] == 'Cached scene (offline)'
    assert scene['provenance']['cloud_pct_label'] == 'scene-level cloud cover'
    assert scene['fallback_reason'] == ('Offline mode is enabled.' if offline_mode else 'Earth Search unavailable.')
    assert scene['image'].startswith('data:image/png;base64,')
    answer = ask(client, scene['scene_ref'], question='What is the land cover?').json()
    assert answer['refusal_reason'] == 'cloud'
    assert tool.calls == 0


def test_fresh_backup_manifest_matches_every_chip(api):
    from backend.fresh_scenes import BACKUPS
    from rasterio.warp import transform_bounds

    _, runtime, _ = api
    records = json.loads((BACKUPS / 'manifest.json').read_text())
    assert len({record['id'] for record in records}) == 5
    assert {record['sensor'] for record in records} == {'S1', 'S2'}
    for record in records:
        scene = runtime.scenes.load(PackRef(pack_id='fresh-backups', patch_id=record['id']))
        assert scene.provenance['item_id'] == record['item_id']
        assert scene.scene.sensor == record['sensor']
        assert scene.scene.acquired.isoformat() == record['date']
        assert transform_bounds(scene.scene.crs, 'EPSG:4326', *scene.scene.bounds) == pytest.approx(record['bbox'])
        assert scene.preview_image().size == (120, 120)
