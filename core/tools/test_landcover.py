import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core.answer import run_query
from core.gate import evaluate
from core.registry import ToolRegistry
from core.router import route
from core.schema import Query, Scene, ToolResult
from core.tools.landcover import BANDS, CLASSES, UNITS, LandcoverTool, select_mode


def scene(mode='s2', **updates):
    keys = ('s1', 's2') if mode == 's1s2' else (mode,)
    return Scene(**{**dict(scene_id='a'*64, sensor=mode.upper(), bands=tuple(b for k in keys for b in BANDS[k]),
                          units=UNITS.get(mode, 'unknown'), gsd_m=10, crs='EPSG:4326',
                          bounds=(75,20,75.01,20.01), source='synthetic', cloud_pct=10, nodata_pct=0), **updates})


def payload(mode):
    return {k: {'pixels': np.ones((len(BANDS[k]),120,120), np.float32), 'bands': BANDS[k], 'units': UNITS[k]}
            for k in (('s1','s2') if mode == 's1s2' else (mode,))}


class FakeModel:
    def __call__(self, x):
        assert torch.isfinite(x).all()
        return torch.arange(19, dtype=torch.float32)[None] / 10


def fake_tool(monkeypatch):
    tool = LandcoverTool()
    stats = {'s1': {'mean': [0,0], 'std':[1,1], 'clip_lo':[-50,-50], 'clip_hi':[10,10]},
             's2': {'mean':[0]*4, 'std':[1]*4, 'scale':0.0001}}
    members = {c:[i, i+5] for i,c in enumerate(CLASSES)}
    monkeypatch.setattr(tool, '_load', lambda mode:(FakeModel(), stats, members, 'b'*64))
    return tool


@pytest.mark.parametrize('mode', ['s1','s2','s1s2'])
def test_synthetic_landcover_contract(mode, monkeypatch):
    tool = fake_tool(monkeypatch)
    result = tool.run(scene(mode), Query(text='land cover', qtype='landcover'), payload(mode))
    assert 'error' not in result.value, result.value
    assert ToolResult.model_validate_json(result.model_dump_json()) == result
    assert tuple(result.value['scores']) == CLASSES
    for i,c in enumerate(CLASSES):
        assert result.value['scores'][c] == pytest.approx(float(torch.sigmoid(torch.tensor((i+5)/10))))
    assert result.value['mode'] == mode and result.raw_confidence is None


@pytest.mark.parametrize('mode,cloud,expected', [('s1',10,'s1'),('s2',10,'s2'),('s1s2',10,'s1s2'),('s1s2',80,'s1')])
def test_mode_selection(mode, cloud, expected):
    assert select_mode(scene(mode, cloud_pct=cloud)) == expected


@pytest.mark.parametrize('bad', ['band_order','units','shape','nonfinite'])
def test_bad_inputs_refuse_before_weights_load(bad, monkeypatch):
    tool = LandcoverTool()
    monkeypatch.setattr(tool, '_load', lambda _: pytest.fail('Must validate before loading'))
    px=payload('s2')
    if bad=='band_order': px['s2']['bands']=tuple(reversed(BANDS['s2']))
    if bad=='units': px['s2']['units']='dB'
    if bad=='shape': px['s2']['pixels']=np.zeros((4,10,10))
    if bad=='nonfinite': px['s2']['pixels'][0,0,0]=np.nan
    assert 'error' in tool.run(scene(), Query(text='land cover',qtype='landcover'),px).value


@pytest.mark.parametrize('mode', ['s1','s2','s1s2'])
def test_real_checkpoint(mode):
    tool = LandcoverTool()
    path=Path.home()/f'ben-work/transfer-run-1/ben-landcover-{mode}-checkpoint.pt'
    if not path.is_file(): pytest.skip('BEN checkpoint absent')
    result=tool.run(scene(mode),Query(text='land cover',qtype='landcover'),payload(mode))
    assert 'error' not in result.value, result.value
    assert len(result.value['scores'])==5 and result.checkpoint_sha
    assert all(0<=v<=1 for v in result.value['scores'].values())


def test_registered_landcover_route_and_monsoon(tmp_path, monkeypatch):
    reg=ToolRegistry(tmp_path/'absent.json')
    tool=reg.register(fake_tool(monkeypatch))
    q=Query(text='What is the land cover?',qtype='landcover')
    for mode, cloud, expected_mode in [('s2',10,'s2'),('s1s2',10,'s1s2'),('s1s2',80,'s1')]:
        a=scene(mode,cloud_pct=cloud)
        choice=route(q,a)
        gate=evaluate(q,a,choice,reg)
        assert choice.tools==('landcover',) and gate.passed
        assert gate.mode==expected_mode.upper()
        answer=run_query(q,a,{gate.mode:payload(expected_mode)},reg,trace_id='landcover-route')
        assert answer.status=='low_confidence' and answer.confidence is None
        assert answer.primary.value['mode']==expected_mode


def test_lazy_registration_does_not_load_weights():
    assert LandcoverTool()._loaded == {}
