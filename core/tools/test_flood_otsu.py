import numpy as np
import pytest

from core.answer import run_query
from core.gate import evaluate
from core.registry import ToolRegistry
from core.router import route
from core.schema import Query, Scene, ToolResult
from core.tools.flood_otsu import CLAMP_DB, FloodOtsuTool, water_mask


def scene(**changes):
    return Scene(**{**dict(scene_id='a'*64,sensor='S1',bands=('VV','VH'),units='dB',gsd_m=10,
                          crs='EPSG:4326',bounds=(75,20,75.01,20.01),nodata_pct=0,source='synthetic'),**changes})


def pixels():
    vh=np.array([[-24,-24],[-12,-12]],dtype=np.float32)
    return np.stack((vh+5,vh))


def test_contract_and_mask():
    result=FloodOtsuTool().run(scene(),Query(text='flood water',qtype='flood'),pixels())
    assert 'error' not in result.value, result.value
    assert ToolResult.model_validate_json(result.model_dump_json())==result
    assert result.raw_confidence is None
    assert result.value['water_mask']==[[1,1],[0,0]]
    assert result.value['water_fraction']==0.5
    assert CLAMP_DB[0]<=result.value['threshold_db']<=CLAMP_DB[1]


@pytest.mark.parametrize('values,expected', [([-40,-30],-26),([0,10],-14)])
def test_clamp_fixed(values,expected):
    _,raw,threshold=water_mask(np.array(values,dtype=float))
    assert threshold==expected and raw!=threshold


def test_nonfinite_dry_even_with_waterlike_values():
    vh=np.array([-24,-12,np.nan,np.inf,-np.inf])
    mask,_,threshold=water_mask(vh)
    assert mask.tolist()==[True,False,False,False,False]
    assert threshold is not None
    mask,raw,threshold=water_mask(np.array([np.nan,np.inf]))
    assert not mask.any() and raw is None and threshold is None


def test_event_pooling_uses_one_shared_window_and_extracts_patch():
    a=pixels();b=pixels()+8
    stack=np.stack((a,b))
    all_masks,_,threshold=water_mask(stack[:,1])
    tool=FloodOtsuTool();q=Query(text='Map flood',qtype='flood')
    for i in range(2):
        result=tool.run(scene(),q,{'s1':{'pixels':stack,'bands':('VV','VH'),'units':'dB','chip_index':i}})
        assert result.value['threshold_db']==threshold
        assert np.array_equal(result.value['water_mask'],all_masks[i])
        assert result.evidence[0].summary['window_pixels']==8


@pytest.mark.parametrize('changes', [dict(units='unknown'),dict(sensor='S2'),dict(bands=('VV','B02'))])
def test_metadata_rejected(changes):
    assert 'error' in FloodOtsuTool().run(scene(**changes),Query(text='flood',qtype='flood'),pixels()).value


@pytest.mark.parametrize('bad', [np.zeros((2,0,1)),np.zeros((3,2,2)),np.zeros((2,2))])
def test_invalid_pixel_shape_rejected(bad):
    assert 'error' in FloodOtsuTool().run(scene(),Query(text='flood',qtype='flood'),bad).value


def test_flood_routes_to_registered_radar_with_joint_monsoon_payload(tmp_path):
    reg=ToolRegistry(tmp_path/'absent.json');reg.register(FloodOtsuTool())
    a=scene(sensor='S1S2',bands=('VV','VH','B02','B03','B04','B08'),units='unknown',cloud_pct=80)
    q=Query(text='Map flood water',qtype='flood')
    choice=route(q,a);gate=evaluate(q,a,choice,reg)
    assert choice.tools==('flood-otsu',) and gate.passed and gate.mode=='S1'
    answer=run_query(q,a,{'S1':{'s1':{'pixels':pixels(),'bands':('VV','VH'),'units':'dB'}}},reg,trace_id='actual-flood')
    assert answer.status=='low_confidence' and answer.confidence is None
    assert answer.primary.value['water_mask']==[[1,1],[0,0]]
    assert any(e.summary.get('warning')=='monsoon_mode' for e in answer.evidence)


def test_no_radar_refuses_sensor_even_if_flood_intent(tmp_path):
    reg=ToolRegistry(tmp_path/'absent.json');reg.register(FloodOtsuTool())
    a=scene(sensor='S2',units='reflectance_x1e4',bands=('B02','B03','B04','B08'))
    q=Query(text='Map flood water',qtype='flood')
    gate=evaluate(q,a,route(q,a),reg)
    assert gate.refusal_reason=='sensor_missing' and gate.evidence[-1].needs_sensor==('S1',)


def test_eval_counts_ignore_only_label_nodata_and_bootstrap_chips():
    from scripts.evaluate_t07c import counts, metrics
    pred=np.array([True,True,False,False])
    label=np.array([-1,0,1,0])
    assert counts(pred,label)==[0,1,1]
    measured=metrics([[1,1,0],[1,1,0]])
    assert measured['iou']==0.5 and measured['ci95']==[0.5,0.5]
    assert metrics([[0,0,0]])['iou'] is None
    assert metrics([[0,0,0]])['undefined_bootstraps']==2000
