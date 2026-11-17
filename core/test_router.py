"""Executable precommitted examples plus adversarial gate/answer properties."""
import json
from pathlib import Path

from hypothesis import given, strategies as st
import pytest

from core.answer import assemble, decide, run_query
from core.gate import GateDecision, evaluate
from core.registry import ToolNeeds, ToolRegistry
from core.router import route
from core.schema import Answer, Evidence, Query, Scene, ToolResult, ToolStatus
from orchestrator import trace

SPEC = Path(__file__).resolve().parents[1] / 'docs/plan/ROUTER_SPEC.md'
EXAMPLES = [tuple(part.strip() for part in line.strip('|').split('|'))
            for line in SPEC.read_text().splitlines() if line.startswith('| E')]
NAMES = ('landcover', 'flood-otsu', 'change-deterministic', 'grounding-dino-swint',
         'qwen2.5vl-3b', 'optical-sar-deterministic')
BANDS = ('B02', 'B03', 'B04', 'B08', 'dataMask')
RADAR = ('VV', 'VH', 'dataMask')


def scene(**updates):
    return Scene(**{**dict(scene_id='a' * 64, sensor='S2', bands=BANDS,
                          units='reflectance_x1e4', gsd_m=10, crs='EPSG:4326',
                          bounds=(75, 20, 75.1, 20.1), cloud_pct=10, nodata_pct=0,
                          source='synthetic'), **updates})


class FakeTool:
    version = 'test-v1'
    needs = ToolNeeds()

    def __init__(self, name):
        self.name, self.calls = name, []
        if name in ('landcover', 'change-deterministic'):
            self.needs = ToolNeeds(sensors=('S1', 'S2'))
        elif name == 'flood-otsu':
            self.needs = ToolNeeds(sensors=('S1',), bands=('VH',))
        elif name == 'optical-sar-deterministic':
            self.needs = ToolNeeds(sensors=('S1', 'S2'), bands=(*BANDS, 'VV', 'VH'))

    def run(self, scene, query, pixels):
        self.calls.append((scene, query, pixels))
        return ToolResult(tool=self.name, tool_version=self.version, latency_ms=1,
                          raw_confidence=0.9, value={'answer': 'Synthetic answer',
                                                   'confidence': 0.99, 'nested': [{'raw_confidence': 0.98, 'area': 12}]},
                          evidence=(Evidence(kind='text', summary={'confidence': 0.8}),))


def registry(tmp_path, *, status='experimental_unvalidated', missing=(), disabled=()):
    path = tmp_path / 'status.json'
    evidence = tmp_path / 'results.json'
    evidence.write_text('{}')
    path.write_text(json.dumps({'schema_version': 1, 'tools': {
        name: {'status': 'disabled' if name in disabled else status,
               'evidence_refs': [{'commit': '1' * 40, 'results': 'results.json'}]}
        for name in NAMES if name not in missing}}))
    reg = ToolRegistry(path, tmp_path)
    for name in NAMES:
        if name not in missing:
            reg.register(FakeTool(name))
    return reg


def fixture(profile, tmp_path):
    updates = {}
    if profile.startswith('s1'):
        updates.update(sensor='S1', bands=RADAR, units='dB', cloud_pct=None)
    if 'joint' in profile:
        updates.update(sensor='S1S2', bands=(*BANDS, 'VV', 'VH'))
    if profile == 'rgb':
        updates.update(sensor='RGB_HR', bands=('red', 'green', 'blue'), units='uint8')
    if 'cloud' in profile:
        updates['cloud_pct'] = 80
    if 'outside' in profile:
        updates['bounds'] = (-2, 45, -1, 46)
    if 'nodata' in profile:
        updates['nodata_pct'] = 60
    if 'missing_band' in profile:
        updates['bands'] = tuple(b for b in BANDS if b != 'B08')
    a = scene(**updates)
    second = None
    if profile.startswith('change'):
        if profile == 'change_s1':
            a = scene(sensor='S1', bands=RADAR, units='dB', cloud_pct=None)
        second = a.model_copy(update={'scene_id': 'b' * 64})
    missing = tuple(name for marker, name in [('no_radar', 'flood-otsu'),
                    ('no_vlm', 'qwen2.5vl-3b'), ('no_landcover', 'landcover')] if marker in profile)
    reg = registry(tmp_path, missing=missing, disabled=('landcover',) if profile == 'tool_disabled' else ())
    means = {b: 100.0 for b in a.bands}
    if profile == 'ood':
        means['B04'] = 131.8
    inputs = dict(second_scene=second, band_means=(means, means),
                  training_stats={b: (100.0, 10.0) for b in a.bands})
    return a, reg, inputs


@pytest.mark.parametrize('case_id,text,profile,expected', EXAMPLES, ids=[e[0] for e in EXAMPLES])
def test_spec_examples(case_id, text, profile, expected, tmp_path):
    a, reg, inputs = fixture(profile, tmp_path)
    q = Query(text=text, qtype='open', needs_gsd_m=1 if profile == 'coarse_count' else None)
    choice, gate = decide(q, a, reg, trace_id=case_id, **inputs)
    actual = f'refuse:{gate.refusal_reason}' if not gate.passed else gate.tool
    assert actual == expected
    assert trace.verify_chain()[0]
    if case_id in ('E12', 'E13', 'E18'):
        assert any(e.summary.get('warning') == 'monsoon_mode' for e in gate.evidence)
    if case_id == 'E14':
        assert any(e.summary.get('warning') == 'out_of_region' for e in gate.evidence)
    if case_id == 'E26':
        assert any(e.summary.get('warning') == 'out_of_distribution' for e in gate.evidence)
    assert not any(tool.calls for tool in reg.list())


def test_spec_has_forty_examples_and_twelve_refusals():
    assert len(EXAMPLES) == 40
    assert len({e[0] for e in EXAMPLES}) == 40
    assert sum(e[3].startswith('refuse:') for e in EXAMPLES) >= 12


@given(reason=st.sampled_from(('cloud', 'nodata', 'not_answerable', 'tool_missing',
                              'bands_missing', 'second_scene_missing', 'tool_disabled',
                              'sensor_missing', 'gsd_too_coarse', 'checks_disagree')))
def test_refusal_property(reason):
    result = assemble(None, GateDecision(None, 'S2', reason, ()), None, trace_id='property')
    assert result.status == 'refused' and result.refusal_reason and result.confidence is None
    assert result.primary is None


@given(raw=st.one_of(st.none(), st.floats(min_value=0, max_value=1, allow_nan=False)), clean=st.booleans())
def test_unvalidated_never_answered_property(raw, clean):
    result = ToolResult(tool='fake', tool_version='v1', value={'confidence': 0.99, 'nested': [{'raw_confidence': 0.9, 'count': 2}]},
                        raw_confidence=raw, latency_ms=1)
    gate = GateDecision('fake', 'S2', None, () if clean else (Evidence(kind='text', summary={'warning': 'out_of_distribution'}),))
    answer = assemble(result, gate, ToolStatus(name='fake', version='v1'), trace_id='property')
    assert answer.status == 'low_confidence' and answer.confidence is None
    assert answer.primary.raw_confidence is None
    assert answer.primary.value == {'nested': [{'count': 2}]}
    assert answer.evidence[-1].summary['label'] == 'experimental_unvalidated'


@given(cloud=st.floats(min_value=0, max_value=100, allow_nan=False),
       nodata=st.floats(min_value=0, max_value=100, allow_nan=False),
       text=st.sampled_from([e[1] for e in EXAMPLES]))
def test_pure_gate_property(cloud, nodata, text):
    reg = ToolRegistry('/nonexistent/t08-test-status')
    reg.register(FakeTool('landcover'))
    a = scene(cloud_pct=cloud, nodata_pct=nodata)
    q = Query(text=text, qtype='open')
    choice = route(q, a)
    before = a.model_dump_json(), q.model_dump_json(), tuple(trace.records())
    assert evaluate(q, a, choice, reg) == evaluate(q, a, choice, reg)
    assert before == (a.model_dump_json(), q.model_dump_json(), tuple(trace.records()))
    assert not reg.get('landcover').calls


def test_monsoon_executes_only_radar_and_traces_every_decision(tmp_path):
    a, reg, inputs = fixture('joint_cloud', tmp_path)
    q = Query(text='Map flood water', qtype='flood')
    answer = run_query(q, a, {'S1': b'radar', 'S1S2': b'cloudy'}, reg, trace_id='monsoon', **inputs)
    assert answer.status == 'low_confidence' and answer.confidence is None
    assert reg.get('flood-otsu').calls[0][2] == b'radar'
    assert not reg.get('optical-sar-deterministic').calls
    events = trace.records()
    assert sum(e.get('phase') == 'route' and 'rank' in e for e in events) == 1
    assert any(e.get('warning') == 'monsoon_mode' for e in events)
    for rule in ('cloud_threshold', 'nodata_threshold', 'region', 'ood_threshold', 'calibration', 'status'):
        assert any(e.get('rule') == rule for e in events)
    assert trace.verify_chain()[0]


def test_monsoon_missing_radar_pixels_never_runs_optical(tmp_path):
    a, reg, inputs = fixture('joint_cloud', tmp_path)
    answer = run_query(Query(text='Map flood', qtype='flood'), a, {'S1S2': b'cloudy'}, reg,
                       trace_id='missing-pixels', **inputs)
    assert answer.refusal_reason == 'sensor_missing'
    assert not any(t.calls for t in reg.list())


@pytest.mark.parametrize('profile,reason', [('s2_cloud', 'cloud'), ('s2_nodata', 'nodata'),
                         ('s2_missing_band', 'bands_missing'), ('tool_disabled', 'tool_disabled'),
                         ('no_landcover', 'tool_missing')])
def test_refused_preflight_never_executes(profile, reason, tmp_path):
    a, reg, inputs = fixture(profile, tmp_path)
    answer = run_query(Query(text='land cover', qtype='landcover'), a, {'S2': b'pixels'}, reg,
                       trace_id=profile, **inputs)
    assert answer.refusal_reason == reason and answer.confidence is None
    assert not any(t.calls for t in reg.list())
    if reason == 'cloud':
        assert any(e.needs_sensor == ('S1',) for e in answer.evidence)
    if reason == 'bands_missing':
        assert any(e.summary.get('missing_bands') == ['B08'] for e in answer.evidence)


@pytest.mark.parametrize('value,passed', [(50, True), (50.00001, False), (100, False), (0, True)])
@pytest.mark.parametrize('field,reason', [('cloud_pct', 'cloud'), ('nodata_pct', 'nodata')])
def test_threshold_boundaries(value, passed, field, reason, tmp_path):
    reg = registry(tmp_path)
    a = scene(**{field: value})
    q = Query(text='land cover', qtype='open')
    gate = evaluate(q, a, route(q, a), reg)
    assert gate.passed == passed
    if not passed:
        assert gate.refusal_reason == reason


@pytest.mark.parametrize('update,warning', [({'crs': 'INVALID'}, 'region_unknown'),
    ({'cloud_pct': None}, 'cloud_unknown'), ({'bounds': (-2, 45, -1, 46)}, 'out_of_region')])
def test_unknown_metadata_warns(update, warning, tmp_path):
    reg = registry(tmp_path)
    a = scene(**update)
    q = Query(text='land cover', qtype='open')
    gate = evaluate(q, a, route(q, a), reg)
    assert gate.passed and not gate.clean
    assert any(e.summary.get('warning') == warning for e in gate.evidence)


@pytest.mark.parametrize('change', [dict(crs='EPSG:3857'), dict(gsd_m=20), dict(bands=('B02',)), dict(bounds=(76, 20, 76.1, 20.1))])
def test_temporal_grid_mismatch_refuses(change, tmp_path):
    reg = registry(tmp_path)
    a = scene()
    q = Query(text='What changed?', qtype='change')
    gate = evaluate(q, a, route(q, a), reg, second_scene=scene(**change))
    assert gate.refusal_reason == 'checks_disagree'


def test_second_scene_cloud_and_bands_checked(tmp_path):
    a, reg, inputs = fixture('change_joint_cloud', tmp_path)
    q = Query(text='What changed?', qtype='change')
    inputs['second_scene'] = a.model_copy(update={'nodata_pct': 80})
    assert evaluate(q, a, route(q, a), reg, **inputs).refusal_reason == 'nodata'
    # Identical grids, but second date lacks S1: no monsoon replacement.
    inputs['second_scene'] = a.model_copy(update={'sensor': 'S2'})
    assert evaluate(q, a, route(q, a), reg, **inputs).refusal_reason == 'cloud'


def test_explicit_change_qtype_and_sensor_requirement(tmp_path):
    reg = registry(tmp_path)
    q = Query(text='Describe the scene', qtype='change')
    assert evaluate(q, scene(), route(q, scene()), reg).refusal_reason == 'second_scene_missing'
    q = Query(text='land cover', qtype='open', needs_sensor=('S1',))
    assert evaluate(q, scene(), route(q, scene()), reg).refusal_reason == 'sensor_missing'


@pytest.mark.parametrize('shift,warn', [(2.0, False), (2.001, True), (-3.18, True)])
def test_ood_boundary_and_second_date(shift, warn, tmp_path):
    reg = registry(tmp_path)
    a = scene()
    q = Query(text='What changed?', qtype='change')
    stats = {b: (100, 10) for b in BANDS}
    means = {b: 100 for b in BANDS}
    gate = evaluate(q, a, route(q, a), reg, second_scene=scene(), training_stats=stats,
                    band_means=(means, {**means, 'B04': 100 + shift * 10}))
    assert gate.passed
    assert any(e.summary.get('warning') == 'out_of_distribution' for e in gate.evidence) == warn
    assert gate.clean == (not warn)


@pytest.mark.parametrize('stats', [None, {}, {'B04': (100, 0)}, {'B04': (float('nan'), 1)}])
def test_missing_or_invalid_training_stats_never_clean(stats, tmp_path):
    reg = registry(tmp_path)
    q = Query(text='land cover', qtype='open')
    gate = evaluate(q, scene(), route(q, scene()), reg, training_stats=stats)
    assert gate.passed and not gate.clean


def result(**updates):
    return ToolResult(**{**dict(tool='fake', tool_version='v1', checkpoint_sha='c'*64,
                              value={'answer': 'Result', 'confidence': 0.95}, raw_confidence=0.5,
                              latency_ms=1), **updates})


def calibration(tmp_path, **updates):
    path = tmp_path / 'calibration.json'
    path.write_text(json.dumps({**dict(tool='fake', tool_version='v1', checkpoint_sha='c'*64,
                                     raw=[0, 1], calibrated=[0.1, 0.7]), **updates}))
    return path


def test_validated_clean_answer_and_map_only_confidence(tmp_path):
    status = ToolStatus(name='fake', version='v1', status='validated')
    gate = GateDecision('fake', 'S2', None, ())
    a = assemble(result(), gate, status, trace_id='without-map')
    assert a.status == 'answered' and a.confidence is None
    a = assemble(result(), gate, status, trace_id='mapped', calibration_path=calibration(tmp_path))
    assert a.status == 'answered' and a.confidence == pytest.approx(0.4)
    assert 'confidence' not in a.primary.value
    mapped = [e for e in trace.records() if e.get('outcome') == 'mapped']
    assert len(mapped[-1]['map_sha256']) == 64


@pytest.mark.parametrize('updates', [dict(tool='other'), dict(tool_version='v2'), dict(checkpoint_sha=None),
     dict(raw=[1, 0]), dict(raw=[0, 0]), dict(raw=[0]), dict(calibrated=[0.1, 1.1]),
     dict(calibrated=[0.9, 0.1]), dict(raw=[True, 1]), dict(raw=[0.6, 1])])
def test_invalid_calibration_never_uses_raw_confidence(updates, tmp_path):
    a = assemble(result(), GateDecision('fake', 'S2', None, ()),
                 ToolStatus(name='fake', version='v1', status='validated'), trace_id='invalid-map',
                 calibration_path=calibration(tmp_path, **updates))
    assert a.confidence is None


def test_bad_map_error_result_and_identity_fail_closed(tmp_path):
    path = tmp_path / 'bad.json'
    path.write_text('broken')
    status = ToolStatus(name='fake', version='v1', status='validated')
    gate = GateDecision('fake', 'S2', None, ())
    assert assemble(result(), gate, status, trace_id='bad-map', calibration_path=path).confidence is None
    for bad in (result(tool='impostor'), result(tool_version='v2'), result(value={'error': 'failed'})):
        assert assemble(bad, gate, status, trace_id='bad-output').refusal_reason == 'checks_disagree'


def test_trace_failure_prevents_execution(tmp_path, monkeypatch):
    reg = registry(tmp_path)
    def fail(_):
        raise OSError('audit disk unavailable')
    monkeypatch.setattr('core.answer.append_record', fail)
    with pytest.raises(OSError):
        run_query(Query(text='land cover', qtype='open'), scene(), {'S2': b'px'}, reg, trace_id='trace-fail')
    assert not any(t.calls for t in reg.list())


def test_old_answer_wire_data_and_new_fields_round_trip():
    old = Answer(status='refused', text='Cloud', refusal_reason='cloud', trace_id='old')
    assert Answer.model_validate_json(old.model_dump_json()) == old
    new = Answer(status='refused', text='Need radar', refusal_reason='bands_missing', trace_id='new',
                 evidence=(Evidence(kind='text', needs_sensor=('S1',)),))
    assert Answer.model_validate_json(new.model_dump_json()) == new


def test_optical_only_registration_is_not_a_monsoon_target(tmp_path):
    a, reg, inputs = fixture('joint_cloud', tmp_path)
    reg.get('landcover').needs = ToolNeeds(sensors=('S2',))
    q = Query(text='land cover', qtype='open')
    gate = evaluate(q, a, route(q, a), reg, **inputs)
    assert gate.refusal_reason == 'cloud'
    assert gate.evidence[-1].needs_sensor == ('S1',)


def test_validated_ood_and_monsoon_never_answered_or_calibrated(tmp_path):
    status = ToolStatus(name='fake', version='v1', status='validated')
    for warning in ('out_of_distribution', 'monsoon_mode', 'out_of_region', 'ood_unknown'):
        gate = GateDecision('fake', 'S2', None, (Evidence(kind='text', summary={'warning': warning}),))
        a = assemble(result(), gate, status, trace_id=warning, calibration_path=calibration(tmp_path))
        assert a.status == 'low_confidence' and a.confidence is None


@pytest.mark.parametrize('bad', [float('inf'), float('nan'), -1e308, None])
def test_unusable_ood_observations_fail_to_uncertainty(bad, tmp_path):
    reg = registry(tmp_path)
    q = Query(text='land cover', qtype='open')
    gate = evaluate(q, scene(), route(q, scene()), reg,
                    training_stats={'B04': (1e308, 1e-308)}, band_means=({'B04': bad},))
    assert gate.passed and not gate.clean


@pytest.mark.parametrize('text', ['How expensive is this land?', 'What is this city called?',
                                  'Map next year flooding', 'Count personnel on the road'])
def test_other_non_imagery_phrasings(text, tmp_path):
    reg = registry(tmp_path)
    q = Query(text=text, qtype='open')
    assert evaluate(q, scene(), route(q, scene()), reg).refusal_reason == 'not_answerable'


def test_malformed_tool_return_refuses(tmp_path):
    reg = registry(tmp_path)
    reg.get('landcover').run = lambda *args: {'answer': 'not a ToolResult'}
    answer = run_query(Query(text='land cover', qtype='open'), scene(), {'S2': b'px'}, reg,
                       trace_id='malformed')
    assert answer.refusal_reason == 'checks_disagree' and answer.confidence is None


def test_tool_exception_refuses_and_is_traced(tmp_path):
    reg = registry(tmp_path)
    def fail(*args):
        raise RuntimeError('inference unavailable')
    reg.get('landcover').run = fail
    answer = run_query(Query(text='land cover', qtype='open'), scene(), {'S2': b'px'}, reg,
                       trace_id='exception')
    assert answer.refusal_reason == 'checks_disagree'
    assert any(e.get('error_type') == 'RuntimeError' for e in trace.records())


def test_count_and_general_do_not_get_radar_replacements(tmp_path):
    a, reg, inputs = fixture('joint_cloud', tmp_path)
    for text in ('Count cars', 'Describe this scene'):
        q = Query(text=text, qtype='open')
        gate = evaluate(q, a, route(q, a), reg, **inputs)
        assert gate.refusal_reason == 'cloud' and gate.evidence[-1].needs_sensor == ('S1',)


def test_disabled_radar_cannot_rescue_cloudy_scene(tmp_path):
    a, reg, inputs = fixture('joint_cloud', tmp_path)
    t = reg.get('flood-otsu')
    t.status = ToolStatus(name=t.name, version=t.version, status='disabled')
    q = Query(text='Map flood water', qtype='open')
    assert evaluate(q, a, route(q, a), reg, **inputs).refusal_reason == 'tool_disabled'


def test_declared_required_bands_for_custom_general_tool(tmp_path):
    reg = registry(tmp_path)
    reg.get('qwen2.5vl-3b').needs = ToolNeeds(sensors=('S2',), bands=('B12',))
    q = Query(text='Describe this scene', qtype='open')
    gate = evaluate(q, scene(), route(q, scene()), reg)
    assert gate.refusal_reason == 'bands_missing'
    assert gate.evidence[-1].summary['missing_bands'] == ['B12']
