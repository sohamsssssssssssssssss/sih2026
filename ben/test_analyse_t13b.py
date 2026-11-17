"""Summary-analysis checks only; no model, fitting or inference."""
import pytest
import numpy as np
from ben.analyse_t13b import (ARMS, CLASSES, MODES, NPZ, contrast, paired_bootstrap, positive_ci,
                             read_predictions, required_all, sha, unanimity)


@pytest.mark.parametrize('votes,expected', [
    (['PASS']*3,'PASS'), (['FAIL']*3,'FAIL'),
    (['PASS','PASS','FAIL'],'INCONCLUSIVE'),
    (['FAIL','FAIL','PASS'],'INCONCLUSIVE'),
    (['FAIL','FAIL','INCONCLUSIVE'],'INCONCLUSIVE'),
])
def test_per_seed_deviation_requires_unanimity(votes, expected):
    assert unanimity(votes)==expected


def test_zero_bound_fails_and_missing_ci_is_not_a_pass():
    assert positive_ci(.1,[0,.2])=='FAIL'
    assert positive_ci(0,[.01,.2])=='FAIL'
    assert positive_ci(.1,None)=='INCONCLUSIVE'
    assert positive_ci(.1,[.01,.2])=='PASS'
    assert required_all(['PASS','INCONCLUSIVE','FAIL'])=='FAIL'


def test_missing_paired_ci_is_not_inferred_from_marginal_intervals():
    runs=[]
    for seed,gain in ((1,.2),(2,.3),(3,.4)):
        runs.append({'seed':seed,'modes':{'s2':{
            'out_of_fold':{'full':{'macro_map':.4+gain,'macro_map_ci95':[.6,.9]},
                           'none':{'macro_map':.4,'macro_map_ci95':[.35,.45]},
                           'shuffle':{'macro_map':.3,'macro_map_ci95':[.2,.35]}},
            'hypotheses':{'H1':{'per_arm':{'full':{'gain':gain,'gain_ci95':[gain-.05,gain+.05]}}}}
        }}})
    registered=contrast(runs,'s2','none')
    assert registered['verdict']=='PASS' and registered['mean']==pytest.approx(.3)
    for sample,gain in zip(registered['seeds'],(.2,.3,.4)):
        assert sample['ci95']==pytest.approx([gain-.05,gain+.05])
    shuffled=contrast(runs,'s2','shuffle')
    assert shuffled['verdict']=='INCONCLUSIVE'
    assert all(s['ci95'] is None for s in shuffled['seeds'])


def test_missing_seed_cannot_silently_pass():
    with pytest.raises(ValueError,match='three'):
        unanimity(['PASS','PASS'])


def _stub_run(seed, path):
    folds = {f: {'arms': {a: {'groups': {'pooled': {'n': 6}}} for a in ARMS}} for f in ('A', 'B')}
    return {'seed': seed, 'code': {'git_sha': 'c' * 40}, 'wc_manifest_sha256': 'w' * 64,
            'test_predictions': {'file': path.name, 'sha256': sha(path)},
            'modes': {m: {'folds': folds} for m in MODES}}


def _write_seed(folder, seed):
    """Scores depend only on (seed, patch idx, class), so the seed mean is (2 + idx + class)/100 for seeds 1-3."""
    d = folder/f'seed-{seed}'
    d.mkdir(parents=True)
    rows = {k: [] for k in ('patch_id', 'patch_idx', 'fold', 'mode', 'arm', 'scores', 'labels')}
    for mode in MODES:
        for fold, idx in (('A', np.arange(0, 6)), ('B', np.arange(6, 12))):
            for arm in ARMS:
                order = idx[::-1]  # stored out of order on purpose
                rows['patch_id'] += [f'worldcover:{i}' for i in order]
                rows['patch_idx'] += list(order)
                rows['fold'] += [fold] * len(order)
                rows['mode'] += [mode] * len(order)
                rows['arm'] += [arm] * len(order)
                rows['scores'] += [(seed + i + np.arange(5)) / 100 for i in order]
                rows['labels'] += [(np.arange(5) + i) % 2 for i in order]
    path = d/NPZ.format(seed)
    np.savez_compressed(path, patch_id=np.array(rows['patch_id']), patch_idx=np.array(rows['patch_idx'], dtype=np.int64),
                        fold=np.array(rows['fold']), mode=np.array(rows['mode']), arm=np.array(rows['arm']),
                        scores=np.array(rows['scores'], dtype=np.float32), labels=np.array(rows['labels'], dtype=np.uint8),
                        classes=np.array(CLASSES), seed=np.array(seed), wc_manifest_sha256=np.array('w' * 64),
                        code_commit=np.array('c' * 40))
    return path


def test_npz_reader_aligns_patches_and_averages_seeds(tmp_path):
    runs = [_stub_run(s, _write_seed(tmp_path, s)) for s in (1, 2, 3)]
    preds = read_predictions(tmp_path, runs)
    assert len(preds) == len(MODES) * 2 * len(ARMS)
    entry = preds[('s2', 'A', 'full')]
    assert entry['idx'].tolist() == list(range(6))
    assert np.allclose(entry['scores'], (2 + entry['idx'][:, None] + np.arange(5)[None, :]) / 100)


def test_npz_reader_absent_is_none_and_partial_set_stops(tmp_path):
    assert read_predictions(tmp_path, []) is None
    paths = [_write_seed(tmp_path, s) for s in (1, 2, 3)]
    runs = [_stub_run(s, p) for s, p in zip((1, 2, 3), paths)]
    paths[2].rename(tmp_path/'moved.npz')
    with pytest.raises(ValueError, match='Partial'):
        read_predictions(tmp_path, runs)


def test_paired_bootstrap_identical_arms_zero_and_better_arm_positive():
    rng = np.random.default_rng(1)
    y = (rng.random((200, 5)) < .3).astype(int)
    good = y*.8 + rng.random((200, 5))*.2
    bad = rng.random((200, 5))
    same = paired_bootstrap(y, good, good)
    assert same['seed_avg_point'] == 0 and same['ci95'] == [0, 0]
    better = paired_bootstrap(y, good, bad)
    assert better['seed_avg_point'] > 0 and better['ci95'][0] > 0
