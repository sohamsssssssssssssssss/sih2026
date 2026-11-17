"""T15d mode isolation, fixed recipe and strict diagnostic thresholds."""
import copy
import numpy as np
import pytest
import torch
from torch import nn
from data.flood import learning as l, train as t
from data.flood.debug_optical import ROWS, config_for, verdict, correction_mask
from data.flood.test_build_pack import raw, pack


def test_s2_only_matches_optical_slice_and_imputation(pack, tmp_path, monkeypatch):
    root, _ = pack; pp = l.Pack(root, expected_sha=None)
    arrays = {k: np.array(v, copy=True) for k, v in pp.split('train').items()}
    arrays['s2'][0, 0, 1, 1] = 0
    arrays['s1'][0, 0, 1, 1] = np.nan
    pp.cache['train'] = arrays
    optical = pp.inputs('train', [0], 's2')
    assert optical.shape == (1, 13, 8, 8) and np.isfinite(optical).all()
    assert optical[0, 0, 1, 1] == 0
    np.testing.assert_array_equal(optical, pp.inputs('train', [0], 's1s2')[:, 2:])
    with pytest.raises(ValueError, match='Unknown input mode'):
        pp.inputs('train', [0], 'typo')
    monkeypatch.setattr(t, 'UNet', lambda inc: nn.Conv2d(inc, 1, 1))
    cfg = {'mode': 's2', 'control': 'none', 'epochs': 2, 'batch_size': 1,
           'lr': .003, 'weight_decay': 0., 'pos_weight': 1., 'threshold': .5,
           'augmentation': False, 'schedule': 'cosine', 'seed': 17,
           'tiny_indices': [0], 'tiny_early_stop': False}
    out = t.fit(pp, cfg, tmp_path/'optical', 'tiny', torch.device('cpu'))
    assert len(out['history']) == 2 and out['selection_split'] == 'train (overfit gate only)'
    state = torch.load(tmp_path/'optical/flood-checkpoint.pt', weights_only=True)
    assert state['state_dict']['weight'].shape[1] == 13


def test_fixed_recipe_and_strict_d1():
    reference = {'train_indices': ROWS.copy(), 'config': {
        'seed': 17, 'epochs': 400, 'batch_size': 2, 'lr': .003,
        'weight_decay': 0., 'pos_weight': 1., 'threshold': .5,
        'augmentation': False, 'optimizer': 'AdamW', 'schedule': 'cosine', 'control': 'none'}}
    original = copy.deepcopy(reference)
    for mode in ('s1s2', 's2'):
        for scope in ('single', 'eight'):
            cfg = config_for(reference, mode, scope)
            assert cfg['tiny_indices'] == ([181] if scope == 'single' else ROWS)
            assert cfg['epochs'] == 400 and not cfg['tiny_early_stop']
    assert reference == original
    reference['config']['lr'] = .03
    with pytest.raises(ValueError, match='recipe changed'):
        config_for(reference, 's2', 'single')
    assert verdict(.951, .901)['D1'] == 'PASS'
    for a, b in ((.95, .99), (.99, .9), (.8, .8)):
        assert verdict(a, b)['D1'] == 'FAIL'
        assert verdict(a, b)['D2'] == 'pipeline cause by registered rule'


def test_correction_mask_excludes_nodata_and_both_wrong():
    y = np.array([[1, 0, -1], [1, 0, 1]])
    s1 = np.array([[0, 1, 0], [1, 1, 0]], dtype=bool)
    both = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
    np.testing.assert_array_equal(correction_mask(y, s1, both), [[1, 1, 0], [0, 0, 0]])
