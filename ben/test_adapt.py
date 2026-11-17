"""T13 leakage, freezing, controls, saved-baseline reproduction and schema checks."""
import copy
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

pytest.importorskip('pyarrow')
import pandas as pd
import torch
from torch import nn
from ben import adapt, landcover as lc, transfer as tr
from ben.test_transfer import BEN19


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2,4,1)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4,19)
    def forward(self,x):
        return self.fc(self.bn(self.conv(x)).mean((2,3)))


def fixture_data():
    rng = np.random.default_rng(17)
    regions = sorted(set(adapt.FOLDS['A'][0]+adapt.FOLDS['A'][1]))
    meta = pd.DataFrame({'idx':np.arange(120),'region':np.repeat(regions,20),'group':'india'})
    arrays = {'s1':rng.normal(-10,4,(120,2,8,8)).astype('float32')}
    labels = (rng.random((120,6)) < .95).astype('float32')
    labels[:,5] = 0
    return arrays,labels,meta


def test_folds_disjoint_and_each_india_patch_tested_once():
    _,_,meta = fixture_data()
    tested = []
    for fold in adapt.FOLDS:
        train,val,test = adapt.fold_indices(meta,fold)
        assert len(train)==48 and len(val)==12 and len(test)==60
        assert not set(train)&set(test) and not set(val)&set(test) and not set(train)&set(val)
        assert set(meta.iloc[train].region).isdisjoint(meta.iloc[test].region)
        assert all((meta.iloc[train].region == r).sum()==16 for r in adapt.FOLDS[fold][0])
        tested.extend(test)
    assert sorted(tested)==list(range(120))
    assert all(np.array_equal(a,b) for a,b in zip(adapt.fold_indices(meta,'A'),adapt.fold_indices(meta,'A')))


def test_no_label_renorm_leaves_weights_unchanged_and_excludes_val_test():
    arrays,_,meta=fixture_data(); model=Tiny(); before=copy.deepcopy(model.state_dict())
    train,val,test=adapt.fold_indices(meta,'A')
    stats=lc.train_stats(arrays,train)
    changed={'s1':arrays['s1'].copy()}; changed['s1'][np.r_[val,test]]=999
    assert stats==lc.train_stats(changed,train)
    lc.predict(model,arrays,stats,test,torch.device('cpu'),16)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())


@pytest.mark.parametrize('arm', ['head', 'full'])
def test_training_freezes_required_weights_and_masks_inactive_rows(arm):
    arrays,y,meta=fixture_data(); train,val,_=adapt.fold_indices(meta,'A')
    model=Tiny(); before=copy.deepcopy(model.state_dict())
    members,_=tr.coarse_members(BEN19)
    result=adapt.fit(model,arrays,lc.train_stats(arrays,train),y,train,val,members,arm,2,16,torch.device('cpu'))
    assert 'wetland' not in result['loss_classes']
    if arm == 'head':
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items() if not k.startswith('fc.'))
    else:
        assert not torch.equal(before['conv.weight'], model.conv.weight)
    assert not torch.equal(before['fc.weight'],model.fc.weight)
    for i in result['inactive_classifier_rows']:
        assert torch.equal(before['fc.weight'][i],model.fc.weight[i])
        assert torch.equal(before['fc.bias'][i],model.fc.bias[i])
    assert BEN19.index('Beaches, dunes, sands') in result['inactive_classifier_rows']


def test_shared_coarse_pooling_is_differentiable():
    members,_=tr.coarse_members(BEN19)
    x=torch.randn(8,19,requires_grad=True)
    pooled=tr.coarse_logits(x,members)
    assert np.allclose(torch.sigmoid(pooled).detach().numpy(),tr.to_coarse(x.detach().numpy(),members),atol=1e-7)
    pooled.sum().backward()
    assert torch.all(x.grad[:,BEN19.index('Beaches, dunes, sands')]==0)


def test_shuffle_is_confined_to_train_and_val():
    _,y,meta=fixture_data(); train,val,test=adapt.fold_indices(meta,'A')
    shuffled,perms=adapt.shuffled_labels(y,train,val)
    assert np.array_equal(shuffled[test],y[test])
    assert set(perms['train'])==set(train) and set(perms['val'])==set(val)
    assert perms['train']!=train.tolist()
    assert np.array_equal(shuffled[train],y[perms['train']])
    assert np.array_equal(shuffled[val],y[perms['val']])


def test_saved_baseline_reproduction_and_mismatch_stop():
    # Real artifacts are read-only; CI without them skips this integration test explicitly.
    pack=Path(os.environ.get('T13_WC_PACK',Path.home()/'ben-work/wc/pack'))
    runs=Path(os.environ.get('T13_RUNS',Path.home()/'ben-work/transfer-run-1'))
    if not (runs/'ben-transfer.json').exists() or not pack.exists():
        pytest.skip('real transfer artifacts unavailable')
    _,y,meta,info=tr.load_wc(pack)
    ref=json.loads((runs/'ben-transfer.json').read_text())
    for mode in lc.MODES:
        ckpt=torch.load(runs/f'ben-landcover-{mode}-checkpoint.pt',map_location='cpu',weights_only=True)
        members,_=tr.coarse_members(ckpt['classes'])
        logits=np.load(runs/f'wc-{mode}-logits.npy')
        checks=adapt.check_baseline(y,meta,info,logits,members,ref['modes'][mode])
        assert len(checks)==14 and all(abs(v['difference'])<=1e-4 for v in checks.values())
        bad=copy.deepcopy(ref['modes'][mode]); bad['groups']['india']['macro_map']+=.001
        with pytest.raises(ValueError,match='STOP: arm none reproduction failed'):
            adapt.check_baseline(y,meta,info,logits,members,bad)


def test_output_schema_and_shuffled_control_wiring(tmp_path,monkeypatch):
    arrays,y,meta=fixture_data()
    members,_=tr.coarse_members(BEN19)
    stats=lc.train_stats(arrays,np.arange(120))
    model=Tiny(); ckpt={'classes':BEN19,'in_channels':2,'inputs':['s1'],'preprocess_stats':stats,'state_dict':model.state_dict()}
    runs=tmp_path/'runs'; runs.mkdir()
    torch.save(ckpt,runs/'ben-landcover-s1-checkpoint.pt')
    np.save(runs/'wc-s1-logits.npy',np.zeros((120,19)))
    (runs/'ben-transfer.json').write_text(json.dumps({'wc_manifest_sha256':'manifest','modes':{'s1':{}}}))
    pack=tmp_path/'pack'; pack.mkdir(); np.save(pack/'fractions.npy',y*.1)
    monkeypatch.setattr(tr,'verify_wc_manifest',lambda p:'manifest')
    monkeypatch.setattr(tr,'load_wc',lambda p:(arrays,y,meta,{'coarse_classes':list(members),'present_fraction':.05}))
    monkeypatch.setattr(adapt,'check_baseline',lambda *a:{'europe_common':{'actual':.8}})
    monkeypatch.setattr(lc,'build_model',lambda *a:Tiny())
    monkeypatch.setattr(tr,'BOOT',10)
    observed=[]
    original=adapt.fit
    def recording(model,arrays,stats,labels,train,val,members,arm,*rest,**kw):
        if arm=='shuffle':
            shuffled,_=adapt.shuffled_labels(y,train,val)
            assert np.array_equal(labels,shuffled)
            observed.append(arm)
        return original(model,arrays,stats,labels,train,val,members,arm,*rest,**kw)
    monkeypatch.setattr(adapt,'fit',recording)
    out=adapt.main(['--wc-pack',str(pack),'--runs',str(runs),'--out-dir',str(tmp_path/'out'),
                    '--modes','s1','--epochs','2','--limit-per-region','20','--device','cpu','--batch-size','16'])
    saved=json.loads((tmp_path/'out/ben-adapt.json').read_text())
    assert observed==['shuffle','shuffle']
    assert saved['schema_version']==1 and saved['smoke_only'] and saved['seed']==17
    assert {'config','code','wc_manifest_sha256','reference_sha256','runtime_s'}<=saved.keys()
    m=saved['modes']['s1']; assert len(m['checkpoint_sha256'])==64
    for f in m['folds'].values():
        assert set(f['arms'])==set(adapt.ARMS)
        for a in f['arms'].values():
            assert len(a['groups'])==4
            assert a['validation_classes'] == [c for c,j in zip(adapt.COMMON,range(5)) if y[f['val_idx'],j].sum() > 0]
            assert all(g['min_positives'] == 10 for g in a['groups'].values())
            assert {'macro_map','macro_map_ci95','prevalence_macro_map','shuffled_macro_map'}<=a['groups']['pooled'].keys()
    assert m['out_of_fold']['none']['n']==120
    assert m['hypotheses']['H1']['any_arm_pass'] is None
    assert set(m['out_of_fold']) == set(adapt.ARMS)
    assert all('selected_arm_from_validation' not in f for f in m['folds'].values())
    assert saved['config']['arm_selection'] is False
    assert (tmp_path/'out/ben-adapt.md').exists()


def test_kaggle_cell_contract():
    text=Path(adapt.__file__).with_name('run_adapt_gpu.txt').read_text()
    assert text.startswith('%%bash\n') and text.rstrip().endswith('echo "CELL DONE"')
    assert 'set -euo pipefail' in text and 'trap ' in text and 'ben-transfer-run-1' in text
    assert '/kaggle/tmp' not in text
    for name in ('ben/adapt.py','fractions.npy','ben-transfer.json','T13_CODE_COMMIT.txt'):
        assert name in text
    subprocess.run(['bash','-n'],input='\n'.join(text.splitlines()[1:]),text=True,check=True)


def test_validation_any_positive_and_empty_hard_stop():
    _,y,meta = fixture_data()
    _,val,_ = adapt.fold_indices(meta,'A')
    members,_ = tr.coarse_members(BEN19)
    y[val] = 0
    with pytest.raises(ValueError,match='zero common classes with any positive'):
        adapt.validation_eligibility(y,val,members)
    y[val[0],0] = 1
    _,keep,names = adapt.validation_eligibility(y,val,members)
    assert names == ['trees'] and keep.tolist() == [True,False,False,False,False]
    # The reporting rule remains >=10 even when epoch selection can use one positive.
    metric = adapt.scored(y[val,:5],np.ones((len(val),5)))
    assert metric['classes_used'] == []


def test_default_train_seed_is_t13_seed_and_reproduces_its_shuffle():
    _,y,meta=fixture_data(); train,val,_=adapt.fold_indices(meta,'A')
    default,dperm=adapt.shuffled_labels(y,train,val)
    explicit,eperm=adapt.shuffled_labels(y,train,val,seed=lc.SEED)
    assert dperm==eperm and np.array_equal(default,explicit)
    assert adapt.lc.SEED == 17


def test_train_seed_changes_shuffle_only_within_splits():
    _,y,meta=fixture_data(); train,val,test=adapt.fold_indices(meta,'A')
    s1,p1=adapt.shuffled_labels(y,train,val,seed=1)
    s2,p2=adapt.shuffled_labels(y,train,val,seed=2)
    assert p1!=p2 and not np.array_equal(s1,s2)
    for shuffled,perms in ((s1,p1),(s2,p2)):
        assert np.array_equal(shuffled[test],y[test])
        assert set(perms['train'])==set(train) and set(perms['val'])==set(val)


def test_seed_everything_is_reproducible_and_seed_sensitive():
    adapt.seed_everything(1); a=torch.rand(4)
    adapt.seed_everything(1); b=torch.rand(4)
    adapt.seed_everything(2); c=torch.rand(4)
    assert torch.equal(a,b) and not torch.equal(a,c)


def test_fold_split_does_not_depend_on_train_seed():
    _,_,meta=fixture_data()
    before=[x.tolist() for x in adapt.fold_indices(meta,'A')]
    adapt.seed_everything(99); torch.rand(1)
    after=[x.tolist() for x in adapt.fold_indices(meta,'A')]
    assert before==after


def test_shuffle_fixed_keeps_final_epoch_without_validation_choice():
    arrays,y,meta=fixture_data(); train,val,_=adapt.fold_indices(meta,'A')
    members,_=tr.coarse_members(BEN19)
    labels,_=adapt.shuffled_labels(y,train,val)
    model=Tiny()
    result=adapt.fit(model,arrays,lc.train_stats(arrays,train),labels,train,val,members,'shuffle-fixed',3,16,torch.device('cpu'),select_by_val=False)
    assert result['best_epoch']==3 and len(result['history'])==3
    assert result['best_val_map']==result['history'][-1]['val_map']
    assert 'shuffle-fixed' in adapt.ARMS and 'bn-only' in adapt.ARMS


def test_seed_repeat_cell_contract():
    text=(Path(adapt.__file__).parent/'run_adapt_seeds_gpu.txt').read_text()
    assert text.splitlines()[0]=='%%bash' and text.rstrip().endswith('echo "CELL DONE"')
    for needle in ('set -euo pipefail', 'trap ', 'for S in 1 2 3', '--train-seed "$S"', 'seed-$S'):
        assert needle in text, needle
    code='\n'.join(l for l in text.splitlines() if not l.lstrip().startswith('#'))
    assert not re.search(r'/kaggle/input/[A-Za-z0-9]',code)


def test_bn_only_changes_only_batchnorm_statistics():
    arrays,y,meta=fixture_data(); train,_,_=adapt.fold_indices(meta,'A')
    model=Tiny(); before=copy.deepcopy(model.state_dict())
    stats=lc.train_stats(arrays,train)
    result=adapt.bn_recalibrate(model,arrays,stats,train,2,16,torch.device('cpu'))
    after=model.state_dict()
    assert torch.equal(before['conv.weight'],after['conv.weight']) and torch.equal(before['fc.weight'],after['fc.weight'])
    assert torch.equal(before['fc.bias'],after['fc.bias'])
    assert not torch.equal(before['bn.running_mean'],after['bn.running_mean'])
    assert all(not p.requires_grad for p in model.parameters())
    assert result['best_epoch']==2 and not model.training


def test_prediction_archive_preserves_test_alignment_and_all_arms(tmp_path):
    meta = pd.DataFrame({'idx': [0, 1, 2], 'region': ['punjab', 'delhi', 'assam']})
    indices = np.array([2, 0])
    scores = np.array([[.1, .2, .3, .4, .5], [.9, .8, .7, .6, .5]], dtype=np.float32)
    labels = np.array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]])
    blocks = [adapt.prediction_block(meta,indices,'A','s2',arm,scores,labels) for arm in adapt.ARMS]
    target = tmp_path/'predictions.npz'
    info = adapt.save_test_predictions(target,blocks,3,'a'*64,'b'*40)
    with np.load(target,allow_pickle=False) as saved:
        assert saved['classes'].tolist() == adapt.COMMON
        assert saved['patch_id'][:2].tolist() == ['worldcover:2', 'worldcover:0']
        assert saved['patch_idx'][:2].tolist() == [2, 0]
        assert saved['scores'].shape == (2*len(adapt.ARMS), 5)
        assert np.array_equal(saved['scores'][:2],scores)
        assert np.array_equal(saved['labels'][:2],labels)
        assert set(saved['arm']) == set(adapt.ARMS)
        assert set(saved['fold']) == {'A'} and set(saved['mode']) == {'s2'}
        assert int(saved['seed']) == 3 and str(saved['wc_manifest_sha256']) == 'a'*64
        assert str(saved['code_commit']) == 'b'*40
    assert info['sha256'] == lc.sha256(target) and info['rows'] == 2*len(adapt.ARMS)
    with pytest.raises(FileExistsError):
        adapt.save_test_predictions(target,blocks,3,'a'*64,'b'*40)
    with pytest.raises(ValueError,match='duplicate'):
        adapt.save_test_predictions(tmp_path/'duplicate.npz',[blocks[0],blocks[0]],3,'a'*64,'b'*40)
    with pytest.raises(ValueError,match='shape'):
        adapt.prediction_block(meta,indices,'A','s2','full',scores[:1],labels)
