"""T15 tiny synthetic tests; no network, no held-out choices, no GPU required."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from data.flood import learning as l, train as t
from data.flood.model import UNet,loss
from data.flood.test_build_pack import raw,pack  # shared synthetic pack fixtures


def test_nodata_never_enters_loss_or_metric():
    z=torch.tensor([[[[.5,1.],[-.5,7.]]]],requires_grad=True)
    y=torch.tensor([[[[1.,0.],[1.,-1.]]]])
    a=loss(z,y);a.backward()
    assert z.grad[0,0,1,1]==0
    changed=z.detach().clone();changed[0,0,1,1]=float('nan')
    assert loss(changed,y).item()==pytest.approx(a.item())
    target=y.numpy()[0,0];mask=target!=-1
    p=np.array([[True,True],[False,True]])
    assert l.counts(p,target,mask)==[1,1,1,3]
    p[-1,-1]=False
    assert l.counts(p,target,mask)==[1,1,1,3]
    assert loss(z,torch.full_like(y,-1)).item()==0


def test_iou_hand_example_and_chip_bootstrap():
    assert l.iou([1,1,1,3])==pytest.approx(1/3)
    assert l.iou([0,0,0,0]) is None
    assert l.iou([0,0,3,3])==0
    rows=[{'counts':[1,0,0,1]},{'counts':[0,1,0,1]}]
    result=l.summarize(rows)
    assert result['water_iou']==.5 and result['ci95']==[0.,1.]
    assert result==l.summarize(rows) and result['bootstrap_reps']==2000


def test_masks_input_imputation_coverage_and_blank(pack):
    root,_=pack;pp=l.Pack(root,expected_sha=None)
    arrays={k:np.array(v,copy=True) for k,v in pp.split('train').items()}
    arrays['s1'][0,1,1,1]=np.nan
    arrays['otsu'][0,0,1,2]=-1
    arrays['s2'][0,2,1,3]=0
    pp.cache['train']=arrays
    mm=l.masks(arrays,0)
    for y,x in ((1,1),(1,2),(1,3)):
        assert mm['primary'][y,x] and not mm['strict'][y,x]
    assert not mm['primary'][0,0]
    x=pp.inputs('train',[0],'s1s2')
    assert np.isfinite(x).all() and x[0,1,1,1]==0 and x[0,4,1,3]==0
    assert pp.inputs('train',[0],'s1').shape==(1,2,8,8)
    assert np.count_nonzero(pp.inputs('train',[0],'s1s2','blank'))==0
    c=l.coverage(arrays)
    assert c['nonfinite_vh']['count']==1 and c['s2_nodata']['count']==1 and c['missing_otsu']['count']==1


def test_selection_valid_only_and_labels_train_only():
    seen=[]
    def evaluate(split,state):
        seen.append(split)
        assert split=='valid'
        return state['metric']
    result=t.select_epoch(evaluate,[(1,{'metric':.3}),(2,{'metric':.7}),(3,{'metric':.4})])
    assert result['epoch']==2 and seen==['valid']*3
    ids=np.arange(8)
    mapping=t.labels_for_training(None,ids,'shuffle')
    assert sorted(mapping.values())==ids.tolist() and list(mapping.values())!=ids.tolist()
    assert t.labels_for_training(None,ids,'none')==dict(zip(ids,ids))


def test_small_unet_size_and_forward():
    model=UNet(2)
    assert 1_000_000<=sum(p.numel() for p in model.parameters())<=8_000_000
    assert model(torch.zeros(1,2,32,32)).shape==(1,1,32,32)
    assert UNet(15)(torch.zeros(1,15,32,32)).shape==(1,1,32,32)


def test_output_schema_control_wiring_and_selection_isolation(pack,tmp_path,monkeypatch):
    root,_=pack;pp=l.Pack(root,expected_sha=None)
    class Tiny(nn.Module):
        def __init__(self,inc):
            super().__init__();self.layer=nn.Conv2d(inc,1,1)
        def forward(self,x):
            assert torch.count_nonzero(x)==0
            return self.layer(x)
    monkeypatch.setattr(t,'UNet',Tiny)
    cfg={'mode':'s1','control':'blank','epochs':2,'batch_size':2,'lr':.001,'weight_decay':0.,
         'pos_weight':1.,'threshold':.5,'augmentation':False,'schedule':'cosine','seed':17,
         'baselines':str(tmp_path/'baseline.json')}
    baseline={'pack_manifest_sha256':pp.sha,'metrics':{'B0':{m:{'valid':{'water_iou':0.}} for m in l.MASKS}}}
    Path(cfg['baselines']).write_text(json.dumps(baseline))
    calls=[];original=t.evaluate
    def tracking(model,pack,split,*args):
        calls.append(split)
        if split in ('test','bolivia'):
            assert calls.count('valid')>=2
        return original(model,pack,split,*args)
    monkeypatch.setattr(t,'evaluate',tracking)
    out=t.fit(pp,cfg,tmp_path/'out','train',torch.device('cpu'))
    saved=json.loads((tmp_path/'out/flood-run.json').read_text())
    assert out['selection_split']=='valid' and calls.count('test')==calls.count('bolivia')==1
    assert {'config','checkpoint_sha256','norm_stats','history','per_chip','metrics','coverage','pack_manifest_sha256','code'}<=saved.keys()
    assert set(saved['metrics'])==set(l.MASKS) and saved['config']['control']=='blank'
    assert len(saved['checkpoint_sha256'])==64 and (tmp_path/'out/config.yaml').exists()


def test_mask_verdict_change_flag():
    def metrics(lower,upper):
        return {g:{'ci95':[lower,upper]} for g in ('test','India-test')}
    base={'B1':{m:metrics(.3,.5) for m in l.MASKS}}
    s1={'primary':metrics(.6,.8),'strict':metrics(.2,.6)}
    s2={m:metrics(.85,.95) for m in l.MASKS}
    result=l.compare_hypotheses(s1,s2,base)
    assert result['mask_verdict_changes']==['H1','H2']
    assert result['verdicts']['primary']['H3'] and result['verdicts']['strict']['H3']


def test_timing_keeps_all_weights_unchanged(pack,tmp_path,monkeypatch):
    root,_=pack;pp=l.Pack(root,expected_sha=None)
    monkeypatch.setattr(t,'UNet',lambda inc:nn.Conv2d(inc,1,1))
    cfg={'mode':'s1','control':'none','epochs':30,'batch_size':2,'lr':.001,'pos_weight':1.,'augmentation':True}
    result=t.benchmark(pp,cfg,tmp_path/'timing',torch.device('cpu'))
    assert all(v['weights_unchanged'] and v['optimizer_steps']==0 for v in result['modes'].values())


def test_exact_pixelwise_collision_bound():
    from itertools import product
    from data.flood.debug_overfit import pixelwise_bound
    pos=np.array([5,2,0,3]);neg=np.array([0,7,4,1])
    brute=max(sum(p*s for p,s in zip(pos,selection))/(sum(pos)+sum(n*s for n,s in zip(neg,selection)))
              for selection in product((0,1),repeat=len(pos)))
    assert pixelwise_bound(pos,neg)==pytest.approx(brute)
    assert pixelwise_bound(np.array([0]),np.array([5])) is None


def test_explicit_single_train_chip_and_fixed_epoch_budget(pack,tmp_path,monkeypatch):
    root,_=pack;pp=l.Pack(root,expected_sha=None)
    monkeypatch.setattr(t,'UNet',lambda inc:nn.Conv2d(inc,1,1))
    cfg={'mode':'s1','control':'none','epochs':3,'batch_size':1,'lr':.001,'weight_decay':0.,
         'pos_weight':1.,'threshold':.5,'augmentation':False,'schedule':'cosine','seed':17,
         'tiny_indices':[0],'tiny_early_stop':False}
    out=t.fit(pp,cfg,tmp_path/'single','tiny',torch.device('cpu'))
    assert out['train_indices']==[0] and len(out['history'])==3
    assert out['selection_split']=='train (overfit gate only)'
    assert list(out['metrics']['primary'])==['train','train/event/India']
