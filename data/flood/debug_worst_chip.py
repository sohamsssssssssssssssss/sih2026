"""T15 worst-chip probes. Train-row 181 only; no held-out model selection."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy import ndimage
from data.flood import train
from data.flood.model import UNet,loss
from data.flood.learning import Pack,counts,iou,code_info
from data.flood.build_pack import assert_alignment,sha256


def component_sizes(water):
    labels,n=ndimage.label(water,structure=np.ones((3,3),dtype=np.int8))
    sizes=np.bincount(labels.ravel());sizes[0]=0
    distance=ndimage.distance_transform_edt(np.pad(water,1))[1:-1,1:-1]
    max_distance=ndimage.maximum(distance,labels,index=np.arange(1,n+1))
    thin_ids=np.flatnonzero(2*max_distance<4)+1
    total=int(water.sum())
    return {'connectivity':8,'components':int(n),'water_pixels':total,
        'water_pixels_components_under_16':int(sizes[sizes<16].sum()),
        'pct_water_components_under_16':100*int(sizes[sizes<16].sum())/total if total else None,
        'water_pixels_components_width_under_4':int(sizes[thin_ids].sum()),
        'pct_water_components_width_under_4':100*int(sizes[thin_ids].sum())/total if total else None,
        'width_definition':'component max 2*EDT-to-background pixel-center distance <4; width proxy, not exact morphological width',
        'pct_water_local_width_under_4':100*int((water&(2*distance<4)).sum())/total if total else None}


def inspect(pack,source,dest):
    r=json.loads((source/'flood-run.json').read_text());assert r['train_chip_ids']==['Spain_5678382']
    row=r['train_indices'][0];assert row==181
    a=pack.split('train');y=a['label'][row,0];mask=y!=-1
    state=torch.load(source/'flood-checkpoint.pt',map_location='cpu',weights_only=True)
    assert sha256(source/'flood-checkpoint.pt')==r['checkpoint_sha256']
    dev=train.device();m=UNet(2).to(dev);m.load_state_dict(state['state_dict'])
    x,target=train.batch(pack,'train',[row],'s1',dev)
    with torch.no_grad():
        m.train();z_train=m(x);train_prob=torch.sigmoid(z_train).cpu().numpy()[0,0]
        m.eval();z=m(x);prob=torch.sigmoid(z).cpu().numpy()[0,0]
        total=float(loss(z,target));bce=float(loss(z,target,kind='bce'))
    sweep=[{'threshold':v/10,'iou':iou(counts(prob>=v/10,y,mask))} for v in range(1,10)]
    bns=[layer for layer in m.modules() if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm)]
    out={'code':code_info(),'chip_id':'Spain_5678382','train_row':row,'checkpoint_sha256':sha256(source/'flood-checkpoint.pt'),
        'train_mode_iou':iou(counts(train_prob>=.5,y,mask)),'eval_mode_iou':iou(counts(prob>=.5,y,mask)),
        'max_train_eval_probability_difference':float(np.max(np.abs(train_prob-prob))),
        'batchnorm_layers':len(bns),'BN_momentum_None':'not applicable: no BatchNorm layers',
        'BN_recompute_running_stats':'not applicable: GroupNorm has no running statistics',
        'threshold_sweep':sweep,'best_threshold':max(sweep,key=lambda x:x['iou']),
        'last_epoch_recorded_loss':r['history'][-1]['train_loss'],
        'best_checkpoint_loss':{'total':total,'BCE':bce,'Dice':total-bce},
        'objects':component_sizes(y==1),'input_channels':list(pack.manifest['band_order']['s1']),
        'S2_used':False,'config':r['config'],'alignment':assert_alignment(pack.manifest['chips']['Spain_5678382']['metadata']),
        'native_shape':list(y.shape),'resampled':False,'target_valid_pixels':int(mask.sum()),
        'source_VH_band_description':pack.manifest['chips']['Spain_5678382']['metadata']['S1Hand']['descriptions'][1]}
    np.save(dest/'worst_probabilities.npy',prob)
    (dest/'worst-inspection.json').write_text(json.dumps(out,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    pred=prob>=.5
    errors=np.zeros((*y.shape,3),dtype=np.float32)
    errors[(pred & (y==0) & mask)]=[1,0,0]
    errors[(~pred & (y==1) & mask)]=[0,0,1]
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    vh=a['s1'][row,1];lo,hi=np.percentile(vh[np.isfinite(vh)],[2,98])
    axes[0].imshow(vh,cmap='gray',vmin=lo,vmax=hi);axes[0].set_title('VH dB')
    axes[1].imshow(y,cmap=ListedColormap(['gray','tan','royalblue']),vmin=-1,vmax=1);axes[1].set_title('Label')
    axes[2].imshow(pred,cmap='gray',vmin=0,vmax=1);axes[2].set_title('Prediction at 0.5')
    axes[3].imshow(errors);axes[3].set_title('FP red / FN blue / other black')
    for axis in axes:axis.axis('off')
    fig.suptitle('Spain_5678382 — saved U-Net best single-chip checkpoint');fig.tight_layout()
    fig.savefig(dest/'worst_chip_errors.png',dpi=180);plt.close(fig)
    print(json.dumps(out,indent=2),flush=True)
    return out


def experiment(pack,source,dest,name):
    original=json.loads((source/'flood-run.json').read_text());cfg=copy.deepcopy(original['config'])
    cfg.update({'epochs':400,'tiny_indices':original['train_indices'],'tiny_early_stop':False,'weight_decay':0.,'architecture':'unet'})
    if name.startswith('low_'):cfg['lr']/=10
    if name.startswith('high_'):cfg['lr']*=10
    cfg['loss_kind']='bce' if name.endswith('_bce') else 'bce_dice'
    if name=='full_resolution':
        cfg['architecture']='full_resolution';cfg['tiny_early_stop']=True;cfg['tiny_stop_iou']=.95
    budget=dest/'worst-compute-budget.json'
    if not budget.exists():budget.write_text(json.dumps({'started_unix':time.time(),'wall_budget_s':3600},indent=2)+'\n')
    timer=json.loads(budget.read_text());remaining=timer['wall_budget_s']-(time.time()-timer['started_unix'])
    if remaining<60:raise ValueError('STOP: one-hour diagnostic budget exhausted')
    cfg['tiny_max_runtime_s']=min(1800 if name=='full_resolution' else 180,max(1,remaining-600))
    try:
        return train.fit(pack,cfg,dest/name,'tiny',train.device())
    except ValueError as exc:
        if 'non-finite training loss' not in str(exc):raise
        (dest/(name+'-failure.json')).write_text(json.dumps({'config':cfg,'error':str(exc)},indent=2)+'\n')
        print(str(exc),flush=True)
        return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['inspect','low_combo','high_combo','base_bce','low_bce','high_bce','full_resolution'])
    p.add_argument('--pack',default=str(Path.home()/'satquery-data/flood/packs/sen1floods11-hand-v1'))
    p.add_argument('--source',type=Path,default=Path('outputs/t15-debug/single'))
    p.add_argument('--out',type=Path,default=Path('outputs/t15-debug'))
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    pack=Pack(args.pack)
    if args.action=='inspect':return inspect(pack,args.source,args.out)
    return experiment(pack,args.source,args.out,args.action)


if __name__=='__main__':main()
