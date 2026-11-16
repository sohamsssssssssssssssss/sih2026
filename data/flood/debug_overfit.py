"""Training-only T15 stall diagnostics. Never trains on or selects using held-out data."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch

from data.flood.learning import Pack,code_info,grouped,MASKS,metric_table
from data.flood.build_pack import assert_alignment,sha256
from data.flood.model import UNet
from data.flood import train


def pixelwise_bound(positive,negative):
    """Exact best IoU when each unique input vector must receive one binary prediction."""
    if not positive.sum():return None
    ratios=np.divide(positive,negative,out=np.full(len(positive),np.inf),where=negative>0)
    order=np.argsort(-ratios,kind='stable')
    return float(np.max(np.cumsum(positive[order])/(positive.sum()+np.cumsum(negative[order]))))


def inspect(pack,original,dest):
    r=json.loads((original/'flood-run.json').read_text());arrays=pack.split('train');index=pack.rows('train')
    expected=np.random.default_rng(17).choice(np.arange(len(index)),8,replace=False)
    assert r['train_indices']==expected.tolist(),'STOP: original tiny sample differs from seed-17 sample'
    model=UNet(2)
    checkpoint=torch.load(original/'flood-checkpoint.pt',map_location='cpu',weights_only=True)
    assert sha256(original/'flood-checkpoint.pt')==r['checkpoint_sha256']
    model.load_state_dict(checkpoint['state_dict'])
    result=[];all_positive=[];all_negative=[]
    for i in r['train_indices']:
        row=index.iloc[i];y=arrays['label'][i,0];valid=y!=-1;x=pack.inputs('train',[i],'s1')[0]
        _,inverse=np.unique(x[:,valid].T,axis=0,return_inverse=True)
        positive=np.bincount(inverse,weights=(y[valid]==1)).astype(np.int64)
        total=np.bincount(inverse);negative=total-positive;mixed=(positive>0)&(negative>0)
        old=next(q for q in r['per_chip']['primary'] if q['chip_id']==row.chip_id)
        imputed=(~np.isfinite(arrays['s1'][i])).any(0)
        info={'chip_id':row.chip_id,'row':i,'best_epoch_iou':old['water_iou'],
              'water_pct_valid':float((y[valid]==1).mean()*100),'target_nodata_pct':float((~valid).mean()*100),
              'imputed_input_spatial_pct':float(imputed.mean()*100),
              'imputed_pct_target_valid':float((imputed&valid).sum()/valid.sum()*100),
              'opposite_label_identical_input_pixels':int(total[mixed].sum()),'conflicting_input_groups':int(mixed.sum()),
              'pixelwise_optimal_iou':pixelwise_bound(positive,negative)}
        result.append(info);all_positive.append(positive);all_negative.append(negative)
        print(json.dumps(info),flush=True)
    finite=[x for x in result if x['best_epoch_iou'] is not None]
    out={'code':code_info(),'original_commit':r['code'],'original_checkpoint_sha256':r['checkpoint_sha256'],
         'actual_config':r['config'],'norm_stats':r['norm_stats'],'input_channels':2,
         'dropout_modules':sum(isinstance(m,(torch.nn.Dropout,torch.nn.Dropout2d)) for m in model.modules()),
         'batchnorm_modules':sum(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()),
         'mode_contract':{'train_epoch':'model.train()','evaluation':'model.eval(); no gradients',
                          'normalization':'GroupNorm, identical training/evaluation behavior'},
         'loss_weights':{'BCE':1.,'Dice':1.,'pos_weight':r['config']['pos_weight']},
         'per_chip':result,'pooled_pixelwise_optimal_iou':pixelwise_bound(np.concatenate(all_positive),np.concatenate(all_negative)),
         'ceiling_scope':'Exact pointwise-input collision limit; not a CNN context ceiling. A value above .9 demonstrates these collisions alone do not force failure.',
         'true_unet_ceiling_below_gate':False,'worst_chip':min(finite,key=lambda x:x['best_epoch_iou'])['chip_id']}
    (dest/'inspection.json').write_text(json.dumps(out,indent=2)+'\n')
    return out


def overlays(pack,inspection,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    chips=[inspection['worst_chip'],'India_956930']
    observations=[]
    for chip in chips:
        row=next(c['row'] for c in inspection['per_chip'] if c['chip_id']==chip)
        a=pack.split('train');vh=a['s1'][row,1];label=a['label'][row,0]
        metadata=pack.manifest['chips'][chip]['metadata']
        alignment=assert_alignment(metadata)
        lo,hi=np.percentile(vh[np.isfinite(vh)],[2,98])
        fig,ax=plt.subplots(1,3,figsize=(13,4))
        ax[0].imshow(vh,cmap='gray',vmin=lo,vmax=hi,interpolation='nearest');ax[0].set_title('VH dB')
        ax[1].imshow(label,cmap=ListedColormap(['gray','tan','royalblue']),vmin=-1,vmax=1,interpolation='nearest');ax[1].set_title('Label: gray invalid / tan dry / blue water')
        ax[2].imshow(vh,cmap='gray',vmin=lo,vmax=hi,interpolation='nearest')
        ax[2].imshow(np.ma.masked_where(label!=1,label),cmap=ListedColormap(['cyan']),alpha=.4,vmin=0,vmax=1,interpolation='nearest')
        if (label==1).any() and (label!=1).any():ax[2].contour(label==1,levels=[.5],colors=['yellow'],linewidths=.5)
        ax[2].set_title('Water overlay / yellow label boundary')
        for panel in ax:panel.axis('off')
        fig.suptitle(chip+' — native arrays, no resampling');fig.tight_layout()
        file=dest/(chip+'-alignment.png');fig.savefig(file,dpi=160);plt.close(fig)
        observations.append({'chip_id':chip,'file':str(file),'alignment':alignment,'resampled':False})
    (dest/'alignment.json').write_text(json.dumps(observations,indent=2)+'\n')


def b2_check(baseline,dest):
    b=json.loads(baseline.read_text());measurements={}
    records=b['per_chip']
    for mask in MASKS:
        measurements[mask]={}
        for group,condition in [('water_lt_1pct',lambda x:x<.01),('water_ge_1pct',lambda x:x>=.01)]:
            allowed=set()
            for row in records['B0']['primary']:
                _,_,water,n=row['counts']
                if n and condition(water/n):allowed.add(row['chip_id'])
            measurements[mask][group]={arm:grouped([r for r in records[arm][mask] if r['chip_id'] in allowed]) for arm in ('B1','B2')}
        # A positive value means B2 has MORE FP or FN than B1 in that stratum.
        measurements[mask]['error_excess']={}
        for group in ('water_lt_1pct','water_ge_1pct'):
            rr=measurements[mask][group]
            measurements[mask]['error_excess'][group]={}
            for split in ('valid','test','bolivia','India-test'):
                if split in rr['B1'] and split in rr['B2']:
                    a,z=rr['B1'][split],rr['B2'][split]
                    measurements[mask]['error_excess'][group][split]={'B2_minus_B1_FP':z['fp']-a['fp'],
                        'B2_minus_B1_FN':z['fn']-a['fn'],'B2_minus_B1_TP':z['tp']-a['tp']}
    out={'source_sha256':sha256(baseline),'stratum_rule':'primary-label water fraction; zero-valid chips excluded/unknown',
         'measurements':measurements}
    (dest/'b2-strata.json').write_text(json.dumps(out,indent=2)+'\n')
    return out


def run_experiment(pack,config,dest):
    return train.fit(pack,config,dest,'tiny',train.device())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['inspect','single','cosine200','constant400','cosine400'])
    p.add_argument('--pack',default=str(Path.home()/'satquery-data/flood/packs/sen1floods11-hand-v1'))
    p.add_argument('--original',type=Path,default=Path('outputs/t15-tiny'))
    p.add_argument('--out',type=Path,default=Path('outputs/t15-debug'))
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.action=='inspect':
        pack=Pack(args.pack);out=inspect(pack,args.original,args.out)
        if out['pooled_pixelwise_optimal_iou']<.9:
            print('Pixelwise bound below .9: inspect spatial-context evidence before further training.',flush=True)
            return
        overlays(pack,out,args.out)
        b2_check(Path('outputs/t15-baselines.json'),args.out)
        print('Inspection done. Pointwise collision bound',out['pooled_pixelwise_optimal_iou'],flush=True)
    else:
        inspection=json.loads((args.out/'inspection.json').read_text())
        if inspection['true_unet_ceiling_below_gate']:
            raise ValueError('STOP: true input ceiling below gate; user decision required')
        pack=Pack(args.pack)
        original=json.loads((args.original/'flood-run.json').read_text())
        cfg=copy.deepcopy(original['config']);cfg['tiny_indices']=original['train_indices'];cfg['tiny_early_stop']=False
        if args.action=='single':
            cfg['tiny_indices']=[next(x['row'] for x in inspection['per_chip'] if x['chip_id']==inspection['worst_chip'])]
            cfg['epochs']=400;cfg['tiny_early_stop']=True;cfg['tiny_stop_iou']=.99
        elif args.action=='cosine200':cfg['schedule']='cosine'
        elif args.action=='constant400':cfg['epochs']=400
        elif args.action=='cosine400':cfg['schedule']='cosine';cfg['epochs']=400
        run_experiment(pack,cfg,args.out/args.action)


if __name__=='__main__':main()
