"""T15 train/controls, eight-chip overfit gate and NO-UPDATE full-split timing.

No full training occurs unless --action train is explicitly supplied.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import yaml

from data.flood.learning import Pack,SEED,MASKS,masks,counts,iou,grouped,code_info,coverage,compare_hypotheses,metric_table
from data.flood.build_pack import sha256
from data.flood.model import UNet,FullResolutionCNN,loss


def seed_all():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


def synchronize(dev):
    if dev.type=='mps':torch.mps.synchronize()
    if dev.type=='cuda':torch.cuda.synchronize()


def labels_for_training(pack,indices,control):
    """Shuffle training chip-label correspondence only; never modify validation labels."""
    permutation=np.random.default_rng(SEED).permutation(indices) if control=='shuffle' else indices.copy()
    return {int(a):int(b) for a,b in zip(indices,permutation)}


def batch(pack,split,indices,mode,dev,control='none',label_rows=None):
    x=pack.inputs(split,indices,mode,control)
    rows=[label_rows[int(i)] for i in indices] if label_rows else indices
    y=pack.split(split)['label'][rows].astype(np.float32)
    return torch.from_numpy(x).to(dev),torch.from_numpy(y).to(dev)


def augment(x,y,rng):
    for i in range(len(x)):
        k=int(rng.integers(4))
        x[i]=torch.rot90(x[i],k,(-2,-1));y[i]=torch.rot90(y[i],k,(-2,-1))
        if rng.integers(2):
            x[i]=torch.flip(x[i],(-1,));y[i]=torch.flip(y[i],(-1,))
    return x,y


@torch.no_grad()
def evaluate(model,pack,split,indices,config,dev):
    model.eval();rows=pack.rows(split);arrays=pack.split(split)
    records={mask:[] for mask in MASKS}
    for start in range(0,len(indices),config['batch_size']):
        ids=indices[start:start+config['batch_size']]
        x,_=batch(pack,split,ids,config['mode'],dev,config['control'])
        predictions=(torch.sigmoid(model(x))>=config['threshold']).cpu().numpy()[:,0]
        for i,pred in zip(ids,predictions):
            row=rows.iloc[int(i)];target=arrays['label'][i,0]
            for mask,valid in masks(arrays,int(i)).items():
                c=counts(pred,target,valid)
                records[mask].append({'chip_id':row.chip_id,'split':split,'event':row.event,'counts':c,'water_iou':iou(c)})
    return records


def pooled_metric(records,mask='primary'):
    return iou(np.asarray([r['counts'] for r in records[mask]],dtype=np.int64).sum(0))


def select_epoch(evaluate_fn,epochs):
    """Only the validation callback is reachable from checkpoint selection."""
    best=None
    for epoch,state in epochs:
        value=evaluate_fn('valid',state)
        if value is not None and (best is None or value>best['water_iou']):
            best={'epoch':epoch,'water_iou':value,'state':state}
    if best is None:
        raise ValueError('STOP: validation water IoU is undefined for all epochs')
    return best


def train_epoch(model,pack,indices,labels,config,dev,opt,rng,update=True):
    model.train();order=rng.permutation(indices);total=0.;n=0
    for start in range(0,len(order),config['batch_size']):
        ids=order[start:start+config['batch_size']]
        x,y=batch(pack,'train',ids,config['mode'],dev,config['control'],labels)
        if config['augmentation']:
            x,y=augment(x,y,rng)
        opt.zero_grad(set_to_none=True)
        value=loss(model(x),y,config['pos_weight'],config.get('loss_kind','bce_dice'))
        if not torch.isfinite(value):
            raise ValueError('STOP: non-finite training loss')
        value.backward()
        if update:opt.step()
        total+=float(value.detach())*len(ids);n+=len(ids)
    return total/n


def result_schema(pack,config,model):
    return {'schema_version':1,'config':config,'seed':SEED,'pack_manifest_sha256':pack.sha,
            'code':code_info(),'norm_stats':pack.manifest['norm_stats'],
            'n_parameters':sum(p.numel() for p in model.parameters()),
            'label_mask':'label != -1','strict_mask':'label valid AND all S1/S2 finite/nonmissing AND Otsu valid',
            'mask_verdict_changes':'not measured until both trained modes are available'}


def fit(pack,config,out_dir,action,dev):
    seed_all();dest=Path(out_dir)
    if (dest/'flood-run.json').exists():raise ValueError('STOP: run output already exists')
    dest.mkdir(parents=True,exist_ok=True)
    indices=np.arange(len(pack.rows('train')))
    if action=='tiny':
        indices=np.asarray(config['tiny_indices'],dtype=int) if 'tiny_indices' in config else np.random.default_rng(SEED).choice(indices,size=8,replace=False)
        if not 1<=len(indices)<=8 or len(set(indices))!=len(indices) or np.any(indices<0) or np.any(indices>=len(pack.rows('train'))):
            raise ValueError('STOP: tiny indices must be 1..8 unique train rows')
    architecture=config.get('architecture','unet')
    if architecture not in ('unet','full_resolution'):
        raise ValueError('STOP: unknown architecture')
    model_class=FullResolutionCNN if architecture=='full_resolution' else UNet
    model=model_class(2 if config['mode']=='s1' else 15).to(dev)
    labels=labels_for_training(pack,indices,config['control'])
    opt=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=config['epochs']) if config['schedule']=='cosine' else None
    rng=np.random.default_rng(SEED)
    history=[];best=None;start=time.perf_counter();stop_reason='epoch_budget'
    def epochs():
        nonlocal best,stop_reason
        for epoch in range(1,config['epochs']+1):
            loss_value=train_epoch(model,pack,indices,labels,config,dev,opt,rng)
            if scheduler:scheduler.step()
            training=evaluate(model,pack,'train',indices,config,dev)
            train_iou=pooled_metric(training)
            entry={'epoch':epoch,'train_loss':loss_value,'train_water_iou':train_iou,
                   'train_strict_water_iou':pooled_metric(training,'strict')}
            history.append(entry)
            print(json.dumps(entry),flush=True)
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if action=='tiny':
                if train_iou is not None and (best is None or train_iou>best['water_iou']):
                    best={'epoch':epoch,'water_iou':train_iou,'state':state}
                if config.get('tiny_early_stop',True) and train_iou is not None and train_iou>config.get('tiny_stop_iou',.9):
                    stop_reason='target_reached';break
                if time.perf_counter()-start>=config.get('tiny_max_runtime_s',float('inf')):
                    stop_reason='time_limit';break
            else:
                yield epoch,state
    if action=='tiny':
        for _ in epochs():pass
    else:
        def valid_only(split,state):
            assert split=='valid'
            model.load_state_dict(state)
            metrics=evaluate(model,pack,split,np.arange(len(pack.rows(split))),config,dev)
            value=pooled_metric(metrics)
            history[-1]['valid_water_iou']=value
            history[-1]['valid_strict_water_iou']=pooled_metric(metrics,'strict')
            return value
        best=select_epoch(valid_only,epochs())
    if best is None:raise ValueError('STOP: no defined training metric')
    model.load_state_dict(best['state'])
    records={mask:[] for mask in MASKS}
    splits=('train',) if action=='tiny' else ('valid','test','bolivia')
    for split in splits:
        ids=indices if split=='train' else np.arange(len(pack.rows(split)))
        rr=evaluate(model,pack,split,ids,config,dev)
        for mask in MASKS:records[mask].extend(rr[mask])
    checkpoint=dest/'flood-checkpoint.pt'
    torch.save({'state_dict':best['state'],'config':config,'norm_stats':pack.manifest['norm_stats'],
                'pack_manifest_sha256':pack.sha,'best_epoch':best['epoch']},checkpoint)
    out=result_schema(pack,config,model)
    out.update({'action':action,'device':str(dev),'history':history,'best_epoch':best['epoch'],
                'selection_split':'train (overfit gate only)' if action=='tiny' else 'valid',
                'train_indices':indices.tolist(),'train_chip_ids':pack.rows('train').iloc[indices].chip_id.tolist(),
                'shuffle_label_rows':labels,'checkpoint_sha256':sha256(checkpoint),'checkpoint':checkpoint.name,
                'per_chip':records,'metrics':{mask:grouped(rr) for mask,rr in records.items()},
                'coverage':{s:coverage(pack.split(s)) for s in splits},
                'runtime_s':time.perf_counter()-start,'stop_reason':stop_reason,'tiny_gate_pass':best['water_iou']>.9 if action=='tiny' else None})
    if action!='tiny':
        b=json.loads(Path(config['baselines']).read_text())
        if b['pack_manifest_sha256']!=pack.sha:raise ValueError('STOP: baseline pack mismatch')
        out['control_check']={mask:out['metrics'][mask]['valid']['water_iou']<=b['metrics']['B0'][mask]['valid']['water_iou']+.02
                              for mask in MASKS} if config['control']!='none' else None
    (dest/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    (dest/'flood-run.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    (dest/'flood-run.md').write_text('# T15 '+action+'\n\n'+metric_table({'model':out['metrics']},['model'])+'\nTraining curve: see flood-run.json.\n')
    print(f'DONE action={action} best={best["water_iou"]:.6f} epoch={best["epoch"]} runtime={out["runtime_s"]:.1f}s',flush=True)
    return out


def benchmark(pack,config,out_dir,dev):
    seed_all();dest=Path(out_dir);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'timing.json').exists():raise ValueError('STOP: timing output exists')
    results={}
    for mode in ('s1','s1s2'):
        cfg={**config,'mode':mode,'augmentation':True,'control':'none'}
        model=UNet(2 if mode=='s1' else 15).to(dev)
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'])
        indices=np.arange(len(pack.rows('train')))
        labels=labels_for_training(pack,indices,'none')
        before=copy.deepcopy({k:v.cpu().clone() for k,v in model.state_dict().items()})
        # Warm up one batch; gradients only, with no optimizer step or buffer updates.
        x,y=batch(pack,'train',indices[:cfg['batch_size']],mode,dev)
        loss(model(x),y,cfg['pos_weight'],cfg.get('loss_kind','bce_dice')).backward();model.zero_grad(set_to_none=True)
        synchronize(dev);start=time.perf_counter()
        train_epoch(model,pack,indices,labels,cfg,dev,opt,np.random.default_rng(SEED),update=False)
        synchronize(dev);seconds=time.perf_counter()-start
        unchanged=all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in before.items())
        assert unchanged,'STOP: timing mutated model weights/buffers'
        results[mode]={'train_chips':len(indices),'seconds_per_epoch_forward_backward_no_updates':seconds,
                       'weights_unchanged':unchanged,'optimizer_steps':0,
                       'training_only_30_epoch_estimate_s':30*seconds,
                       'estimate_excludes':'optimizer steps, validation, final evaluation, serialization'}
        print(mode,json.dumps(results[mode]),flush=True)
        del model,opt
    out={'device':str(dev),'pack_manifest_sha256':pack.sha,'code':code_info(),'config':config,'modes':results}
    (dest/'timing.json').write_text(json.dumps(out,indent=2)+'\n')
    return out


def compare(s1_path,s1s2_path,baseline_path,output):
    s1,s2,b=[json.loads(Path(p).read_text()) for p in (s1_path,s1s2_path,baseline_path)]
    if not (s1['pack_manifest_sha256']==s2['pack_manifest_sha256']==b['pack_manifest_sha256']):
        raise ValueError('STOP: comparison pack mismatch')
    if s1['action']!='train' or s2['action']!='train' or s1['config']['control']!='none' or s2['config']['control']!='none':
        raise ValueError('STOP: hypotheses require both final trained modes, not tiny or controls')
    result=compare_hypotheses(s1['metrics'],s2['metrics'],b['metrics'])
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=['tiny','train','timing','compare'],required=True)
    p.add_argument('--pack',default=str(Path.home()/'satquery-data/flood/packs/sen1floods11-hand-v1'))
    p.add_argument('--config',default='training/configs/flood-s1.yaml');p.add_argument('--out-dir',required=True)
    p.add_argument('--control',choices=['none','blank','shuffle']);p.add_argument('--baselines',default='outputs/t15-baselines.json')
    p.add_argument('--s1-run');p.add_argument('--s1s2-run')
    args=p.parse_args(argv)
    if args.action=='compare':return compare(args.s1_run,args.s1s2_run,args.baselines,Path(args.out_dir)/'hypotheses.json')
    config=yaml.safe_load(Path(args.config).read_text())
    if config['seed']!=SEED:raise ValueError('STOP: fixed seed must be 17')
    config['control']=args.control or config['control'];config['baselines']=args.baselines
    pack=Pack(args.pack);dev=device()
    print('device',dev,flush=True)
    return benchmark(pack,config,args.out_dir,dev) if args.action=='timing' else fit(pack,config,args.out_dir,args.action,dev)


if __name__=='__main__':main()
