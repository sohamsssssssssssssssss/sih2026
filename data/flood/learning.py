"""Shared T15 verified loading, masks and chip-bootstrap IoU. No training here."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from data.flood.build_pack import load_split, sha256, verify_pack

SEED, REPS = 17, 2000
PACK_SHA = '250c230b4698ac4b7cab33e2c1666ebed2b1699455aafca5e56fb429ac6c60ad'
MASKS = ('primary','strict')


def code_info():
    root=Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
    try:
        commit,dirty=git('rev-parse','HEAD'),bool(git('status','--porcelain'))
    except (OSError,subprocess.CalledProcessError):
        import os
        commit,dirty=os.environ.get('T15_CODE_COMMIT','unknown'),None
    return {'commit':commit,'dirty':dirty,'source_sha256':{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}}


class Pack:
    def __init__(self,path,expected_sha=PACK_SHA):
        self.path=Path(path)
        self.sha=sha256(self.path/'MANIFEST.json')
        if expected_sha is not None and self.sha!=expected_sha:
            raise ValueError('STOP: unexpected flood manifest hash')
        self.manifest=verify_pack(self.path)
        self.index=pd.read_parquet(self.path/'index.parquet')
        self.cache={}

    def split(self,name):
        if name not in self.cache:
            self.cache[name]=load_split(self.path,name)
        return self.cache[name]

    def rows(self,name):
        return self.index.loc[self.index.split==name].sort_values('row').reset_index(drop=True)

    def inputs(self,split,indices,mode,control='none'):
        arrays=self.split(split)
        parts=[]
        for sensor in (('s1',) if mode=='s1' else ('s1','s2')):
            x=arrays[sensor][indices].astype(np.float32)
            missing=~np.isfinite(x)
            if sensor=='s2':
                missing |= x==0
            stats=self.manifest['norm_stats'][sensor]
            mean=np.asarray(stats['mean'],dtype=np.float32)[None,:,None,None]
            std=np.asarray(stats['std'],dtype=np.float32)[None,:,None,None]
            std=np.maximum(std,1e-6)
            x=(x-mean)/std
            x[missing]=0
            parts.append(x)
        result=np.concatenate(parts,1)
        return np.zeros_like(result) if control=='blank' else result


def masks(arrays,index):
    y=arrays['label'][index,0]
    primary=y!=-1
    s1=arrays['s1'][index]
    s2=arrays['s2'][index]
    strict=primary & np.isfinite(s1).all(0) & np.isfinite(s2).all(0) & (s2!=0).all(0) & (arrays['otsu'][index,0]!=-1)
    return {'primary':primary,'strict':strict}


def counts(pred,target,mask):
    valid=mask & (target!=-1)
    water=target==1
    pred=np.asarray(pred,dtype=bool)
    return [int((pred & water & valid).sum()),int((pred & ~water & valid).sum()),
            int((~pred & water & valid).sum()),int(valid.sum())]


def iou(c):
    tp,fp,fn=c[:3]
    return float(tp/(tp+fp+fn)) if tp+fp+fn else None


def summarize(rows):
    c=np.asarray([r['counts'] for r in rows],dtype=np.int64)
    if not len(c):
        return {'n_chips':0,'water_iou':None,'ci95':[None,None]}
    total=c.sum(0)
    rng=np.random.default_rng(SEED)
    draws=c[rng.integers(len(c),size=(REPS,len(c)))].sum(1)
    union=draws[:,:3].sum(1)
    defined=union>0
    values=draws[defined,0]/union[defined]
    interval=np.quantile(values,[.025,.975]).tolist() if len(values) else [None,None]
    return {'n_chips':len(c),'water_iou':iou(total),'ci95':interval,'tp':int(total[0]),'fp':int(total[1]),
            'fn':int(total[2]),'valid_pixels':int(total[3]),'bootstrap_reps':REPS,'seed':SEED,
            'bootstrap_unit':'chip','zero_union_resamples':int((~defined).sum())}


def grouped(rows):
    groups={}
    for split in sorted({r['split'] for r in rows}):
        rr=[r for r in rows if r['split']==split]
        groups[split]=summarize(rr)
        for event in sorted({r['event'] for r in rr}):
            groups[f'{split}/event/{event}']=summarize([r for r in rr if r['event']==event])
    india=[r for r in rows if r['split']=='test' and r['event']=='India']
    if india:
        groups['India-test']=summarize(india)
    return groups


def coverage(arrays):
    sums={'target_valid':0,'missing_otsu':0,'nonfinite_vh':0,'s2_nodata':0,'strict_excluded':0}
    for i in range(len(arrays['label'])):
        mm=masks(arrays,i);valid=mm['primary']
        sums['target_valid']+=int(valid.sum())
        sums['missing_otsu']+=int((valid & (arrays['otsu'][i,0]==-1)).sum())
        sums['nonfinite_vh']+=int((valid & ~np.isfinite(arrays['s1'][i,1])).sum())
        s2=arrays['s2'][i]
        sums['s2_nodata']+=int((valid & ((s2==0)|~np.isfinite(s2)).any(0)).sum())
        sums['strict_excluded']+=int((valid & ~mm['strict']).sum())
    n=sums['target_valid']
    return {'target_valid_pixels':n,**{k:{'count':v,'pct_target_valid':100*v/n if n else None} for k,v in sums.items() if k!='target_valid'}}


def metric_table(results,arms):
    lines=['| Group | Arm | Primary IoU [CI95] | Strict IoU [CI95] | Chips |',
           '|---|---|---|---|---:|']
    for group in ('valid','test','bolivia','India-test'):
        for arm in arms:
            primary=results[arm]['primary'].get(group)
            strict=results[arm]['strict'].get(group)
            if primary is None:
                continue
            def fmt(m):
                return 'unknown' if m['water_iou'] is None else f'{m["water_iou"]:.6f} [{m["ci95"][0]:.6f}, {m["ci95"][1]:.6f}]'
            lines.append(f'| {group} | {arm} | {fmt(primary)} | {fmt(strict)} | {primary["n_chips"]} |')
    return '\n'.join(lines)+'\n'


def compare_hypotheses(s1,s1s2,baselines):
    verdicts={}
    for mask in MASKS:
        verdicts[mask]={}
        for h,a,b,group in [('H1',s1,baselines['B1'],'test'),('H2',s1,baselines['B1'],'India-test'),
                            ('H3',s1s2,s1,'test')]:
            aa=a[mask].get(group) if a else None
            bb=b[mask].get(group) if b else None
            verdicts[mask][h]=(aa['ci95'][0]>bb['ci95'][1]) if aa and bb and aa['ci95'][0] is not None and bb['ci95'][1] is not None else None
    return {'verdicts':verdicts,'mask_verdict_changes':[h for h in ('H1','H2','H3')
            if verdicts['primary'][h] is not None and verdicts['strict'][h] is not None and verdicts['primary'][h]!=verdicts['strict'][h]]}
