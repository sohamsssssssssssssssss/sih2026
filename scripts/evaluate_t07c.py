"""Fixed T07c evaluation. Read-only pack, label-free event thresholds, chip bootstrap."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.schema import Query, Scene
from core.tools.flood_otsu import FloodOtsuTool, water_mask
from models.change.sar_water import otsu_threshold

PACK_SHA = '250c230b4698ac4b7cab33e2c1666ebed2b1699455aafca5e56fb429ac6c60ad'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8<<20), b''): h.update(block)
    return h.hexdigest()


def counts(pred, target):
    valid=target!=-1; water=target==1
    return [int((pred & water & valid).sum()),int((pred & ~water & valid).sum()),int((~pred & water & valid).sum())]


def metrics(rows):
    a=np.asarray(rows,dtype=np.int64)
    c=a.sum(0); denominator=int(c.sum())
    rng=np.random.default_rng(17)
    resamples=a[rng.integers(0,len(a),size=(2000,len(a)))].sum(1)
    denominators=resamples.sum(1); ok=denominators>0
    boot=resamples[ok,0]/denominators[ok]
    return {'chips':len(a),'counts':c.tolist(),'iou':float(c[0]/denominator) if denominator else None,
            'ci95':np.percentile(boot,[2.5,97.5]).tolist() if len(boot) else None,
            'undefined_bootstraps':int((~ok).sum())}


def run(pack, destination):
    pack=Path(pack); destination=Path(destination)
    if destination.with_suffix('.json').exists() or destination.with_suffix('.md').exists():
        raise FileExistsError('Refusing to overwrite a T07c report')
    if sha(pack/'MANIFEST.json')!=PACK_SHA: raise ValueError('Pack manifest hash mismatch')
    manifest=json.loads((pack/'MANIFEST.json').read_text())
    if manifest['band_order']['s1']!=['VV','VH'] or manifest['units']['s1']['value']!='dB':
        raise ValueError('S1 bands/units mismatch')
    splits=tuple(manifest['counts']['per_split'])
    verified={}
    for split in splits:
        for layer in ('s1','label','otsu'):
            name=f'{split}/{layer}.npy';digest=sha(pack/name)
            if digest!=manifest['files'][name]['sha256']: raise ValueError(f'Hash mismatch: {name}')
            verified[name]=digest
    images={s:np.load(pack/s/'s1.npy',mmap_mode='r') for s in splits}
    events=defaultdict(list)
    for name, row in manifest['chips'].items():
        events[row['event']].append({'chip_id':name,**{k:row[k] for k in ('split','row')}})
    for rows in events.values(): rows.sort(key=lambda r:(r['split'],r['row']))
    # Audit TRAIN imagery before reading any held-out label arrays; endpoints stay fixed.
    train_audit={}
    for event,rows in sorted(events.items()):
        train=[r for r in rows if r['split']=='train']
        if train:
            vh=np.stack([images[r['split']][r['row'],1] for r in train])
            _,raw,threshold=water_mask(vh)
            train_audit[event]={'chips':len(train),'raw_db':raw,'clamped_db':threshold}
    print('TRAIN-only threshold audit:',json.dumps(train_audit),flush=True)
    thresholds={};predictions={}
    for event,rows in sorted(events.items()):
        vh=np.stack([images[r['split']][r['row'],1] for r in rows])
        masks,raw,threshold=water_mask(vh)
        thresholds[event]={'raw_db':raw,'clamped_db':threshold,'chips':[r['chip_id'] for r in rows]}
        for row,mask in zip(rows,masks): predictions[row['chip_id']]=mask
    del vh,masks
    # Thresholds and predictions are complete before any hand labels are opened.
    records=[]
    for split in splits:
        labels=np.load(pack/split/'label.npy',mmap_mode='r')
        shipped=np.load(pack/split/'otsu.npy',mmap_mode='r')
        rows=sorted([(name,r) for name,r in manifest['chips'].items() if r['split']==split], key=lambda t:t[1]['row'])
        for name,r in rows:
            i=r['row'];target=labels[i,0];vh=images[split][i,1]
            if not np.isin(target,[-1,0,1]).all(): raise ValueError('Invalid hand-label encoding')
            finite=np.isfinite(vh)&(target!=-1)
            threshold=otsu_threshold(vh[finite]) if finite.any() else None
            per_chip=(vh<threshold)&np.isfinite(vh) if threshold is not None else np.zeros(vh.shape,dtype=bool)
            records.append({'chip_id':name,'split':split,'event':r['event'],
                            'B1':counts(shipped[i,0]==1,target),'B2':counts(per_chip,target),
                            'T07c':counts(predictions[name],target)})
    results={}
    for group in (*splits,'India-test'):
        rows=[r for r in records if r['split']==group or (group=='India-test' and r['split']=='test' and r['event']=='India')]
        results[group]={arm:metrics([r[arm] for r in rows]) for arm in ('B1','B2','T07c')}
    verdict='PASS' if results['India-test']['T07c']['iou'] is not None and results['India-test']['T07c']['iou']>=0.65 else 'FAIL'
    # One real test patch, with all India-event VH as the threshold window.
    rows=events['India'];chip_index=next(i for i,r in enumerate(rows) if r['split']=='test')
    row=rows[chip_index]
    x=np.stack([images[r['split']][r['row']] for r in rows])
    scene=Scene(scene_id='a'*64,sensor='S1',bands=('VV','VH'),units='dB',gsd_m=10,
                crs='EPSG:4326',bounds=(75,20,75.1,20.1),nodata_pct=0,source='Sen1Floods11 real patch')
    tool=FloodOtsuTool();t=time.perf_counter()
    result=tool.run(scene,Query(text='Map flood water',qtype='flood'),{'s1':{'pixels':x,'bands':('VV','VH'),'units':'dB','chip_index':chip_index}})
    latency=(time.perf_counter()-t)*1000
    if 'error' in result.value: raise ValueError(result.value)
    if not np.array_equal(np.asarray(result.value['water_mask'],dtype=bool),predictions[row['chip_id']]):
        raise AssertionError('Adapter and evaluation masks disagree')
    report={'preregistration_commit':'f5bd22d','manifest_sha256':PACK_SHA,'verified_files':verified,
            'method':{'bins':256,'clip_percentiles':[0.1,99.9],'clamp_db':[-26,-14],
                      'clamp_choice':'user fixed before evaluation; train-only audit, no tuning',
                      'pool_scope':'all chips of same event across splits; imagery only',
                      'bootstrap_unit':'chip','bootstrap_reps':2000,'seed':17,'valid_mask':'label != -1'},
            'train_only_audit':train_audit,'event_thresholds':thresholds,'metrics':results,
            'per_chip':records,'verdict':verdict,'india_test_pass_iou':0.65,
            'latency':{'cpu':True,'patch':row['chip_id'],'patch_shape':[512,512],
                       'event_context_chips':len(rows),'wall_ms':latency,'tool_ms':result.latency_ms},
            'source_sha256':{str(p):sha(p) for p in (Path('core/tools/flood_otsu.py'),Path(__file__))}}
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    text=f'# T07c flood Otsu — {verdict}\n\nIndia-test pass rule: point water IoU >= 0.65. No post-verdict tuning.\n\n'
    text+='| Split | Arm | Water IoU | Chip-bootstrap CI95 | Chips |\n|---|---|---:|---|---:|\n'
    for group,arms in results.items():
        for arm,m in arms.items():
            text+=f"| {group} | {arm} | {m['iou']:.6f} | [{m['ci95'][0]:.6f}, {m['ci95'][1]:.6f}] | {m['chips']} |\n"
    text+='\nAll scoring uses label != -1. B1 is the shipped mask, B2 is the fixed historical\nper-chip baseline. Non-finite VH is dry for T07c, including label-valid pixels.\nOtsu pools all chips of the same event across splits, with imagery only; labels\nare opened after thresholds/predictions are complete. This is transductive event\ncontext, not independent-chip prediction. CI95 conditions on the event thresholds;\nIndia-test has one event, so chip bootstrap does not estimate new-event uncertainty.\n\n'
    text+='The clamp [-26,-14] was supplied by the user before this run, not learned from\ntest. Train-only raw/clamped thresholds (dB), audited first without any labels:\n\n| Train event | Chips | Raw | Clamped |\n|---|---:|---:|---:|\n'
    for event,a in train_audit.items():text+=f"| {event} | {a['chips']} | {a['raw_db']:.4f} | {a['clamped_db']:.4f} |\n"
    text+=f"\nCPU latency for one real patch ({row['chip_id']}) including threshold computation\nover {len(rows)} India-event chips and patch-mask assembly: **{latency:.1f} ms**.\nThe adapter mask exactly matches the evaluated mask for this patch.\n\n"
    text+='Tool remains experimental_unvalidated whatever the verdict; no calibrated\nconfidence. This detects open water, not flood causality or water under vegetation.\nPreregistration: docs/hypotheses/T07c.md at f5bd22d. JSON includes hashes of consumed\narrays, fixed method parameters, pool membership, thresholds and per-chip counts.\n'
    destination.with_suffix('.md').write_text(text)
    print(text,flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack',default=str(Path.home()/'satquery-data/flood/packs/sen1floods11-hand-v1'))
    parser.add_argument('--out',default='outputs/T07c-flood')
    args=parser.parse_args();run(args.pack,args.out)
