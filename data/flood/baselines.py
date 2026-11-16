"""Final CPU flood baselines. No fitting on held-out labels or model training."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from data.flood.learning import Pack,MASKS,counts,iou,grouped,coverage,masks,code_info,metric_table
from models.change.sar_water import otsu_threshold,OTSU_BINS,OTSU_CLIP_PERCENTILES
import numpy as np


def run(pack,out_dir):
    dest=Path(out_dir);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'t15-baselines.json').exists():
        raise ValueError('STOP: baseline output already exists')
    records={a:{mask:[] for mask in MASKS} for a in ('B0','B1','B2','B3')}
    cov,agreements,thresholds={},{},{}
    for split in pack.manifest['counts']['per_split']:
        arrays=pack.split(split);index=pack.rows(split)
        cov[split]=coverage(arrays)
        equal={mask:0 for mask in MASKS};total={mask:0 for mask in MASKS}
        overlap_equal=overlap_n=0
        for i,row in index.iterrows():
            target=arrays['label'][i,0];mm=masks(arrays,i)
            vh=arrays['s1'][i,1];finite=mm['primary'] & np.isfinite(vh)
            threshold=otsu_threshold(vh[finite]) if finite.any() else None
            pred2=(vh<threshold)&finite if threshold is not None else np.zeros_like(target,dtype=bool)
            pred1=arrays['otsu'][i,0]==1
            pred={'B0':np.zeros_like(target,dtype=bool),'B1':pred1,'B2':pred2,'B3':arrays['jrc'][i,0]==1}
            thresholds[row.chip_id]=threshold
            overlap=finite & (arrays['otsu'][i,0]!=-1)
            overlap_equal+=int(((pred1==pred2)&overlap).sum());overlap_n+=int(overlap.sum())
            for mask,valid in mm.items():
                equal[mask]+=int(((pred1==pred2)&valid).sum());total[mask]+=int(valid.sum())
                for arm,prediction in pred.items():
                    c=counts(prediction,target,valid)
                    records[arm][mask].append({'chip_id':row.chip_id,'split':split,'event':row.event,'counts':c,'water_iou':iou(c)})
        agreements[split]={mask:{'equal_pixels':equal[mask],'pixels':total[mask],
                           'agreement_pct':100*equal[mask]/total[mask] if total[mask] else None} for mask in MASKS}
        agreements[split]['finite_VH_and_shipped_valid']={'equal_pixels':overlap_equal,'pixels':overlap_n,
                    'agreement_pct':100*overlap_equal/overlap_n if overlap_n else None}
        print(f'Baseline counts complete: {split}',flush=True)
    metrics={a:{mask:grouped(rr) for mask,rr in masks_.items()} for a,masks_ in records.items()}
    out={'schema_version':1,'pack_manifest_sha256':pack.sha,'code':code_info(),'n_chips':len(pack.index),
         'config':{'seed':17,'bootstrap_reps':2000,'bootstrap_unit':'chip','metric':'pooled water IoU',
                   'primary_mask':'label != -1','strict_mask':'primary AND all S1/S2 finite/nonmissing AND Otsu != -1',
                   'B2_implementation':'models.change.sar_water.otsu_threshold','B2_bins':OTSU_BINS,
                   'B2_clip_percentiles':list(OTSU_CLIP_PERCENTILES),'B2_water_rule':'VH < threshold',
                   'missing_prediction':'dry','zero_union_iou':'unknown'},
         'coverage':cov,'B2_vs_B1_agreement':agreements,'B2_thresholds':thresholds,'metrics':metrics,'per_chip':records}
    (dest/'t15-baselines.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    text='# T15 final CPU baselines\n\n'+metric_table(metrics,records)
    text+='\nOtsu recomputation uses the existing 256-bin helper with its fixed 0.1/99.9 percentile clipping. No agreement is assumed.\n'
    text+='\n| Split | Agreement primary % | Agreement strict % | Agreement finite VH / shipped valid % |\n|---|---:|---:|---:|\n'
    for split,agreement in agreements.items():
        text+=f'| {split} | {agreement["primary"]["agreement_pct"]} | {agreement["strict"]["agreement_pct"]} | {agreement["finite_VH_and_shipped_valid"]["agreement_pct"]} |\n'
    text+='\n| Split | Target-valid pixels | Missing Otsu count (%) | Non-finite VH count (%) | S2 nodata count (%) |\n|---|---:|---|---|---|\n'
    for split,c in cov.items():
        cells=[f'{c[k]["count"]} ({c[k]["pct_target_valid"]:.6f}%)' for k in ('missing_otsu','nonfinite_vh','s2_nodata')]
        text+=f'| {split} | {c["target_valid_pixels"]} | '+ ' | '.join(cells)+' |\n'
    text+='\nAffected rules may overlap. Both masks use identical fixed predictions. Per-event metrics, per-chip counts, strict exclusions and provenance are in JSON. No model hypothesis has been tested.\n'
    (dest/'t15-baselines.md').write_text(text)
    print(metric_table(metrics,records),flush=True)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pack',default=str(Path.home()/'satquery-data/flood/packs/sen1floods11-hand-v1'))
    p.add_argument('--out-dir',default='outputs')
    args=p.parse_args()
    run(Pack(args.pack),args.out_dir)
