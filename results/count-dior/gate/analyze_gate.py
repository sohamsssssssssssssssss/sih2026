"""Offline analysis; run with python3.11 results/count-dior/gate/analyze_gate.py."""
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import random
import numpy as np

HERE = Path(__file__).resolve().parent
SHA = 'f77636624bd4824404b327f1168e40a3a450f74e7ca59a107fb3fb6375387df2'
SEED, REPS = 17, 2000


def metrics(y, p):
    e = np.abs(y-p)
    return dict(n=len(y), exact=float(np.mean(e == 0)), mae=float(np.mean(e)),
                within1=float(np.mean(e <= 1)), absent_false_count_rate=float(np.mean(p[y == 0] > 0)),
                present_exact=float(np.mean(e[y > 0] == 0)),
                presence_false_negative_n=int(np.sum((y > 0) & (p == 0))),
                presence_false_positive_n=int(np.sum((y == 0) & (p > 0))),
                counting_error_when_both_present_n=int(np.sum((y > 0) & (p > 0) & (e > 0))))


def split_images(rows):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r['image_id']].append(i)
    strata = defaultdict(list)
    for image, indices in sorted(groups.items()):
        npos = sum(rows[i]['expected'] > 0 for i in indices)
        strata[(npos, len(indices)-npos)].append(image)
    keys = sorted(strata)
    # Find near-half stratum quotas using labels only, never detector predictions.
    choices = [range(max(0, len(strata[k])//2-2), min(len(strata[k]), len(strata[k])//2+2)+1) for k in keys]
    feasible = [q for q in itertools.product(*choices)
                if sum(qi*k[0] for qi,k in zip(q,keys)) == 250
                and sum(qi*k[1] for qi,k in zip(q,keys)) == 250]
    assert feasible, 'Cannot produce balanced halves with near-half stratum quotas'
    quotas = min(feasible, key=lambda q: (abs(2*sum(q)-len(groups)),
                                         sum(abs(2*qi-len(strata[k])) for qi,k in zip(q,keys)), q))
    rng = random.Random(SEED)
    selected = set()
    for k,q in zip(keys,quotas):
        rng.shuffle(strata[k])
        selected.update(strata[k][:q])
    select = np.array([r['image_id'] in selected for r in rows])
    report = ~select
    assert select.sum() == report.sum() == 500
    assert not selected & {rows[i]['image_id'] for i in np.flatnonzero(report)}
    return select, report, sorted(selected), sorted(set(groups)-selected)


def bootstrap(rows, y, p):
    ids = sorted({r['image_id'] for r in rows})
    groups = [[i for i,r in enumerate(rows) if r['image_id'] == image] for image in ids]
    sums = np.array([[len(g), np.sum((p[g] == y[g]).astype(int)-(y[g] == 0)),
                      np.sum(np.abs(p[g]-y[g])-y[g])] for g in groups])
    rng = np.random.default_rng(SEED)
    draws = sums[rng.integers(len(ids), size=(REPS,len(ids)))].sum(axis=1)
    delta = draws[:,1:]/draws[:,:1]
    return dict(reps=REPS, seed=SEED, unit='image_id', n_images=len(ids),
                exact_diff=float(np.mean((p == y).astype(int)-(y == 0))),
                mae_diff=float(np.mean(np.abs(p-y)-y)),
                exact_diff_ci95=np.quantile(delta[:,0],[.025,.975]).tolist(),
                mae_diff_ci95=np.quantile(delta[:,1],[.025,.975]).tolist(),
                method='paired image cluster percentile bootstrap, numpy default_rng, linear quantiles; lower MAE is better')


def main():
    raw = (HERE.parent/'dior-count-val-evaluation.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA
    data = json.loads(raw)
    sweep = data['threshold_sweep']
    ts = sorted(sweep, key=float)
    rows = sweep['0.35']['records']
    keys = [(r['image_id'],r['class']) for r in rows]
    assert len(keys) == len(set(keys)) == 1000
    y = np.array([r['expected'] for r in rows])
    assert np.sum(y == 0) == np.sum(y > 0) == 500
    counts, scores = {}, {}
    for t in ts:
        indexed = {(r['image_id'],r['class']):r for r in sweep[t]['records']}
        rr = [indexed[k] for k in keys]
        assert len(indexed) == 1000 and [r['expected'] for r in rr] == y.tolist()
        counts[t] = np.array([r['prediction'] for r in rr])
        scores[t] = [[e['confidence'] for e in r['evidence'] if e['type'] == 'bounding_box'] for r in rr]
        assert all(len(s) == n for s,n in zip(scores[t],counts[t]))
        assert all(all(float(t) <= s <= 1 for s in ss) for ss in scores[t])
    select, report, select_ids, report_ids = split_images(rows)
    assert y[select].astype(bool).sum() == y[report].astype(bool).sum() == 250
    candidates = []
    def add(name,p):
        candidates.append((name,p))
    for tc in ts:
        for tg in ts:
            if float(tg) >= float(tc):
                for k in (1,2):
                    add(f'A count={tc} gate={tg} k={k}', np.where(counts[tg] >= k, counts[tc], 0))
    for tc in ts:
        top = np.array([max(s,default=0) for s in scores[tc]])
        recounted = np.array([sum(s >= float(tc) for s in ss) for ss in scores['0.25']])
        for si in range(300,701,25):
            s = si/1000
            add(f'B count={tc} score={s:.3f}', np.where(top >= s,counts[tc],0))
            top25 = np.array([max(ss,default=0) for ss in scores['0.25']])
            add(f'C count={tc} score={s:.3f}', np.where(top25 >= s,recounted,0))
    # Stable enumeration order breaks exact ties; no report metric enters selection.
    best_name, best = max(candidates,key=lambda item: np.mean(item[1][select] == y[select]))
    zero = np.zeros_like(y)
    baselines = {'always_zero':zero, 'detector@0.35':counts['0.35'],
                 'oracle_presence@0.35':np.where(y > 0,counts['0.35'],0)}
    oracle = {t:metrics(y[report],np.where(y[report] > 0,counts[t][report],0)) for t in ts}
    boot = bootstrap([r for r,m in zip(rows,report) if m],y[report],best[report])
    verdict = 'beats always-0' if boot['exact_diff_ci95'][0] > 0 else 'does not beat always-0'
    result = dict(source_sha256=SHA, structure={k:type(v).__name__ for k,v in data.items()},
                  record_fields=list(rows[0]), box_fields=list(next(e for r in rows for e in r['evidence'] if e['type']=='bounding_box')),
                  split=dict(seed=SEED, method='Image groups stratified by (present, absent) pair counts; near-half quotas chosen for exact 250/250 balance, then closest image half and quota balance; lexicographic tie break; shuffle sorted image IDs within strata with random.Random(17).',select_image_ids=select_ids,report_image_ids=report_ids),
                  selection=dict(rule='maximum select exact; ties use A then B/C enumeration order, ascending thresholds and k/score',best_gate=best_name,select=metrics(y[select],best[select]),report=metrics(y[report],best[report])),
                  bootstrap=boot, verdict=verdict,
                  baselines={name:{'full':metrics(y,p),'select':metrics(y[select],p[select]),'report':metrics(y[report],p[report])} for name,p in baselines.items()},
                  oracle_report_by_count_threshold=oracle,
                  recount_matches_stored={t:bool(np.array_equal(counts[t],np.array([sum(s >= float(t) for s in ss) for ss in scores['0.25']]))) for t in ts},
                  full_sweep_label='in-sample, optimistic',
                  full_sweep=[dict(gate=name,**metrics(y,p),select_exact=metrics(y[select],p[select])['exact']) for name,p in candidates],
                  unknown={'fresh_frozen_sample':'not measured','deployment_accuracy':'not measured','scores_below_0.25':'not measured','box_localization_correctness':'not measured'})
    assert np.isclose(metrics(y,zero)['exact'],.5)
    assert np.isclose(metrics(y,counts['0.35'])['exact'],data['metrics']['overall']['accuracy'])
    assert np.isclose(metrics(y,counts['0.35'])['mae'],data['metrics']['overall']['mean_absolute_error'])
    assert metrics(np.array([0,1,2]),np.array([1,0,1]))['presence_false_negative_n'] == 1
    (HERE/'gate-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    lines = ['# Offline DIOR presence gate', '', f'Verdict: **{verdict}** under the required report-half exact CI rule.', '',
             f'Source SHA-256: `{SHA}`. Only this committed evaluation file was used. Python stdlib + numpy; no downloads, GPU or model run.', '',
             'Structure: top-level metadata, metrics, baselines and threshold_sweep. Each threshold contains overall/per_class summaries and 1,000 records. Records store image_id, class, image, expected, hashes, prompt, prediction and evidence. Box evidence stores confidence and coordinates; detection_count evidence stores thresholds. A/B/C are possible. Scores are retained post-NMS box scores, not raw model proposals; scores below 0.25 are unavailable.', '',
             f'Split: 500 pairs per half, each 250 present / 250 absent; {len(select_ids)} select images / {len(report_ids)} report images, no image overlap. '+result['split']['method'], '',
             'Selection: '+result['selection']['rule']+'. All candidates were specified before report scoring. B uses retained scores at its count threshold; C recounts retained boxes from threshold 0.25 (no new NMS). Recount equality checks: '+str(result['recount_matches_stored'])+'.', '',
             f'Chosen gate: **{best_name}**. Select exact: {result["selection"]["select"]["exact"]:.3f}.', '',
             '## Report half', '', '| Method | Exact | MAE | Within-1 | Absent false count | Present exact |', '|---|---:|---:|---:|---:|---:|']
    for name,m in [(best_name,result['selection']['report'])]+[(n,v['report']) for n,v in result['baselines'].items()]:
        lines.append(f'| {name} | {m["exact"]:.3f} | {m["mae"]:.3f} | {m["within1"]:.3f} | {m["absent_false_count_rate"]:.3f} | {m["present_exact"]:.3f} |')
    lines += ['',f'Paired image bootstrap: {REPS} reps, seed {SEED}, percentile 95% CIs. Gate minus always-0 exact: {boot["exact_diff"]:+.4f}, CI {boot["exact_diff_ci95"]}; MAE: {boot["mae_diff"]:+.4f}, CI {boot["mae_diff_ci95"]}. Negative MAE difference is improvement.', '',
              'Oracle true presence @0.35 is a ceiling for removing absent-class errors while preserving detector counts; it does not fix missed/incorrect counts on present pairs and is not an implementable gate. Other count-threshold oracle metrics are recorded in JSON.', '',
              '## Presence versus counting errors', '', 'Disjoint error counts (output presence, not latent gate decision):']
    for name,m in [(best_name,result['selection']['report']),('detector@0.35',result['baselines']['detector@0.35']['report'])]:
        lines += [f'- {name}: absent false positives {m["presence_false_positive_n"]}; present output-zero misses {m["presence_false_negative_n"]}; wrong nonzero count on present pairs {m["counting_error_when_both_present_n"]}.']
    lines += ['', '## Limits', '', 'A fresh frozen sample is required before any claim. This split is exploratory reuse of already inspected evidence; bootstrap intervals condition on the selected gate and this split. Fresh-sample performance, deployment accuracy and localization correctness: not measured. Confidence gates use stored retained boxes only. No per-class tuning or model changes.', '',
              '## Full 1,000 sweep — in-sample, optimistic', '', '| Gate | Exact | MAE | Within-1 | Absent false count | Present exact | Select exact |', '|---|---:|---:|---:|---:|---:|---:|']
    for name, baseline in result['baselines'].items():
        m = baseline['full']
        lines.append(f'| {name} (baseline) | {m["exact"]:.3f} | {m["mae"]:.3f} | {m["within1"]:.3f} | {m["absent_false_count_rate"]:.3f} | {m["present_exact"]:.3f} | {baseline["select"]["exact"]:.3f} |')
    for m in result['full_sweep']:
        lines.append(f'| {m["gate"]} | {m["exact"]:.3f} | {m["mae"]:.3f} | {m["within1"]:.3f} | {m["absent_false_count_rate"]:.3f} | {m["present_exact"]:.3f} | {m["select_exact"]:.3f} |')
    (HERE/'GATE.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ('selection','bootstrap','verdict','oracle_report_by_count_threshold')},indent=2))


if __name__ == '__main__':
    main()
