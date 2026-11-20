"""Promotion attempt 2, G2': one temperature per mode (docs/gates/LANDCOVER-INDIA-PROMOTION-ATTEMPT2.md)."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

CLASSES = ('trees', 'shrub_grass', 'cropland', 'built', 'water')
CLIP = (1e-6, 1 - 1e-6)
LOG_T_BOUNDS = (np.log(0.05), np.log(20.0))
GATE_COMMIT = 'db6e127'
FIT_SHA256 = '1c7ba3770e063ff05f3b2d33b11c345657dc9e52c419457d049697d9a97ec15e'
# Attempt-1 values from results/landcover-india-promotion/REPORT.md (Platt, fit A, evaluated on B).
ATTEMPT1 = {
    's1': {'macro': (0.12439133, 0.17925145), 'trees': (0.17148621, 0.34097547), 'shrub_grass': (0.17671411, 0.10814143),
           'cropland': (0.11249902, 0.17641676), 'built': (0.13143497, 0.16600996), 'water': (0.02982235, 0.10471364)},
    's2': {'macro': (0.17777859, 0.18661885), 'trees': (0.27809596, 0.17525744), 'shrub_grass': (0.15461319, 0.12996483),
           'cropland': (0.29977000, 0.36042995), 'built': (0.12497778, 0.20254590), 'water': (0.03143603, 0.06489612)},
}


def to_logits(p):
    return logit(np.clip(np.asarray(p, dtype=float), *CLIP))


def apply_temperature(z, t):
    return expit(np.asarray(z, dtype=float) / t)


def nll(z, y, t):
    q = np.clip(apply_temperature(z, t), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def fit_temperature(z, y):
    """Return (T, hit_bound) minimising mean binary NLL over every row and class."""
    z, y = np.asarray(z, dtype=float), np.asarray(y, dtype=float)
    res = minimize_scalar(lambda s: nll(z, y, np.exp(s)), bounds=LOG_T_BOUNDS, method='bounded')
    hit = bool(min(abs(res.x - b) for b in LOG_T_BOUNDS) < 1e-4)
    return float(np.exp(res.x)), hit


def ece(p, y):
    """Attempt-1 definition: 10 equal-width bins, 1 in the last bin, sum |sum(p - y)| / n."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    bins = np.minimum((p * 10).astype(int), 9)
    return float(sum(abs((p[bins == i] - y[bins == i]).sum()) for i in range(10)) / len(y))


def class_eces(p, y):
    return [ece(p[:, j], y[:, j]) for j in range(p.shape[1])]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(npz):
    a = np.load(npz)
    assert sha256(npz) == FIT_SHA256, 'fit data differs from the pre-registered file'
    assert int(a['seed']) == 1 and tuple(a['classes']) == CLASSES
    out = {}
    for mode in ('s1', 's2'):
        mask = (a['mode'] == mode) & (a['arm'] == 'full')
        folds = {f: mask & (a['fold'] == f) for f in 'AB'}
        assert folds['A'].sum() == folds['B'].sum() == 900 and len(set(a['patch_id'][mask])) == 1800
        p, y = a['scores'].astype(float), a['labels'].astype(float)
        z = to_logits(p)
        directions = {}
        for fit_fold, eval_fold in (('A', 'B'), ('B', 'A')):
            t, hit = fit_temperature(z[folds[fit_fold]], y[folds[fit_fold]])
            raw = class_eces(p[folds[eval_fold]], y[folds[eval_fold]])
            cal = class_eces(apply_temperature(z[folds[eval_fold]], t), y[folds[eval_fold]])
            directions[f'{fit_fold}->{eval_fold}'] = {
                'T': t, 'T_at_bound': hit, 'raw_macro_ece': float(np.mean(raw)), 'cal_macro_ece': float(np.mean(cal)),
                'raw_class_ece': dict(zip(CLASSES, raw)), 'cal_class_ece': dict(zip(CLASSES, cal)),
                'eval_positives': dict(zip(CLASSES, map(int, y[folds[eval_fold]].sum(0)))),
            }
        t_all, hit_all = fit_temperature(z[mask], y[mask])
        out[mode] = {'directions': directions, 'shipped_T': t_all, 'shipped_T_at_bound': hit_all,
                     'positives': {f: dict(zip(CLASSES, map(int, y[folds[f]].sum(0)))) for f in 'AB'}}
    verdict = all(d['cal_macro_ece'] < d['raw_macro_ece'] for m in out.values() for d in m['directions'].values())
    return out, verdict


def report(out, verdict, npz):
    f = lambda v: f'{v:.8f}'
    lines = ['# India land-cover promotion, attempt 2 (G2\' only)', '',
             f'Gate pre-registration `{GATE_COMMIT}` (docs/gates/LANDCOVER-INDIA-PROMOTION-ATTEMPT2.md). Attempt 1: [REPORT.md](../REPORT.md), recorded FAIL, unchanged.', '',
             '| Gate | Attempt 1 (`5e3fcee`) | Attempt 2 |', '|---|---|---|',
             '| G1 | PASS (s1 0.71720671, s2 0.68847330) | carried over: PASS |']
    for m in ('s1', 's2'):
        a1 = ATTEMPT1[m]['macro']
        d = out[m]['directions']
        a2 = '; '.join(f'{k}: {f(v["raw_macro_ece"])} → {f(v["cal_macro_ece"])} (T={v["T"]:.6f})' for k, v in d.items())
        ok = all(v['cal_macro_ece'] < v['raw_macro_ece'] for v in d.values())
        lines.append(f'| G2 {m} | Platt per class, A→B macro ECE {f(a1[0])} → {f(a1[1])}: FAIL | Temperature: {a2}: {"PASS" if ok else "FAIL"} |')
    lines += ['| G3 | Ludhiana FAIL, Punjab PASS (input bug since found) | pending corrected scenes |', '',
              f'**G2\' verdict: {"PASS" if verdict else "FAIL"}.** ' + ('Promotion still needs G3 on the corrected scenes.' if verdict else 'The India model is not shipped with calibrated scores.'), '',
              '## Temperatures', '', '| Mode | T fit A | T fit B | T fit on all 1,800 rows (ships only if G2\' passes) |', '|---|---:|---:|---:|']
    for m in ('s1', 's2'):
        d = out[m]['directions']
        bound = ' (at bound)' if out[m]['shipped_T_at_bound'] else ''
        lines.append(f'| {m} | {d["A->B"]["T"]:.6f} | {d["B->A"]["T"]:.6f} | {out[m]["shipped_T"]:.6f}{bound} |')
    lines += ['', 'T > 1 softens the scores and T < 1 sharpens them. One T per mode cannot change the class ranking.', '',
              '## Per-class ECE (context only, not gated)', '',
              '| Mode | Class | Pos A | Pos B | Attempt 1 raw B | Attempt 1 Platt B | Attempt 2 raw B | Attempt 2 temp A→B | Attempt 2 raw A | Attempt 2 temp B→A |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in ('s1', 's2'):
        d, pos = out[m]['directions'], out[m]['positives']
        for c in CLASSES:
            a1 = ATTEMPT1[m][c]
            lines.append(f'| {m} | {c} | {pos["A"][c]} | {pos["B"][c]} | {f(a1[0])} | {f(a1[1])} | {f(d["A->B"]["raw_class_ece"][c])} | '
                         f'{f(d["A->B"]["cal_class_ece"][c])} | {f(d["B->A"]["raw_class_ece"][c])} | {f(d["B->A"]["cal_class_ece"][c])} |')
    lines += ['', 'Each fold has 900 rows per mode. T is fitted on fold-model OOF predictions and transferred to the final weights, which have no held-out score.', '',
              '## Verified SHA-256', '', f'- Fit NPZ `{Path(npz).name}`: `{sha256(npz)}`']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=str(Path.home() / 'ben-work/adapt-run-3/seed-1/ben-adapt-predictions-seed-1.npz'))
    ap.add_argument('--out', default='results/landcover-india-promotion/attempt2')
    args = ap.parse_args(argv)
    out, verdict = evaluate(args.npz)
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    record = {'gate_commit': GATE_COMMIT, 'method': 'temperature scaling, one T per mode, sigmoid(logit(clip(p))/T)',
              'clip': list(CLIP), 'log_T_bounds': list(LOG_T_BOUNDS), 'fit_npz': args.npz, 'fit_npz_sha256': sha256(args.npz),
              'g2_prime': 'PASS' if verdict else 'FAIL', 'modes': out}
    (dest / 'temperature.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    (dest / 'REPORT.md').write_text(report(out, verdict, args.npz))
    print((dest / 'REPORT.md').read_text())


if __name__ == '__main__':
    main()
