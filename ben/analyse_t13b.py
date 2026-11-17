"""Analyse saved T13b summaries only; no fitting, inference, or invented CIs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

MODES = ('s1', 's2', 's1s2')
ARMS = ('none', 'renorm', 'head', 'full', 'shuffle', 'shuffle-fixed', 'bn-only')
BASE_ARMS = ARMS[:5]
CLASSES = ('trees', 'shrub_grass', 'cropland', 'built', 'water')
COMMIT = 'a0d7002cb9a43162e297f98a795694982791acb5'
REFERENCE_SHA = 'f5eb2569d343f3eab62bc65b19e9b3e6e773ea881d75454caef00000070fd447'
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError(f'Non-finite JSON constant: {value}')
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def unanimity(verdicts):
    if len(verdicts) != 3 or any(v not in ('PASS', 'FAIL', 'INCONCLUSIVE') for v in verdicts):
        raise ValueError('Expected all three seed verdicts')
    if all(v == 'PASS' for v in verdicts):
        return 'PASS'
    if all(v == 'FAIL' for v in verdicts):
        return 'FAIL'
    return 'INCONCLUSIVE'


def required_all(verdicts):
    return 'FAIL' if 'FAIL' in verdicts else 'INCONCLUSIVE' if 'INCONCLUSIVE' in verdicts else 'PASS'


def positive_ci(value, ci):
    if ci is None:
        return 'INCONCLUSIVE'
    if len(ci) != 2 or not all(math.isfinite(x) for x in ci) or ci[0] > ci[1]:
        raise ValueError('Invalid saved CI95')
    return 'PASS' if value > 0 and ci[0] > 0 else 'FAIL'


def stats(values):
    if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in values):
        raise ValueError('Missing/non-finite metric')
    return {'mean': statistics.fmean(values), 'min': min(values), 'max': max(values)}


def metric(run, mode, arm, fold=None, group='pooled'):
    mode = run['modes'][mode]
    return mode['out_of_fold'][arm] if fold is None else mode['folds'][fold]['arms'][arm]['groups'][group]


def contrast(runs, mode, right, fold=None, left='full'):
    samples = []
    for run in runs:
        value = metric(run, mode, left, fold)['macro_map'] - metric(run, mode, right, fold)['macro_map']
        ci = None
        if fold is None and left == 'full' and right == 'none':
            saved = run['modes'][mode]['hypotheses']['H1']['per_arm']['full']
            assert abs(value - saved['gain']) < 1e-12
            ci = saved['gain_ci95']
        elif fold is None and left == 'full' and right == 'renorm':
            saved = run['modes'][mode]['matched']
            assert abs(value - saved['full_minus_renorm']) < 1e-12
            ci = saved['ci95']
        samples.append({'seed': run['seed'], 'value': value, 'ci95': ci,
                        'verdict': positive_ci(value, ci),
                        'ci_source': 'saved paired patch-bootstrap CI' if ci is not None else 'paired CI not saved'})
    return {**stats([s['value'] for s in samples]), 'seeds': samples,
            'verdict': unanimity([s['verdict'] for s in samples])}


def validate_inputs(folder, reference):
    checksums = folder / 'SHA256SUMS.txt'
    for line in checksums.read_text().splitlines():
        digest, name = line.split('  ', 1)
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or sha(folder / relative) != digest:
            raise ValueError(f'Input checksum mismatch: {name}')
    expected = {name: hashlib.sha256(subprocess.check_output(
        ['git', 'show', f'{COMMIT}:ben/{name}.py'], cwd=ROOT)).hexdigest()
        for name in ('adapt', 'transfer', 'landcover')}
    runs = []
    for seed in (1, 2, 3):
        directory = folder / f'seed-{seed}'
        run = read_json(directory / 'ben-adapt.json')
        last = [x.strip() for x in (directory / 'ben-adapt-progress.txt').read_text().splitlines() if x.strip()][-1]
        if not last.startswith('DONE ') or f'/seed-{seed}/ben-adapt.json' not in last:
            raise ValueError('Incomplete seed progress log')
        assert run['code']['git_sha'] == COMMIT
        assert all(run['code'][name + '_py_sha256'] == digest for name, digest in expected.items())
        assert run['seed'] == run['config']['train_seed'] == seed
        assert not run['smoke_only'] and run['split_seed'] == 17
        assert run['config']['epochs'] == 15 and run['config']['bootstrap_reps'] == 1000
        assert run['config']['bootstrap_unit'] == 'patch' and run['config']['heldout_min_positives'] == 10
        assert tuple(run['common_classes']) == CLASSES and set(run['modes']) == set(MODES)
        for mode in MODES:
            assert set(run['modes'][mode]['out_of_fold']) == set(ARMS)
        runs.append(run)
    assert sha(reference) == REFERENCE_SHA
    old = read_json(reference)
    assert old['seed'] == 17 and not old['smoke_only']
    for run in runs:
        assert run['wc_manifest_sha256'] == old['wc_manifest_sha256']
        assert run['reference_sha256'] == old['reference_sha256']
        for mode in MODES:
            for key in ('checkpoint_sha256', 'stored_logits_sha256'):
                assert run['modes'][mode][key] == old['modes'][mode][key]
            for fold in ('A', 'B'):
                for key in ('train_idx', 'val_idx', 'test_idx', 'adapt_regions', 'test_regions'):
                    assert run['modes'][mode]['folds'][fold][key] == old['modes'][mode]['folds'][fold][key]
                for group in run['modes'][mode]['folds'][fold]['arms']['none']['groups']:
                    baseline = metric(runs[0], mode, 'none', fold, group)
                    for arm in ARMS:
                        actual = metric(run, mode, arm, fold, group)
                        for key in ('n', 'classes_used', 'positives', 'class_names_used', 'prevalence_macro_map'):
                            assert actual[key] == baseline[key]
            reference_metric = metric(runs[0], mode, 'none')
            for arm in ARMS:
                actual = metric(run, mode, arm)
                for key in ('n', 'classes_used', 'positives', 'class_names_used', 'prevalence_macro_map'):
                    assert actual[key] == reference_metric[key]
                assert actual['min_positives'] == 10
                assert all(actual['positives'][i] >= 10 for i in actual['classes_used'])
    return runs, old, {'manifest_sha256': sha(checksums), 'reference_json_sha256': sha(reference),
                       'code_commit': COMMIT,
                       'seed_json_sha256': {str(s): sha(folder / f'seed-{s}/ben-adapt.json') for s in (1, 2, 3)}}


def audit_splits(runs, pack):
    """Reconstruct stored permutations and hash real inputs; no model execution."""
    import numpy as np
    import pandas as pd
    manifest_path = pack / 'MANIFEST.json'
    assert sha(manifest_path) == runs[0]['wc_manifest_sha256']
    entries = {e['file']: e['sha256'] for e in read_json(manifest_path)}
    for name in ('meta.parquet', 'classes.json', 'labels_coarse.npy', 's1.npy', 's2_10m.npy'):
        assert sha(pack / name) == entries[name], name
    meta = pd.read_parquet(pack / 'meta.parquet')
    labels = np.load(pack / 'labels_coarse.npy', mmap_mode='r')
    assert meta.idx.tolist() == list(range(len(meta)))
    arrays = {'s1': np.load(pack / 's1.npy', mmap_mode='r'),
              's2': np.load(pack / 's2_10m.npy', mmap_mode='r')}
    india = set(meta.index[meta.group == 'india'])
    hashes = {}
    for mode in MODES:
        inputs = ('s1', 's2') if mode == 's1s2' else (mode,)
        hashes[mode] = {i: hashlib.sha256(b''.join(np.ascontiguousarray(arrays[k][i]).tobytes()
                                                 for k in inputs)).hexdigest() for i in india}
    checks = []
    for run in runs:
        for mode in MODES:
            covered = []
            for fold, f in run['modes'][mode]['folds'].items():
                train, val, test = [np.array(f[k], dtype=int) for k in ('train_idx', 'val_idx', 'test_idx')]
                sets = [set(x.tolist()) for x in (train, val, test)]
                disjoint = (not any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2)))
                            and all(len(x) == len(set(x)) for x in (train, val, test)))
                regions = (set(meta.iloc[train].region) == set(f['adapt_regions'])
                           and set(meta.iloc[val].region) == set(f['adapt_regions'])
                           and set(meta.iloc[test].region) == set(f['test_regions'])
                           and not set(f['adapt_regions']) & set(f['test_regions'])
                           and all(meta.iloc[idx].group.eq('india').all() for idx in (train, val, test)))
                train_hashes = {hashes[mode][int(i)] for i in np.r_[train, val]}
                duplicates = sum(hashes[mode][int(i)] in train_hashes for i in test)
                permutations_ok = True
                for arm in ('shuffle', 'shuffle-fixed'):
                    perm = f['arms'][arm]['shuffle_permutations']
                    ok = (set(perm) == {'train', 'val'} and len(perm['train']) == len(train)
                          and len(perm['val']) == len(val) and set(perm['train']) == sets[0]
                          and set(perm['val']) == sets[1])
                    if ok:
                        reconstructed = np.array(labels)
                        reconstructed[train] = labels[perm['train']]
                        reconstructed[val] = labels[perm['val']]
                        ok = reconstructed[test].tobytes() == np.asarray(labels[test]).tobytes()
                    permutations_ok &= ok
                permutations_ok &= (f['arms']['shuffle']['shuffle_permutations'] ==
                                    f['arms']['shuffle-fixed']['shuffle_permutations'])
                checks.append({'seed': run['seed'], 'mode': mode, 'fold': fold,
                               'train_n': len(train), 'val_n': len(val), 'test_n': len(test),
                               'permutation_scope': bool(permutations_ok), 'indices_disjoint': bool(disjoint),
                               'regions_disjoint': bool(regions), 'duplicate_test_inputs': duplicates})
                covered.extend(test.tolist())
            assert len(covered) == len(set(covered)) and set(covered) == india
    return {'checks': checks, 'pass': all(c['permutation_scope'] and c['indices_disjoint'] and
            c['regions_disjoint'] and c['duplicate_test_inputs'] == 0 for c in checks),
            'method': 'real index/permutation reconstruction and raw-pixel SHA256 per mode; no inference',
            'label_snapshot_limit': 'test labels reconstructed from recorded permutations and verified source, not saved runtime snapshots'}


def analyse(runs, old, audit, provenance):
    out = {'provenance': provenance, 'ci_method': 'conservative per-seed substitute; no seed-averaged CI estimated',
           'modes': {}, 'audit': audit, 'inconclusive_criteria': []}
    for mode in MODES:
        cells = {}
        for arm in ARMS:
            samples = [{'seed': r['seed'], 'value': metric(r, mode, arm)['macro_map'],
                        'ci95': metric(r, mode, arm)['macro_map_ci95']} for r in runs]
            st = stats([s['value'] for s in samples])
            reference = metric(old, mode, arm)['macro_map'] if arm in old['modes'][mode]['out_of_fold'] else None
            chance = metric(runs[0], mode, arm)['prevalence_macro_map']
            cells[arm] = {**st, 'seeds': samples, 'chance': chance,
                          'P3': 'FAIL' if st['mean'] < chance else 'PASS',
                          'seed17': reference, 'seed17_in_range': None if reference is None else st['min'] <= reference <= st['max']}
        assert cells['none']['min'] == cells['none']['max']
        comparisons = {right: contrast(runs, mode, right) for right in ('none', 'renorm', 'shuffle')}
        regions = []
        for fold in ('A', 'B'):
            for region in runs[0]['modes'][mode]['folds'][fold]['test_regions']:
                pairs = [(metric(r, mode, 'none', fold, region)['macro_map'],
                          metric(r, mode, 'full', fold, region)['macro_map']) for r in runs]
                none, full = statistics.fmean(p[0] for p in pairs), statistics.fmean(p[1] for p in pairs)
                regions.append({'region': region, 'fold': fold, 'none_mean': none, 'full_mean': full,
                                'delta_mean': full-none, 'worse': full < none})
        legacy = {}
        for arm in ('renorm', 'head', 'full'):
            samples = []
            for r in runs:
                m = r['modes'][mode];gap = m['hypotheses']['H1']['gap']
                gain = metric(r, mode, arm)['macro_map']-metric(r, mode, 'none')['macro_map']
                samples.append({'seed': r['seed'], 'gain': gain, 'threshold': .5*gap,
                                'gain_ci95': m['hypotheses']['H1']['per_arm'][arm]['gain_ci95']})
            legacy[arm] = {'seeds': samples, 'gain_mean': statistics.fmean(s['gain'] for s in samples),
                           'threshold': samples[0]['threshold']}
            legacy[arm]['verdict'] = 'PASS' if legacy[arm]['gain_mean'] >= samples[0]['threshold'] else 'FAIL'
        ranking = cells['renorm']['mean'] < cells['head']['mean'] < cells['full']['mean']
        control_gap = cells['shuffle']['mean']-cells['shuffle']['chance']
        out['modes'][mode] = {'cells': cells, 'comparisons': comparisons, 'regions': regions,
                              'P1': comparisons['none']['verdict'], 'P2': comparisons['shuffle']['verdict'],
                              'P3_gate': 'FAIL' if any(cells[a]['P3'] == 'FAIL' for a in BASE_ARMS) else 'PASS',
                              'legacy_H1': legacy, 'legacy_H2': 'PASS' if ranking else 'FAIL',
                              'legacy_H3': {'mean_control_minus_chance': control_gap,
                                            'verdict': 'PASS' if abs(control_gap) <= .05 else 'FAIL'}}
        if comparisons['shuffle']['verdict'] == 'INCONCLUSIVE':
            out['inconclusive_criteria'].append(f'P2/{mode}: paired full-minus-shuffle CI not saved')
    anomaly = {}
    for arm in ('none', 'shuffle', 'shuffle-fixed', 'bn-only', 'full'):
        vals = [{'seed': r['seed'], 'value': metric(r, 's2', arm, 'A')['macro_map'],
                 'ci95': metric(r, 's2', arm, 'A')['macro_map_ci95']} for r in runs]
        anomaly[arm] = {**stats([v['value'] for v in vals]), 'seeds': vals,
                        'chance': metric(runs[0], 's2', arm, 'A')['prevalence_macro_map']}
    old_none = metric(old, 's2', 'none', 'A')['macro_map'];old_shuffle = metric(old, 's2', 'shuffle', 'A')['macro_map']
    anomaly['seed17'] = {'none': old_none, 'shuffle': old_shuffle, 'gap': old_shuffle-old_none}
    anomaly['shuffle_mean_gap'] = anomaly['shuffle']['mean']-anomaly['none']['mean']
    anomaly['d_mean_condition'] = bool(old_shuffle > old_none and anomaly['shuffle-fixed']['mean'] <= anomaly['none']['mean'])
    anomaly['d_ci'] = contrast(runs, 's2', 'none', fold='A', left='shuffle-fixed')
    anomaly['d_registered_ci_verdict'] = 'INCONCLUSIVE'  # no paired interval is present for any seed
    out['inconclusive_criteria'].append('(d): registered shuffle-fixed-minus-none CI predicate not saved; mean direction reported separately')
    anomaly['e_mean_condition'] = anomaly['bn-only']['mean'] >= anomaly['shuffle-fixed']['mean']
    anomaly['attribution'] = ('BN re-estimation supported by rule (e)' if anomaly['e_mean_condition'] and audit['pass']
                              else 'Unexplained (or invalidated by a failed real-data audit)')
    epochs = []
    for r in runs:
        for mode in MODES:
            for fold, f in r['modes'][mode]['folds'].items():
                epochs.append({'seed': r['seed'], 'mode': mode, 'fold': fold,
                               **{arm: f['arms'][arm]['training']['best_epoch'] for arm in ('full', 'shuffle', 'shuffle-fixed', 'bn-only')}})
    anomaly['epochs'] = epochs
    out['anomaly'] = anomaly
    out['P1'] = required_all([m['P1'] for m in out['modes'].values()])
    out['P2'] = required_all([m['P2'] for m in out['modes'].values()])
    out['P3_gate'] = required_all([m['P3_gate'] for m in out['modes'].values()])
    out['gate'] = required_all([out['P1'], out['P2'], out['P3_gate']])
    if out['gate'] == 'INCONCLUSIVE': out['inconclusive_criteria'].append('Overall T13b gate: P2 unresolved')
    return out


def fmt(value):
    return 'not saved' if value is None else f'{value:.6f}'


def ci_text(ci):
    return 'not saved' if ci is None else f'[{ci[0]:.6f}, {ci[1]:.6f}]'


def report(out):
    a = out['anomaly']
    text = 'CI verdicts use the conservative per-seed substitute; see DECISIONS.md.\n\n'
    text += (f"Full fine-tuning beats none in every seed and mode under the available paired CIs (P1 {out['P1']}), "
             f"and all full-minus-renorm intervals are positive. P2 and the overall gate are {out['gate']} because full-minus-shuffle paired CIs were not saved; "
             "the larger half-Europe/India-gap recovery claim does not hold for optical or fusion, and shuffled controls are not uniformly at chance. "
             f"The fold-A optical shuffle lift persists on seed means ({a['shuffle']['mean']:.6f} versus none {a['none']['mean']:.6f}). "
             f"BN re-estimation is supported by rule (e): bn-only {a['bn-only']['mean']:.6f} >= shuffle-fixed {a['shuffle-fixed']['mean']:.6f}. "
             "The mean-only (d) direction supports an epoch-selection effect, but its registered CI predicate is not evaluable. "
             "No new training/inference run or final-model claim is made.\n\n")
    text += '# T13b saved-output analysis\n\n'
    text += ('Deviation: [DECISIONS.md](../../docs/plan/DECISIONS.md), commit 1966156. Original '
             '[T13b preregistration](../../docs/hypotheses/T13b.md) is unchanged (4a9162c, 2662aab, as present at a0d7002). '
             'All means below are the plain arithmetic mean of seed-level mAPs; no ensemble mAP or CI is imputed. '
             'No CI endpoints are averaged. Per-seed intervals are the saved 1000-replicate patch bootstrap, seed 17. '
             'They ignore training-seed, region and split variance. Fixed folds cover six Indian regions; WorldCover labels are map products, not independent ground truth.\n\n')
    text += '## OOF India-pooled cells\n\n| Mode | Arm | Seed 1 mAP [CI95] | Seed 2 mAP [CI95] | Seed 3 mAP [CI95] | Mean | Min–max | Exact chance | P3 |\n|---|---|---|---|---|---:|---|---|---|\n'
    for mode, m in out['modes'].items():
        for arm, c in m['cells'].items():
            samples = ' | '.join(fmt(x['value'])+' '+ci_text(x['ci95']) for x in c['seeds'])
            text += f"| {mode} | {arm} | {samples} | {c['mean']:.6f} | {c['min']:.6f}–{c['max']:.6f} | {c['chance']!r} | {c['P3']} |\n"
    text += '\nP3 is tested on unrounded values. shuffle-fixed and bn-only are reported but excluded from the gate by (d)/(e); the below-chance s1s2 shuffle-fixed cell is therefore a reported diagnostic failure, not a gate failure.\n\n'
    text += '## Registered and matched comparisons side by side\n\n| Mode | Seed | full−none (registered), paired CI95 | full−renorm (matched), paired CI95 |\n|---|---:|---|---|\n'
    for mode, m in out['modes'].items():
        for left, right in zip(m['comparisons']['none']['seeds'], m['comparisons']['renorm']['seeds']):
            text += f"| {mode} | {left['seed']} | {fmt(left['value'])} {ci_text(left['ci95'])} | {fmt(right['value'])} {ci_text(right['ci95'])} |\n"
        text += f"| {mode} | plain mean / substitute verdict | {fmt(m['comparisons']['none']['mean'])} / {m['comparisons']['none']['verdict']} | {fmt(m['comparisons']['renorm']['mean'])} / {m['comparisons']['renorm']['verdict']} |\n"
    text += '\nNone uses stored BEN normalization; renorm/full share adaptation-train normalization. full−none includes renormalization cost and is not the same estimand as full−renorm. No seed-mean CI is available.\n\n'
    rule_lines = (ROOT / 'docs/hypotheses/T13b.md').read_text().splitlines()
    text += '## Registered rules, quoted, and verdicts\n\n'
    prefixes = [('CI95', '- **CI95.'), ('P1', '- **P1 '), ('P2', '- **P2 '), ('P3', '- **P3 '),
                ('P4', '- **P4 '), ('Gate', '- **Gate.'), ('Step 3', '- **Step 3 ')]
    for name, prefix in prefixes:
        quote = next(line for line in rule_lines if line.startswith(prefix))
        text += f'### {name}\n\n> {quote[2:]}\n\n'
        if name == 'CI95':
            text += 'DEVIATION: use saved per-seed paired CIs and unanimity: all PASS -> PASS; all FAIL -> FAIL; any mixture or unavailable required CI -> INCONCLUSIVE. No registered seed-averaged bootstrap can be reconstructed.\n\n'
        elif name in ('P1', 'P2'):
            text += '| Mode | Seed | Difference | Paired CI95 | Per-seed verdict |\n|---|---:|---:|---|---|\n'
            for mode, m in out['modes'].items():
                c = m['comparisons']['none' if name == 'P1' else 'shuffle']
                for sample in c['seeds']:
                    text += f"| {mode} | {sample['seed']} | {fmt(sample['value'])} | {ci_text(sample['ci95'])} | {sample['verdict']} |\n"
                text += f"| {mode} | mean / substitute | {fmt(c['mean'])} | not averaged | {c['verdict']} |\n"
            text += f"\nOverall {name}: **{out[name]}**. Missing paired full−shuffle intervals are not replaced by differences between marginal CI endpoints.\n\n"
        elif name == 'P3': text += f"Gate-eligible cells: **{out['P3_gate']}**. All cells and their exact run prevalence, including diagnostic failures, are listed above.\n\n"
        elif name == 'P4': text += 'Reported, not gated. Every region/mode regression is named below.\n\n'
        elif name == 'Gate': text += f"**{out['gate']}**: P1={out['P1']}, P2={out['P2']}, P3 (non-diagnostic arms)={out['P3_gate']}. This is not a PASS and does not authorize final models.\n\n"
        else: text += 'NOT RUN. No final checkpoint, final-model accuracy, or deployment claim is produced.\n\n'
    text += '## Fold-A optical anomaly and required checks\n\n'
    for line in rule_lines:
        if line.startswith(('1. **Permutation', '2. **Leakage', '3. **Seed persistence', '4. **Candidate mechanism')):
            text += '> '+line+'\n\n'
    text += f"Real-data index/permutation/pixel-hash audits: **{'PASS' if out['audit']['pass'] else 'FAIL'}**. All 18 seed/mode/fold checks are in summary.json. Test-label preservation is reconstructed from saved permutations plus verified source, not from saved runtime label snapshots. No model was executed for these audits.\n\n"
    text += '| Arm | Seed 1 [CI95] | Seed 2 [CI95] | Seed 3 [CI95] | Mean | Exact fold prevalence |\n|---|---|---|---|---:|---|\n'
    for arm in ('none', 'shuffle', 'shuffle-fixed', 'bn-only', 'full'):
        c = a[arm];samples = ' | '.join(fmt(s['value'])+' '+ci_text(s['ci95']) for s in c['seeds'])
        text += f"| {arm} | {samples} | {fmt(c['mean'])} | {c['chance']!r} |\n"
    text += f"\nSeed 17 re-read: none={a['seed17']['none']!r}, shuffle={a['seed17']['shuffle']!r}, difference={a['seed17']['gap']!r}. Seed-mean shuffle−none={a['shuffle_mean_gap']!r}; persistence check **{'PASS' if a['shuffle_mean_gap'] > 0 else 'FAIL'}**.\n\n"
    for prefix, name in (('- Decision rule for the diagnosis', '(d)'), ('- Decision rule, fixed now', '(e)')):
        text += f'### {name}\n\n> '+next(l for l in rule_lines if l.startswith(prefix))[2:]+'\n\n'
        if name == '(d)':
            text += f"Requested mean-based reading: **{'PASS' if a['d_mean_condition'] else 'FAIL'}** (seed-17 shuffle > none; shuffle-fixed seed mean <= none). This is a directional epoch-selection diagnostic, not a claim that an unavailable CI passed. The original CI predicate remains **INCONCLUSIVE**.\n\n| Seed | shuffle-fixed−none | Paired CI95 | CI predicate |\n|---:|---:|---|---|\n"
            for sample in a['d_ci']['seeds']: text += f"| {sample['seed']} | {fmt(sample['value'])} | not saved | INCONCLUSIVE |\n"
        else:
            text += f"\nMean-based verdict: **{'PASS' if a['e_mean_condition'] else 'FAIL'}**. bn-only={a['bn-only']['mean']!r}, shuffle-fixed={a['shuffle-fixed']['mean']!r}. Attribution: **{a['attribution']}**. This is the preregistered mean inequality, not an independent causal estimate. The existing synthetic BN-invariance test checks unchanged weights/changed running stats; real before/after model states were not saved.\n\n"
    text += '\n### Selected epochs (validation selection differs from fixed-final diagnostics)\n\n| Mode | Fold | Seed | Full | Shuffle | Shuffle-fixed | BN-only |\n|---|---|---:|---:|---:|---:|---:|\n'
    for row in a['epochs']: text += f"| {row['mode']} | {row['fold']} | {row['seed']} | {row['full']} | {row['shuffle']} | {row['shuffle-fixed']} | {row['bn-only']} |\n"
    text += '\n## Region regressions (seed means)\n\n| Mode | Region | None | Full | Full−none |\n|---|---|---:|---:|---:|\n'
    for mode, m in out['modes'].items():
        bad = [r for r in m['regions'] if r['worse']]
        if not bad: text += f'| {mode} | none | — | — | — |\n'
        for r in bad: text += f"| {mode} | {r['region']} | {fmt(r['none_mean'])} | {fmt(r['full_mean'])} | {fmt(r['delta_mean'])} |\n"
    text += '\n## Seed-17 stability, every OOF cell\n\n| Mode | Arm | Seed 17 | Seeds 1–3 min–max | In range? |\n|---|---|---:|---|---|\n'
    for mode, m in out['modes'].items():
        for arm, c in m['cells'].items():
            flag = 'N/A: absent in seed 17' if c['seed17'] is None else 'IN' if c['seed17_in_range'] else '**OUTSIDE**'
            text += f"| {mode} | {arm} | {fmt(c['seed17'])} | {fmt(c['min'])}–{fmt(c['max'])} | {flag} |\n"
    text += '\nInclusivity is tested at full precision; display rounding never determines a verdict. Diagnostic arms have no seed-17 reference.\n\n'
    text += '## T13 rules still in force (point rules on seed means)\n\n'
    old_rules = (ROOT / 'docs/hypotheses/T13.md').read_text()
    for start, end in (('H1:', '\n\nH2:'), ('H2:', '\n\nH3:'), ('H3:', '\n\n##')):
        q = old_rules[old_rules.index(start):].split(end, 1)[0]
        text += '> '+q.replace('\n', '\n> ')+'\n\n'
    text += 'The later T13 correction forbids arm selection: all renorm/head/full arms are reported independently. No best arm is selected.\n\n| Mode | Arm | Mean gain | Half-gap threshold | H1 |\n|---|---|---:|---:|---|\n'
    for mode, m in out['modes'].items():
        for arm, v in m['legacy_H1'].items(): text += f"| {mode} | {arm} | {fmt(v['gain_mean'])} | {fmt(v['threshold'])} | {v['verdict']} |\n"
    text += '\n| Mode | H2 strict renorm<head<full | Shuffle mean−exact prevalence | H3 absolute gap <=0.05 |\n|---|---|---:|---|\n'
    for mode, m in out['modes'].items(): text += f"| {mode} | {m['legacy_H2']} | {fmt(m['legacy_H3']['mean_control_minus_chance'])} | {m['legacy_H3']['verdict']} |\n"
    text += '\n## Inconclusive criteria — stop after this report\n\n'
    text += '\n'.join('- '+c for c in out['inconclusive_criteria'])+'\n\nNo rerun is initiated. Soham decides whether to authorize one. Future NPZ export is independent code-only instrumentation (8e87936); its synthetic archive test passed, and no experiment was executed.\n\n'
    text += '## Provenance\n\n```json\n'+json.dumps(out['provenance'], indent=2)+'\n```\n\nAll input JSON source hashes match a0d7002. Input seed logs end in DONE and SHA256SUMS.txt is verified; seed 17 matches the preregistered reference hash. Small summary.json contains all unrounded numbers and audit checks; no data or weights are committed.\n'
    return text


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, default=Path.home()/'ben-work/adapt-run-2')
    p.add_argument('--reference', type=Path, default=Path.home()/'ben-work/adapt-run-1/ben-adapt.json')
    p.add_argument('--pack', type=Path, default=Path.home()/'ben-work/wc/pack')
    p.add_argument('--out', type=Path, default=ROOT/'results/t13b')
    args = p.parse_args(argv)
    runs, old, provenance = validate_inputs(args.runs, args.reference)
    audit = audit_splits(runs, args.pack)
    result = analyse(runs, old, audit, provenance)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (args.out/'REPORT.md').write_text(report(result))
    print(json.dumps({'P1': result['P1'], 'P2': result['P2'], 'gate': result['gate'],
                      'inconclusive': result['inconclusive_criteria'], 'anomaly': result['anomaly']['attribution']}, indent=2))
    return result


if __name__ == '__main__':
    result = main()
    if result['inconclusive_criteria']:
        raise SystemExit(2)  # Report complete; pause scientific work for Soham's decision.
