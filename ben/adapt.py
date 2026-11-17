"""T13: region-held-out Indian adaptation. WorldCover labels are map products.

Run: python ben/adapt.py --wc-pack PACK --runs RUNS --out-dir OUT
A complete saved-logit baseline reproduction is mandatory, even for CPU smoke.
"""
import argparse
import json
import os
import random
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ben import landcover as lc, transfer as tr  # noqa: E402

COMMON = ['trees', 'shrub_grass', 'cropland', 'built', 'water']
FOLDS = {'A': (('punjab', 'kerala', 'assam'), ('delhi', 'sundarbans', 'rajasthan')),
         'B': (('delhi', 'sundarbans', 'rajasthan'), ('punjab', 'kerala', 'assam'))}
ARMS = ('none', 'renorm', 'head', 'full', 'shuffle', 'shuffle-fixed', 'bn-only')  # diagnostics: docs/hypotheses/T13b.md (d), (e)


def seed_everything(seed):
    """Training randomness only. Fold splits always use lc.SEED (fold_indices)."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False


def fold_indices(meta, fold, limit=None):
    adapt_regions, test_regions = FOLDS[fold]
    assert set(adapt_regions).isdisjoint(test_regions)
    rng = np.random.default_rng(lc.SEED)
    train, val, test = [], [], []
    for region in sorted(adapt_regions + test_regions):
        indices = np.flatnonzero(meta.region.to_numpy() == region)
        if len(indices) < 5:
            raise ValueError(f'STOP: too few patches in {region}')
        indices = rng.permutation(indices)
        if limit:
            indices = indices[:limit]
        if region in test_regions:
            test.extend(indices)
        else:
            cut = int(.8 * len(indices))
            train.extend(indices[:cut]); val.extend(indices[cut:])
    train, val, test = [np.sort(v).astype(int) for v in (train, val, test)]
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    assert set(meta.iloc[train].group) == {'india'}
    assert set(meta.iloc[val].group) == {'india'}
    assert set(meta.iloc[test].group) == {'india'}
    if 'patch_id' in meta:
        assert set(meta.iloc[np.r_[train,val]].patch_id).isdisjoint(meta.iloc[test].patch_id)
    if {'utm_epsg', 'utm_x', 'utm_y'} <= set(meta.columns):
        coords = ['utm_epsg', 'utm_x', 'utm_y']
        adapt_coords = set(meta.iloc[np.r_[train,val]][coords].itertuples(index=False, name=None))
        test_coords = set(meta.iloc[test][coords].itertuples(index=False, name=None))
        assert adapt_coords.isdisjoint(test_coords), 'STOP: geographic patch overlap'
    return train, val, test


def check_baseline(y, meta, info, logits, members, reference):
    """Reproduce both original six-class rows and five-common-class pooled rows."""
    cols = [info['coarse_classes'].index(c) for c in members]
    yy, scores = y[:, cols], tr.to_coarse(logits, members)
    if len(logits) != len(meta) or logits.shape[1] != sum(len(v) for v in members.values()) + len(reference['unmapped_ben_classes']):
        raise ValueError('STOP: saved baseline logits have incompatible dimensions')
    checks = {}
    for group in ('india', 'europe', *sorted(set(meta.region))):
        mask = (meta.group == group).to_numpy() if group in ('india','europe') else (meta.region == group).to_numpy()
        keep = yy[mask].sum(0) >= tr.MIN_POSITIVES
        actual = lc.macro_map(yy[mask][:,keep], scores[mask][:,keep])[0] if keep.any() else None
        expected = reference['groups'][group]['macro_map']
        if actual is None or expected is None or abs(actual-expected) > 1e-4:
            raise ValueError(f'STOP: arm none reproduction failed {group}: {actual} vs {expected}')
        checks[group] = {'actual':actual, 'reference':expected, 'difference':actual-expected}
    common = [list(members).index(c) for c in COMMON]
    for group in ('india','europe'):
        mask = (meta.group == group).to_numpy()
        actual = lc.macro_map(yy[mask][:,common], scores[mask][:,common])[0]
        expected = reference['groups_common'][group]['macro_map']
        if reference['common_classes'] != COMMON or abs(actual-expected) > 1e-4:
            raise ValueError(f'STOP: arm none common reproduction failed {group}: {actual} vs {expected}')
        checks[group+'_common'] = {'actual':actual,'reference':expected,'difference':actual-expected}
    return checks


def shuffled_labels(labels, train, val, seed=lc.SEED):
    shuffled = labels.copy()
    rng = np.random.default_rng(seed)
    permutations = {}
    for name, idx in (('train',train),('val',val)):
        source = rng.permutation(idx)
        shuffled[idx] = labels[source]
        permutations[name] = source.tolist()
    return shuffled, permutations


def validation_eligibility(labels, val, members):
    cols = [list(members).index(c) for c in COMMON]
    keep = labels[val][:,cols].sum(0) > 0
    if not keep.any():
        raise ValueError('STOP: validation split has zero common classes with any positive')
    return cols, keep, [c for c, eligible in zip(COMMON,keep) if eligible]


def fit(model, arrays, stats, labels, train, val, members, arm, epochs, batch_size, device, seed=lc.SEED, select_by_val=True):
    """Never accepts held-out indices. Freeze the backbone including BN buffers for head."""
    seed_everything(seed)
    vc, val_keep, val_classes = validation_eligibility(labels,val,members)
    used = labels[train].sum(0) > 0
    if not used.any():
        raise ValueError('STOP: no positive adaptation-train classes')
    active = {k:v for (k,v),keep in zip(members.items(),used) if keep}
    active_rows = {i for vv in active.values() for i in vv}
    unused = sorted(set(range(model.fc.out_features))-active_rows)
    frozen_w = model.fc.weight.detach()[unused].clone()
    frozen_b = model.fc.bias.detach()[unused].clone()
    for name, p in model.named_parameters():
        p.requires_grad_(arm != 'head' or name.startswith('fc.'))
    lr = 1e-3 if arm == 'head' else 1e-4
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    loss_fn = nn.BCEWithLogitsLoss()
    rng, gen = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
    best_map, best_state, best_epoch = -1., None, None
    history = []
    for epoch in range(1,epochs+1):
        model.train()
        if arm == 'head':
            model.eval(); model.fc.train()
        order = rng.permutation(train)
        total = 0.
        for start in range(0,len(order),batch_size):
            idx = np.sort(order[start:start+batch_size])
            x = lc.augment(lc.make_batch(arrays,stats,idx,device),gen)
            target = torch.from_numpy(labels[idx][:,used]).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == 'cuda'):
                loss = loss_fn(tr.coarse_logits(model(x).float(),active),target)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            # AdamW decays zero-gradient rows; restore all inactive classifier rows.
            with torch.no_grad():
                model.fc.weight[unused] = frozen_w; model.fc.bias[unused] = frozen_b
            total += float(loss.detach()) * len(idx)
        sched.step()
        val_scores = tr.to_coarse(lc.predict(model,arrays,stats,val,device,batch_size),members)
        value = lc.macro_map(labels[val][:,vc][:,val_keep],val_scores[:,vc][:,val_keep])[0]
        history.append({'epoch':epoch,'loss':total/len(train),'val_map':value})
        if select_by_val and value > best_map:
            best_map, best_epoch = value, epoch
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f'{arm} epoch {epoch}/{epochs} val={value:.6f}',flush=True)
    if not select_by_val:  # shuffle-fixed diagnostic: keep the final epoch, no validation choice
        best_epoch, best_map = epochs, history[-1]['val_map']
        best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {'best_epoch':best_epoch,'best_val_map':best_map,'history':history,'lr':lr,
            'loss_classes':list(active),'inactive_classifier_rows':unused,'validation_classes':val_classes}


def bn_recalibrate(model, arrays, stats, train, epochs, batch_size, device, seed=lc.SEED):
    """bn-only diagnostic (docs/hypotheses/T13b.md (e)): refresh BatchNorm running stats on adaptation-train
    images in train mode, same epochs and batches as the other trained arms. No labels, no weight updates."""
    seed_everything(seed)
    rng, gen = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    model.train()
    with torch.no_grad():
        for _ in range(epochs):
            order = rng.permutation(train)
            for start in range(0,len(order),batch_size):
                idx = np.sort(order[start:start+batch_size])
                x = lc.augment(lc.make_batch(arrays,stats,idx,device),gen)
                with torch.autocast(device.type, enabled=device.type == 'cuda'):
                    model(x)
    model.eval()
    return {'best_epoch':epochs,'best_val_map':None,'history':[],'lr':None,'loss_classes':None,
            'inactive_classifier_rows':[],'validation_classes':None}


def scored(y, scores):
    m, _ = tr.group_metrics(y,scores)
    m['class_names_used'] = [COMMON[i] for i in m['classes_used']]
    return m



def prediction_block(meta, indices, fold, mode, arm, scores, labels):
    """Keep test rows in prediction order; IDs refer to the manifest's global idx."""
    rows = meta.iloc[indices]
    scores, labels = np.asarray(scores), np.asarray(labels)
    if scores.shape != (len(rows), len(COMMON)) or labels.shape != scores.shape:
        raise ValueError('STOP: prediction/label shape mismatch')
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)) or not np.isin(labels, [0, 1]).all():
        raise ValueError('STOP: invalid coarse scores or test labels')
    return {'patch_id': np.array([f'worldcover:{int(i)}' for i in rows.idx]),
            'patch_idx': rows.idx.to_numpy(dtype=np.int64),
            'fold': np.full(len(rows), fold), 'mode': np.full(len(rows), mode),
            'arm': np.full(len(rows), arm), 'scores': scores.astype(np.float32),
            'labels': labels.astype(np.uint8)}


def save_test_predictions(path, blocks, seed, manifest_sha256, code_commit):
    """One pickle-free compressed archive per seed; never overwrite prior output."""
    if not blocks:
        raise ValueError('STOP: no test predictions')
    data = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
    keys = list(zip(data['patch_id'], data['fold'], data['mode'], data['arm']))
    if len(set(keys)) != len(keys):
        raise ValueError('STOP: duplicate test prediction key')
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **data, classes=np.array(COMMON), seed=np.array(seed),
                            wc_manifest_sha256=np.array(manifest_sha256), code_commit=np.array(code_commit))
    return {'file': Path(path).name, 'sha256': lc.sha256(path), 'rows': len(keys),
            'classes': list(COMMON), 'patch_id_format': 'worldcover:<global idx>'}


def table(out):
    lines = ['# T13 India adaptation', '', 'WorldCover map labels; smoke runs are not scientific results.', '',
             '| Mode | Fold | Arm | Group | n | mAP | CI95 | Prevalence | Shuffled-score chance | Original mAP |',
             '|---|---|---|---|---:|---:|---|---:|---:|---:|']
    for mode,m in out['modes'].items():
        for fold,f in m['folds'].items():
            for arm,a in f['arms'].items():
                for group,g in a['groups'].items():
                    lines.append(f'| {mode} | {fold} | {arm} | {group} | {g["n"]} | {g["macro_map"]} | {g["macro_map_ci95"]} | {g["prevalence_macro_map"]} | {g["shuffled_macro_map"]} | {f["arms"]["none"]["groups"][group]["macro_map"]} |')
            # Both comparisons per fold, pooled test set (point differences; T13b (f)).
            pf = {a: f['arms'][a]['groups']['pooled']['macro_map'] for a in f['arms']}
            lines.append(f'| {mode} | {fold} | full minus none (registered) | pooled | | {pf["full"]-pf["none"]} | - | | | |')
            lines.append(f'| {mode} | {fold} | full minus renorm (matched normalisation) | pooled | | {pf["full"]-pf["renorm"]} | - | | | |')
        for arm,g in m['out_of_fold'].items():
            lines.append(f'| {mode} | OOF | {arm} | india | {g["n"]} | {g["macro_map"]} | {g["macro_map_ci95"]} | {g["prevalence_macro_map"]} | {g["shuffled_macro_map"]} | {m["out_of_fold"]["none"]["macro_map"]} |')
        if 'matched' in m:  # both comparisons side by side (T13b (f)); none is the registered reference
            gain = m['hypotheses']['H1']['per_arm']['full']
            lines.append(f'| {mode} | OOF | full minus none (registered) | india | | {gain["gain"]} | {gain["gain_ci95"]} | | | |')
            lines.append(f'| {mode} | OOF | full minus renorm (matched normalisation) | india | | {m["matched"]["full_minus_renorm"]} | {m["matched"]["ci95"]} | | | |')
    lines += ['', 'Bootstrap unit: patch (existing transfer bootstrap), not new regions. Classes require >=10 positives per group.',
              'H1/H2/H3 decisions and exact configuration are in JSON. Null results are reported as-is.']
    return '\n'.join(lines)+'\n'


def run(args):
    start = time.time()
    if not 1 <= args.epochs <= 15 or (args.limit_per_region is not None and args.limit_per_region < 20):
        raise ValueError('STOP: epochs must be 1..15 and smoke limit >=20')
    lc.seed_all()
    pack, runs, dest = Path(args.wc_pack), Path(args.runs), Path(args.out_dir)
    prediction_path = dest/f'ben-adapt-predictions-seed-{args.train_seed}.npz'
    if (dest/'ben-adapt.json').exists() or prediction_path.exists():
        raise ValueError('STOP: output already exists')
    manifest_hash = tr.verify_wc_manifest(pack)
    raw, yall, meta, info = tr.load_wc(pack)
    fractions = np.load(pack/'fractions.npy')
    if not np.array_equal(yall, fractions >= info['present_fraction']):
        raise ValueError('STOP: WC labels do not match fraction-to-presence rule')
    if list(meta.idx) != list(range(len(meta))) or len(yall) != len(meta):
        raise ValueError('STOP: WC row alignment invalid')
    expected_india = set(FOLDS['A'][0]+FOLDS['A'][1])
    if set(meta.loc[meta.group == 'india','region']) != expected_india:
        raise ValueError('STOP: unexpected India region set')
    reference = json.loads((runs/'ben-transfer.json').read_text())
    if reference['wc_manifest_sha256'] != manifest_hash:
        raise ValueError('STOP: reference uses a different WC pack')
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    out = {'schema_version':1,'seed':args.train_seed,'split_seed':lc.SEED,'config':{**vars(args),'optimizer':'AdamW','loss':'BCEWithLogits on max-member coarse logits',
           'schedule':'cosine','head_lr':1e-3,'full_lr':1e-4,'weight_decay':1e-4,
           'bootstrap_reps':tr.BOOT,'bootstrap_unit':'patch','heldout_min_positives':tr.MIN_POSITIVES,
           'validation_min_positives':1,'arm_selection':False,'torch_version':str(torch.__version__)},'common_classes':COMMON,
           'wc_manifest_sha256':manifest_hash,'reference_sha256':lc.sha256(runs/'ben-transfer.json'),
           'code':{'git_sha':lc.git_sha() or os.environ.get('T13_CODE_COMMIT'),'adapt_py_sha256':lc.sha256(Path(__file__)),
                   'transfer_py_sha256':lc.sha256(Path(tr.__file__)),'landcover_py_sha256':lc.sha256(Path(lc.__file__))},
           'smoke_only':args.limit_per_region is not None,'device':str(device),'modes':{}}
    # All requested modes must pass reproduction BEFORE any adaptation training.
    prepared = {}
    for mode in args.modes:
        path = runs/f'ben-landcover-{mode}-checkpoint.pt'
        ckpt = torch.load(path,map_location='cpu',weights_only=True)
        members,_ = tr.coarse_members(ckpt['classes'])
        logits_path = runs/f'wc-{mode}-logits.npy'
        baseline_logits = np.load(logits_path)
        checks = check_baseline(yall,meta,info,baseline_logits,members,reference['modes'][mode])
        prepared[mode] = (ckpt,members,baseline_logits)
        out['modes'][mode] = {'checkpoint_sha256':lc.sha256(path),'stored_logits_sha256':lc.sha256(logits_path),
                              'reproduction':checks,'coarse_classes':list(members),'folds':{}}
        print(f'{mode}: baseline reproduction PASS ({len(checks)} groups; tolerance 1e-4)',flush=True)
    prediction_blocks = []
    for mode,(ckpt,members,baseline_logits) in prepared.items():
        arrays = {k:raw[k] for k in ckpt['inputs']}
        cols = [info['coarse_classes'].index(c) for c in members]
        y = yall[:,cols]
        common_idx = [list(members).index(c) for c in COMMON]
        pooled = {arm:[] for arm in ARMS}
        all_test = []
        res = out['modes'][mode]
        for fold in FOLDS:
            train,val,test = fold_indices(meta,fold,args.limit_per_region)
            all_test.extend(test)
            stats = lc.train_stats(arrays,train)
            if any(not np.all(np.isfinite(s['std'])) or min(s['std']) <= 0 for s in stats.values()):
                raise ValueError('STOP: invalid renorm standard deviation')
            f = {'train_idx':train.tolist(),'val_idx':val.tolist(),'test_idx':test.tolist(),
                 'adapt_regions':list(FOLDS[fold][0]),'test_regions':list(FOLDS[fold][1]),
                 'renorm_stats':stats,'arms':{}}
            res['folds'][fold] = f
            for arm in ARMS:
                seed_everything(args.train_seed)
                model = lc.build_model(ckpt['in_channels'],len(ckpt['classes'])).to(device)
                model.load_state_dict(ckpt['state_dict'])
                norm = ckpt['preprocess_stats'] if arm == 'none' else stats
                labels, permutations = (shuffled_labels(y,train,val,seed=args.train_seed) if arm in ('shuffle','shuffle-fixed') else (y,None))
                try:
                    _, _, val_classes = validation_eligibility(labels,val,members)
                except ValueError as exc:
                    raise ValueError(f'{exc} (fold={fold}, mode={mode}, arm={arm})') from exc
                training = None
                if arm in ('head','full','shuffle','shuffle-fixed'):
                    training = fit(model,arrays,norm,labels,train,val,members,arm,args.epochs,args.batch_size,device,seed=args.train_seed,
                                   select_by_val=arm != 'shuffle-fixed')
                elif arm == 'bn-only':
                    training = bn_recalibrate(model,arrays,stats,train,args.epochs,args.batch_size,device,seed=args.train_seed)
                # None is exactly the original stored checkpoint predictions, not a rerun.
                logits = baseline_logits[test] if arm == 'none' else lc.predict(model,arrays,norm,test,device,args.batch_size)
                scores = tr.to_coarse(logits,members)[:,common_idx]
                prediction_blocks.append(prediction_block(meta,test,fold,mode,arm,scores,y[test][:,common_idx]))
                groups = {'pooled':scored(y[test][:,common_idx],scores)}
                for region in FOLDS[fold][1]:
                    mask = (meta.iloc[test].region == region).to_numpy()
                    groups[region] = scored(y[test][mask][:,common_idx],scores[mask])
                f['arms'][arm] = {'training':training,'validation_classes':val_classes,'shuffle_permutations':permutations,'groups':groups}
                pooled[arm].append(scores)
                print(f'{mode} fold {fold} {arm}: final pooled mAP={groups["pooled"]["macro_map"]}',flush=True)
                del model
        assert len(all_test) == len(set(all_test))
        yy = y[all_test][:,common_idx]
        res['out_of_fold'] = {a:scored(yy,np.concatenate(v)) for a,v in pooled.items()}
        oof = res['out_of_fold']
        i0 = oof['none']['macro_map']
        e0 = res['reproduction']['europe_common']['actual']
        gap = e0-i0
        evaluable = not out['smoke_only'] and len(oof['none']['classes_used']) == len(COMMON) and gap > 0
        # Identical seeds in the existing bootstrap pair each arm with the original.
        boot = {a:tr.group_metrics(yy,np.concatenate(v))[1] for a,v in pooled.items()}
        recovery = {}
        for arm in ('renorm','head','full'):
            gain = oof[arm]['macro_map']-i0
            recovery[arm] = {'gain':gain, 'recovery_fraction':gain/gap if gap > 0 else None,
                             'gain_ci95':lc.ci95(boot[arm]-boot['none']),
                             'pass':gain >= .5*gap if evaluable else None}
        keep = yy.sum(0) >= tr.MIN_POSITIVES
        chance = np.broadcast_to(yy[:,keep].mean(0),yy[:,keep].shape)
        chance_boot = lc.bootstrap_maps(yy[:,keep],chance,len(yy),tr.BOOT)
        res['hypotheses'] = {'evaluated':not out['smoke_only'],
            'H1':{'evaluable':evaluable, 'gap':gap, 'per_arm':recovery,
                  'any_arm_pass':any(r['pass'] for r in recovery.values()) if evaluable else None,
                  'note':'All arms reported; no arm selection. Per-arm intervals are not multiplicity adjusted.'},
            'H2':{'pass':oof['renorm']['macro_map'] < oof['head']['macro_map'] < oof['full']['macro_map'] if not out['smoke_only'] else None},
            'H3':{'control_minus_prevalence':oof['shuffle']['macro_map']-oof['shuffle']['prevalence_macro_map'],
                  'difference_ci95':lc.ci95(boot['shuffle']-chance_boot),
                  'pass':abs(oof['shuffle']['macro_map']-oof['shuffle']['prevalence_macro_map']) <= .05 if not out['smoke_only'] else None}}
        # Matched normalisation comparison, reported beside the registered full-none (T13b (f)); not a gate.
        res['matched'] = {'full_minus_renorm':oof['full']['macro_map']-oof['renorm']['macro_map'],
                          'ci95':lc.ci95(boot['full']-boot['renorm'])}
    out['runtime_s'] = time.time()-start
    out = tr._finite(out)
    dest.mkdir(parents=True,exist_ok=True)
    out['test_predictions'] = save_test_predictions(prediction_path,prediction_blocks,args.train_seed,manifest_hash,out['code']['git_sha'])
    (dest/'ben-adapt.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    (dest/'ben-adapt.md').write_text(table(out))
    print(f'DONE {dest / "ben-adapt.json"} runtime={out["runtime_s"]:.1f}s smoke_only={out["smoke_only"]}',flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wc-pack',required=True); p.add_argument('--runs',required=True); p.add_argument('--out-dir',required=True)
    p.add_argument('--modes',nargs='+',choices=list(lc.MODES),default=list(lc.MODES))
    p.add_argument('--epochs',type=int,default=15); p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--limit-per-region',type=int); p.add_argument('--device',choices=['cpu','cuda'])
    p.add_argument('--train-seed',type=int,default=lc.SEED,help='training randomness only; folds stay on lc.SEED')
    return run(p.parse_args(argv))


if __name__ == '__main__':
    main()
