"""T15d: four fixed train-only optical-input fits and a saved-checkpoint overlay."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from data.flood import train
from data.flood.build_pack import sha256
from data.flood.learning import Pack, INPUT_CHANNELS, code_info
from data.flood.model import UNet

ROWS = [92, 139, 114, 39, 191, 26, 207, 181]
WORST = 181


def read(path):
    return json.loads(Path(path).read_text())


def config_for(reference, mode, scope):
    cfg = copy.deepcopy(reference['config'])
    expected = {'seed': 17, 'epochs': 400, 'batch_size': 2, 'lr': .003,
                'weight_decay': 0., 'pos_weight': 1., 'threshold': .5,
                'augmentation': False, 'optimizer': 'AdamW', 'schedule': 'cosine', 'control': 'none'}
    if reference['train_indices'] != ROWS or any(cfg[k] != v for k, v in expected.items()):
        raise ValueError('STOP: cosine400 reference recipe changed')
    if mode not in ('s1s2', 's2') or scope not in ('single', 'eight'):
        raise ValueError('STOP: only the four registered T15d fits are allowed')
    cfg.update(mode=mode, tiny_indices=[WORST] if scope == 'single' else ROWS.copy(),
               tiny_early_stop=False, architecture='unet', loss_kind='bce_dice')
    return cfg


def verdict(worst, eight):
    passed = worst > .95 and eight > .9
    return {'D1': 'PASS' if passed else 'FAIL', 'D2': 'not triggered' if passed else 'pipeline cause by registered rule',
            'registered_conclusion': 'pipeline correct; S1-only information limit' if passed else 'cause in pipeline, not inputs',
            'stop': 'Soham decides; no further run or gate change'}


@torch.no_grad()
def prediction(pack, folder, row, dev):
    run = read(folder/'flood-run.json')
    checkpoint = folder/'flood-checkpoint.pt'
    if sha256(checkpoint) != run['checkpoint_sha256'] or run['train_indices'] != [row] or run['pack_manifest_sha256'] != pack.sha:
        raise ValueError('STOP: overlay checkpoint provenance mismatch')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = UNet(INPUT_CHANNELS[run['config']['mode']]).to(dev)
    model.load_state_dict(state['state_dict']); model.eval()
    x, _ = train.batch(pack, 'train', [row], run['config']['mode'], dev)
    return (torch.sigmoid(model(x)) >= .5).cpu().numpy()[0, 0]


def correction_mask(target, s1, s1s2):
    valid = target != -1
    return valid & (s1 != (target == 1)) & (s1s2 == (target == 1))


def overlay(pack, dest, dev):
    old = prediction(pack, dest/'single', WORST, dev)
    new = prediction(pack, dest/'t15d-s1s2-single', WORST, dev)
    arrays = pack.split('train'); target = arrays['label'][WORST, 0]
    bands = pack.manifest['band_order']['s2']
    raw = arrays['s2'][WORST, [bands.index(b) for b in ('B04', 'B03', 'B02')]].astype(np.float32)
    rgb = np.zeros_like(raw)
    for k, band in enumerate(raw):
        finite = np.isfinite(band) & (band != 0)
        lo, hi = np.percentile(band[finite], [2, 98]) if finite.any() else (0., 1.)
        rgb[k] = np.where(finite, np.clip((band-lo)/max(hi-lo, 1e-6), 0, 1), 0)
    rgb = rgb.transpose(1, 2, 0)
    corrected = correction_mask(target, old, new)
    old_wrong = (target != -1) & (old != (target == 1))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax in axes:
        ax.imshow(rgb); ax.axis('off')
    axes[0].set_title('S2 RGB B04/B03/B02 (2–98% stretch)')
    axes[1].imshow(np.ma.masked_where(~old_wrong, np.ones_like(target)), cmap=ListedColormap(['red']), alpha=.65)
    axes[1].set_title(f'S1-only errors: {old_wrong.sum():,} (red)')
    axes[2].imshow(np.ma.masked_where(~corrected, np.ones_like(target)), cmap=ListedColormap(['cyan']), alpha=.7)
    axes[2].set_title(f'S1 errors corrected by S1+S2: {corrected.sum():,} (cyan)')
    fig.suptitle('Spain_5678382 — best single-chip checkpoints; training diagnostic')
    fig.tight_layout(); fig.savefig(dest/'s1_vs_s1s2.png', dpi=180); plt.close(fig)
    return {'s1_wrong_pixels': int(old_wrong.sum()), 'corrected_pixels': int(corrected.sum()),
            'corrected_water_pixels': int((corrected & (target == 1)).sum()),
            'corrected_dry_pixels': int((corrected & (target == 0)).sum()),
            's1_wrong_s1s2_right_only': True, 'target_mask': 'label != -1',
            's1_reference_schedule': 'constant', 's1s2_schedule': 'cosine'}


def summarize(pack, dest, dev):
    rows = []
    for mode, names in [('s1', ('single', 'cosine400')),
                        ('s1s2', ('t15d-s1s2-single', 't15d-s1s2-eight')),
                        ('s2', ('t15d-s2-single', 't15d-s2-eight'))]:
        for scope, name in zip(('single', 'eight'), names):
            path = dest/name/'flood-run.json'; r = read(path)
            assert r['pack_manifest_sha256'] == pack.sha and r['device'] == 'mps'
            assert r['action'] == 'tiny' and r['train_indices'] == ([WORST] if scope == 'single' else ROWS)
            assert len(r['history']) == 400
            if mode != 's1':
                assert r['stop_reason'] == 'epoch_budget'
                assert r['config'] == config_for(read(dest/'cosine400/flood-run.json'), mode, scope)
            rows.append({'mode': mode, 'scope': scope, 'best_iou': r['metrics']['primary']['train']['water_iou'],
                         'end_loss': r['history'][-1]['train_loss'], 'best_epoch': r['best_epoch'],
                         'runtime_s': r['runtime_s'], 'schedule': r['config']['schedule'],
                         'json_sha256': sha256(path), 'checkpoint_sha256': r['checkpoint_sha256']})
    optical = [r['best_iou'] for r in rows if r['mode'] == 's1s2']
    out = {'code': code_info(), 'pack_manifest_sha256': pack.sha, 'rows': rows,
           'verdict': verdict(*optical), 'overlay': overlay(pack, dest, dev)}
    (dest/'t15d-summary.json').write_text(json.dumps(out, indent=2)+'\n')
    text = '# T15d optical-input diagnostic\n\n| Inputs | Scope | Best IoU | End loss | Best epoch | Schedule |\n|---|---|---:|---:|---:|---|\n'
    for r in rows:
        text += f"| {r['mode']} | {r['scope']} | {r['best_iou']:.6f} | {r['end_loss']:.6f} | {r['best_epoch']} | {r['schedule']} |\n"
    text += '\n'+json.dumps(out['verdict'], indent=2)+'\n\nS1 references reused; S1 worst-chip used constant, all new fits use cosine. No gate change. No full training. Training IoU only; no generalization claim.\n'
    (dest/'T15d-report.md').write_text(text)
    print(json.dumps(out, indent=2), flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['s1s2-single', 's1s2-eight', 's2-single', 's2-eight', 'report'])
    p.add_argument('--pack', default=str(Path.home()/'satquery-data/flood/packs/sen1floods11-hand-v1'))
    p.add_argument('--out', type=Path, default=Path('outputs/t15-debug'))
    args = p.parse_args()
    dev = torch.device('mps')
    if not torch.backends.mps.is_available():
        raise ValueError('STOP: requested MPS device unavailable')
    pack = Pack(args.pack)
    if pack.rows('train').iloc[WORST].chip_id != 'Spain_5678382':
        raise ValueError('STOP: worst chip identity changed')
    if args.action == 'report':
        return summarize(pack, args.out, dev)
    mode, scope = args.action.split('-')
    cfg = config_for(read(args.out/'cosine400/flood-run.json'), mode, scope)
    return train.fit(pack, cfg, args.out/('t15d-'+args.action), 'tiny', dev)


if __name__ == '__main__':
    main()
