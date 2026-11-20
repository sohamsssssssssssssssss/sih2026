"""G3 on corrected inputs (record only; promotion already failed on G2).

Runs the India final s2 weights through the integration /ask preprocessing
(imported read-only) on the corrected Ludhiana v2 backup and the Punjab
fixture. Thresholds are the attempt-1 G3 rule. Raw scores are the record,
because no calibrator passed. Attempt-1 Platt scores are listed for context.
Usage, from ~/sih2026-ben:
  PYTHONDONTWRITEBYTECODE=1 ~/sih2026-integ/.venv/bin/python -m ben.promotion.g3_corrected
"""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

INTEG = Path.home() / 'sih2026-integ'
WEIGHTS = Path.home() / 'ben-work/final-run-1/ben-landcover-india-s2.pt'
WEIGHTS_SHA = '26e0cdfb7dd40abc93d025b4bd3db2ad6fcf0678e6b7385d218fee3f0675bdf9'
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / 'results/landcover-india-promotion/g3-corrected.json'
# Attempt-1 rule: target class in the top two.
SCENES = (('Ludhiana', 'data/fresh-backups/ludhiana-s2-v2.tif', 'built', 'cached'),
          ('Punjab', 'tests/fixtures/punjab-s2.tif', 'cropland', None))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def top2_pass(scores, target):
    return target in sorted(scores, key=scores.get, reverse=True)[:2]


def main():
    assert sha(WEIGHTS) == WEIGHTS_SHA, 'India s2 weights differ from the promotion candidate'
    sys.path.insert(0, str(INTEG))
    from backend.ask_scenes import SceneStore
    from backend.config import AskConfig
    from core.tools.landcover import CLASSES, LandcoverTool
    platt = json.loads((HERE.parents[1] / 'results/landcover-india-promotion/calibrator.json').read_text())['modes']['s2']
    import subprocess
    integ_head = subprocess.run(['git', '-C', str(INTEG), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    record = {'label': 'G3 on corrected inputs (record only; promotion already failed on G2)',
              'weights': str(WEIGHTS), 'weights_sha256': WEIGHTS_SHA, 'integ_head': integ_head,
              'rule': 'target class in the top two scores (attempt-1 G3 thresholds)', 'scenes': {}}
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / 'cfg.json'
        cfg.write_text(json.dumps({'checkpoints': {'s2': str(WEIGHTS)}}))
        tool = LandcoverTool(cfg)
        for name, rel, target, mode in SCENES:
            path = INTEG / rel
            scene = SceneStore(AskConfig()).raster(path.read_bytes(), fresh_mode=mode)
            result = tool.run(scene.scene, None, scene.pixels('landcover', 'S2', Path(tmp)))
            raw = result.value['scores']
            cal = {c: float(expit(logit(np.clip(raw[c], 1e-6, 1 - 1e-6)) * platt[c]['slope'] + platt[c]['intercept'])) for c in CLASSES}
            record['scenes'][name] = {
                'file': rel, 'sha256': sha(path), 'target': target,
                'raw_top3': [[c, raw[c]] for c in sorted(raw, key=raw.get, reverse=True)[:3]],
                'raw_scores': raw, 'raw_pass': top2_pass(raw, target),
                'attempt1_platt_top3_context': [[c, cal[c]] for c in sorted(cal, key=cal.get, reverse=True)[:3]],
                'attempt1_platt_pass_context': top2_pass(cal, target)}
    OUT.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
