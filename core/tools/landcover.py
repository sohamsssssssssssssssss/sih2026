"""BEN-19 checkpoints, projected to five uncalibrated coarse presence scores."""
import json
from pathlib import Path
import time

from core.gate import CLOUD_PCT
from core.registry import ToolNeeds
from core.schema import Evidence, ToolResult
from core.tools.base import BaseTool

BANDS = {'s1': ('VV', 'VH'), 's2': ('B02', 'B03', 'B04', 'B08')}
UNITS = {'s1': 'dB', 's2': 'reflectance_x1e4'}
CLASSES = ('trees', 'shrub_grass', 'cropland', 'built', 'water')
CONFIG = Path(__file__).resolve().parents[2] / 'configs/landcover.json'


def select_mode(scene):
    if scene.cloud_pct is not None and scene.cloud_pct > CLOUD_PCT:
        if scene.sensor not in ('S1', 'S1S2'):
            raise ValueError('Cloudy optical input requires S1')
        return 's1'
    modes = {'S1': 's1', 'S2': 's2', 'S1S2': 's1s2'}
    if scene.sensor not in modes:
        raise ValueError('Land cover requires S1 and/or S2')
    return modes[scene.sensor]


class LandcoverTool(BaseTool):
    name = 'landcover'
    version = 'ben19-coarse5-v1'
    needs = ToolNeeds(sensors=('S1', 'S2'), bands=(*BANDS['s1'], *BANDS['s2']))

    def __init__(self, config_path=CONFIG):
        super().__init__()
        self.config_path = Path(config_path)
        self._loaded = {}

    def _load(self, mode):
        if mode not in self._loaded:
            import torch
            from ben import landcover as lc
            from ben.transfer import coarse_members
            config = json.loads(self.config_path.read_text())
            path = Path(config['checkpoints'][mode]).expanduser()
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            inputs = ('s1', 's2') if mode == 's1s2' else (mode,)
            channels = sum(len(BANDS[key]) for key in inputs)
            if (checkpoint['mode'] != mode or tuple(checkpoint['inputs']) != inputs
                    or checkpoint['in_channels'] != channels or len(checkpoint['classes']) != 19):
                raise ValueError('Checkpoint identity/channel contract mismatch')
            members, _ = coarse_members(checkpoint['classes'])
            if any(name not in members for name in CLASSES):
                raise ValueError('Checkpoint lacks a coarse class')
            stats = checkpoint['preprocess_stats']
            import numpy as np
            for key in inputs:
                for field in ('mean', 'std', 'clip_lo', 'clip_hi') if key == 's1' else ('mean', 'std'):
                    values = np.asarray(stats[key][field])
                    if values.shape != (len(BANDS[key]),) or not np.isfinite(values).all():
                        raise ValueError('Invalid checkpoint normalization')
                if np.any(np.asarray(stats[key]['std']) <= 0):
                    raise ValueError('Invalid checkpoint standard deviation')
                if key == 's2' and stats[key]['scale'] != 1 / 10000:
                    raise ValueError('Checkpoint optical scale mismatch')
                if key == 's1' and np.any(np.asarray(stats[key]['clip_lo']) > stats[key]['clip_hi']):
                    raise ValueError('Checkpoint clipping bounds mismatch')
            model = lc.build_model(channels, 19)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            model.eval()
            self._loaded[mode] = (model, stats, {c: members[c] for c in CLASSES}, lc.sha256(path))
        return self._loaded[mode]

    def run(self, scene, query, pixels):
        started = time.perf_counter()
        try:
            import numpy as np
            mode = select_mode(scene)
            inputs = ('s1', 's2') if mode == 's1s2' else (mode,)
            # Joint data need per-sensor units: Scene.units cannot represent mixed dB/DN.
            payload = pixels if isinstance(pixels, dict) else {
                mode: {'pixels': pixels, 'bands': scene.bands, 'units': scene.units}}
            arrays = {}
            for key in inputs:
                if not all(b in scene.bands for b in BANDS[key]):
                    raise ValueError(f'Missing {key} scene bands: {BANDS[key]}')
                item = payload[key]
                if tuple(item['bands']) != BANDS[key] or item['units'] != UNITS[key]:
                    raise ValueError(f'{key} requires band order {BANDS[key]} and units {UNITS[key]}')
                if scene.sensor != 'S1S2' and scene.units != UNITS[key]:
                    raise ValueError('Scene and payload units disagree')
                x = np.asarray(item['pixels'])
                if x.shape != (len(BANDS[key]), 120, 120) or not np.isfinite(x).all():
                    raise ValueError(f'{key} requires finite CHW 120x120 pixels')
                arrays[key] = x[None].astype(np.float32)
            import torch
            from ben.landcover import make_batch
            model, stats, members, digest = self._load(mode)
            with torch.inference_mode():
                logits = model(make_batch(arrays, stats, np.array([0]), torch.device('cpu')))
                if logits.shape != (1, 19) or not torch.isfinite(logits).all():
                    raise ValueError('Model returned invalid BEN-19 logits')
                scores = {name: float(torch.sigmoid(logits[0, indices].max())) for name, indices in members.items()}
            return ToolResult(tool=self.name, tool_version=self.version, checkpoint_sha=digest,
                              value={'answer': 'Uncalibrated coarse land-cover presence scores.', 'mode': mode, 'scores': scores},
                              raw_confidence=None, latency_ms=int((time.perf_counter() - started) * 1000),
                              evidence=(Evidence(kind='text', summary={'mode': mode, 'classes': list(CLASSES),
                                  'score_kind': 'sigmoid(max member logit), not area fraction or calibrated confidence',
                                  'tool_status': 'experimental_unvalidated'}),))
        except Exception as error:
            return self.error_result(error, started)
