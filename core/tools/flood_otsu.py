"""Label-free, event/scene-window VH Otsu in dB; no per-chip fitting."""
import time
import numpy as np

from core.registry import ToolNeeds
from core.schema import Evidence
from core.tools.base import BaseTool
from models.change.sar_water import otsu_threshold

CLAMP_DB = (-26.0, -14.0)  # fixed by T07c instruction, train-only audit; no test tuning


def water_mask(vh):
    """One threshold over the entire provided window, including a pooled chip stack."""
    vh = np.asarray(vh)
    finite = np.isfinite(vh)
    raw = otsu_threshold(vh[finite]) if finite.any() else None
    threshold = float(np.clip(raw, *CLAMP_DB)) if raw is not None else None
    mask = (vh < threshold) & finite if threshold is not None else np.zeros(vh.shape, dtype=bool)
    return mask, raw, threshold


class FloodOtsuTool(BaseTool):
    name = 'flood-otsu'
    version = 'vh-event-otsu-clamp-v1'
    needs = ToolNeeds(sensors=('S1',), bands=('VH',))

    def run(self, scene, query, pixels):
        started = time.perf_counter()
        try:
            if scene.sensor not in ('S1', 'S1S2'):
                raise ValueError('Flood Otsu requires S1 VH in dB, never linear backscatter')
            if 'VH' not in scene.bands or len(set(scene.bands)) != len(scene.bands):
                raise ValueError('Missing/duplicate VH band metadata')
            item = pixels['s1'] if isinstance(pixels, dict) else {'pixels': pixels, 'bands': scene.bands, 'units': scene.units}
            bands = tuple(item['bands'])
            if item['units'] != 'dB' or (scene.sensor == 'S1' and scene.units != 'dB'):
                raise ValueError('VH must be in dB')
            if 'VH' not in bands or len(set(bands)) != len(bands) or any(b not in scene.bands for b in bands):
                raise ValueError('Payload band metadata mismatch')
            x = np.asarray(item['pixels'])
            if x.ndim not in (3, 4) or x.shape[-3] != len(bands) or min(x.shape) == 0:
                raise ValueError('Expected CHW scene or NCHW event stack matching Scene.bands')
            vh = x[..., bands.index('VH'), :, :]
            mask, raw, threshold = water_mask(vh)
            if 'chip_index' in item:
                index = item['chip_index']
                if x.ndim != 4 or type(index) is not int or not 0 <= index < len(mask):
                    raise ValueError('Invalid chip_index for event stack')
                mask = mask[index]
            water = int(mask.sum())
            return self._result(started, {'answer': 'Experimental radar open-water mask; water alone does not establish flooding.',
                'water_mask': mask.astype(np.uint8).tolist(), 'water_pixels': water,
                'water_fraction': water / mask.size, 'threshold_db': threshold},
                evidence=(Evidence(kind='mask', summary={'method': 'event-window VH Otsu',
                    'raw_threshold_db': raw, 'threshold_db': threshold, 'clamp_db': list(CLAMP_DB),
                    'window_pixels': int(vh.size), 'finite_vh_pixels': int(np.isfinite(vh).sum()),
                    'nonfinite_prediction': 'dry', 'scope': 'entire supplied scene or event stack',
                    'tool_status': 'experimental_unvalidated'}),))
        except Exception as error:
            return self.error_result(error, started)
