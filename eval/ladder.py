"""Resolution degradation of real imagery; only resolution is simulated.

The source pixels come from LoveDA or DOTA. ``sensor`` therefore records the
real source dataset and must never be set to ``synthetic``.
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import TileSample, validate_tile_sample  # noqa: E402

SOURCE_GSD = 0.3
RUNGS = (0.3, 1.0, 2.0, 5.0, 10.0)
CLASS_NAMES = {
    1: "background",
    2: "building",
    3: "road",
    4: "water",
    5: "barren",
    6: "forest",
    7: "agriculture",
}
QUESTIONS = (
    "What is the dominant land cover in this image?",
    "Is there a building in this image?",
)


def blur_sigma(target_gsd: float, source_gsd: float = SOURCE_GSD) -> float:
    return 0.5 * (target_gsd / source_gsd)


def degrade(
    rgb: np.ndarray,
    target_gsd: float,
    rng: np.random.Generator,
    source_gsd: float = SOURCE_GSD,
) -> tuple[np.ndarray, float]:
    """PSF blur, area-average decimation, then read and shot noise."""
    if target_gsd == source_gsd:
        # The native rung is the zero-degradation anchor: no blur, resize, or noise.
        return rgb.copy(), 0.0
    sigma = blur_sigma(target_gsd, source_gsd)
    blurred = gaussian_filter(rgb.astype(np.float32), sigma=(sigma, sigma, 0), mode="reflect")
    height, width = rgb.shape[:2]
    output_size = (
        max(1, round(width * source_gsd / target_gsd)),
        max(1, round(height * source_gsd / target_gsd)),
    )
    decimated = np.stack(
        [
            np.asarray(
                Image.fromarray(blurred[:, :, channel]).resize(
                    output_size, Image.Resampling.BOX
                ),
                dtype=np.float32,
            )
            for channel in range(blurred.shape[2])
        ],
        axis=-1,
    )
    photons = 50.0
    shot = rng.poisson(np.clip(decimated, 0, 255) / 255.0 * photons) / photons * 255.0
    read = rng.normal(0.0, rng.uniform(1.0, 2.0), size=shot.shape)
    return np.clip(shot + read, 0, 255).astype(np.uint8), sigma


def find_loveda_pairs(raw_root: Path) -> list[tuple[Path, Path]]:
    images = sorted(path for path in raw_root.rglob("*.png") if path.parent.name == "images_png")
    pairs = []
    for image_path in images:
        candidates = [
            image_path.parent.parent / "masks_png" / image_path.name,
            Path(str(image_path).replace("images_png", "masks_png")),
        ]
        label_path = next((path for path in candidates if path.exists()), None)
        if label_path is not None:
            pairs.append((image_path, label_path))
    return pairs


def label_answers(label_path: Path) -> dict[str, str]:
    labels = np.asarray(Image.open(label_path))
    counts = Counter(int(value) for value in labels.ravel() if int(value) in CLASS_NAMES)
    dominant = CLASS_NAMES[counts.most_common(1)[0][0]] if counts else "background"
    return {QUESTIONS[0]: dominant, QUESTIONS[1]: "yes" if np.any(labels == 2) else "no"}


def generate_ladder(
    raw_root: Path,
    output_root: Path,
    limit: int = 200,
    seed: int = 26167,
) -> list[dict]:
    pairs = find_loveda_pairs(raw_root)[:limit]
    if not pairs:
        raise FileNotFoundError(
            f"No paired LoveDA images/masks found under {raw_root}; run data/download_ladder_data.py"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    sanity_rows = []
    for index, (image_path, label_path) in enumerate(pairs):
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        relative_stem = image_path.relative_to(raw_root).with_suffix("")
        orig_id = "loveda_" + "_".join(relative_stem.parts)
        answers = label_answers(label_path)
        for target_gsd in RUNGS:
            degraded, sigma = degrade(rgb, target_gsd, rng)
            rung_name = f"{target_gsd:.1f}"
            rung_dir = output_root / rung_name
            rung_dir.mkdir(parents=True, exist_ok=True)
            output_path = rung_dir / f"{orig_id}_gsd{rung_name}.png"
            Image.fromarray(degraded).save(output_path)
            optical = degraded.astype(np.float32).transpose(2, 0, 1)
            sample: TileSample = {
                "id": f"{orig_id}_gsd{target_gsd}",
                "optical": optical,
                "sar": None,
                "optical_t2": None,
                "gsd": float(target_gsd),
                "sensor": "loveda",
                "meta": {
                    "source_id": orig_id,
                    "degradation": "psf_blur_decimate_noise",
                    "native_gsd": SOURCE_GSD,
                },
            }
            validate_tile_sample(sample)
            record = {
                "id": sample["id"],
                "image_path": str(output_path.relative_to(ROOT)),
                "optical": str(output_path.relative_to(ROOT)),
                "sar": None,
                "optical_t2": None,
                "gsd": sample["gsd"],
                "sensor": sample["sensor"],
                "meta": sample["meta"],
                "answers": answers,
            }
            records.append(record)
            if index == 0:
                sanity_rows.append(
                    (target_gsd, sigma, target_gsd != SOURCE_GSD, rgb.shape[:2], degraded.shape[:2])
                )
    manifest = output_root / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    print(f"Generated {len(records)} tiles ({len(pairs)} real sources x {len(RUNGS)} rungs)")
    print(f"Example source: {pairs[0][0]} (pixels HxW={sanity_rows[0][3]})")
    for gsd, sigma, processing_applied, source_shape, output_shape in sanity_rows:
        print(
            f"  gsd={gsd:.1f}m sigma_used={sigma:.6f} "
            f"processing_applied={str(processing_applied).lower()} "
            f"source={source_shape[1]}x{source_shape[0]} output={output_shape[1]}x{output_shape[0]}"
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data" / "raw" / "loveda")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "ladder")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=26167)
    args = parser.parse_args()
    if not 1 <= args.limit <= 200:
        parser.error("--limit must be between 1 and 200")
    try:
        generate_ladder(args.raw_root, args.out, args.limit, args.seed)
    except Exception as exc:
        print(f"LADDER GENERATION FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
