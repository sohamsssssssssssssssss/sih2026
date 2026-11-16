# BigEarthNet land-cover baseline: radar vs optical vs both

**Status: measured once (16 Nov 2026), one seed.** See the Measured section below and `results/ben-landcover/`.

## What it measures

This baseline asks one question: how well does Sentinel-1 C-band radar alone (VV, VH; RISAT-like band) predict multi-label land cover, compared with Sentinel-2 optical (B02, B03, B04, B08 at 10 m) and with both together? The data is the BigEarthNet v2 (reBEN) pilot pack `ben-pilot-20k`, built in `~/ben-work/pack`. It holds 20,000 patches of 120×120 px, all from Europe at 10 m. The split follows the official metadata: train 12,000, validation 2,000, test 6,000. There are 19 CLC classes.

`ben/landcover.py` trains one from-scratch ResNet-18 (`weights=None`) per mode:

| Mode | Channels | Preprocessing (stats from TRAIN only, saved in the output) |
|---|---|---|
| `s1` | VV, VH (dB) | clip to train 0.5/99.5 percentile per channel, then train mean/std |
| `s2` | B02, B03, B04, B08 | ÷10000, then train mean/std |
| `s1s2` | the 6 channels concatenated | as above, per source |

Training uses:
- BCEWithLogits loss.
- AdamW optimiser, lr 1e-3, weight decay 1e-4, cosine schedule.
- 30 epochs, batch 128.
- AMP on CUDA.
- Augmentation: random flips and 90° rotations only.
- Seed 17.

**Model selection** uses VALIDATION macro mAP. TEST is evaluated exactly once, with the best checkpoint.

The **reported metrics** are computed from the saved test logits:
- Macro mAP over classes with at least one test positive. The number of skipped classes is reported.
- Micro-F1 and macro-F1 at per-class thresholds chosen on VALIDATION (maximum F1).
- A per-class AP table.
- A label-prior baseline that scores every patch with the train prevalence of each class.

**Controls and uncertainty:**
- **Shuffle control:** the same trained model is scored with test images permuted against their labels (seed 17). Its macro mAP should fall to about the prior baseline. Both numbers are reported.
- **Bootstrap:** 1,000 resamples of TEST patches (not pixels), seed 17, giving a 95% CI for macro mAP.
- **Paired comparison:** `--compare a.json b.json` reuses the same resamples on both runs' saved logits and reports the difference in macro mAP with a 95% CI.

Each run first checks the sha256 of the pack's `.npy`/`.json` files and `selection.parquet` against `MANIFEST.json`, and stops on any mismatch. It records the MANIFEST sha256, the code git sha (when available) and the sha256 of `landcover.py`.

## How to run

Local synthetic tests (CPU, no real data):

```
python3.11 -m pytest ben -q
```

To run on Kaggle (GPU T4):
1. Upload `satquery-ben-code.zip` (built by `git archive -o satquery-ben-code.zip HEAD ben`) as a private dataset.
2. Attach it and the `ben-pilot-20k` pack dataset to a notebook.
3. Paste `ben/run_landcover_gpu.txt` into a single cell and run the notebook as a committed version.

The cell finds the code and the pack by searching `/kaggle/input`, and requires exactly one match for each. It then runs `s1`, `s2` and `s1s2` with `--control shuffle`, followed by the comparisons `s1s2` vs `s2` and `s2` vs `s1`. It prints a summary table and ends with `CELL DONE`. Per-epoch progress also goes to `/kaggle/working/ben-landcover-progress.txt`.

A single mode can be run by hand:

```
python ben/landcover.py --pack PACK_DIR --mode s1 --control shuffle --out-dir OUT --progress OUT/progress.txt
python ben/landcover.py --compare OUT/ben-landcover-s1s2.json OUT/ben-landcover-s2.json --output OUT/diff.json
```

## What it does NOT claim

- **Scope:** results cover Europe only, at 10 m, with Sentinel-1/Sentinel-2 and BigEarthNet's CLC labels. They do not transfer to Cartosat, RISAT, other regions or other resolutions without new measurement.
- **Task:** this is patch-level multi-label land cover only. It says nothing about counting, change detection or VQA.
- **Purpose:** it is a single-seed baseline on a 20k pilot subset, not a deployed tool and not a tuned model. A gap between modes shows what this recipe achieves, not what each sensor can achieve at best.
- **Labels:** CLC labels are inherited from reBEN and their accuracy is not verified here.

## Measured (16 Nov 2026)

This is one run on Kaggle T4 of code `af4b4c4` on pack MANIFEST `dc9c43ad…`. The full analysis is in [`results/ben-landcover/ANALYSIS.md`](../results/ben-landcover/ANALYSIS.md) and provenance is in [`SOURCE.md`](../results/ben-landcover/SOURCE.md).

On 6,000 BigEarthNet v2 test patches (Europe, 10 m, official split), a ResNet-18 trained from scratch on 12,000 patches reaches land-cover macro mAP 0.560 from Sentinel-1 radar alone, 0.625 from Sentinel-2 optical (4 bands) and 0.628 from both. The label prior scores 0.157 and shuffled images about 0.159. One seed.

- **Optical vs radar:** optical beats radar by +0.066 (paired 95% CI +0.050 to +0.084).
- **Adding radar to optical:** no measurable gain, +0.003 (95% CI −0.008 to +0.013).
- **Training length:** radar was still improving at epoch 30. Optical and the combined model had levelled off.
