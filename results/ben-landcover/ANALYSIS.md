# BigEarthNet land-cover baseline: analysis (16 Nov 2026)

Run `python3.11 analyze_ben_landcover.py` to reproduce every number below from the saved test logits and labels. It also cross-checks them against the run JSONs: 78 checks at tolerance 1e-3, 0 failures. `analysis.json` holds the full output. Provenance is in [SOURCE.md](SOURCE.md).

## Setup

- **Data:** 6,000 test patches, all 19 classes with at least one test positive.
- **Model:** ResNet-18 from scratch, trained on 12,000 patches with the epoch picked on 2,000 validation patches.
- **Runs:** one seed per mode: `s1` (VV, VH), `s2` (B02, B03, B04, B08) and `s1s2` (all 6).
- **Uncertainty:** 95% CIs come from 1,000 bootstrap resamples of test patches, seed 17. Comparisons are paired, using the same resamples for both runs.

## Verdict

| | Test macro mAP | 95% CI | Shuffled-image control |
|---|---|---|---|
| s1 (radar only) | 0.5598 | [0.5512, 0.5785] | 0.1594 |
| s2 (optical only) | 0.6254 | [0.6121, 0.6483] | 0.1585 |
| s1s2 (both) | 0.6283 | [0.6193, 0.6492] | 0.1586 |
| Label prior (train prevalence) | 0.1568 | — | — |

1. **Radar alone predicts land cover well above the controls.** s1 scores 0.5598, against 0.1568 for the label prior and 0.1594 with shuffled images. The shuffle control falls to the prior level for all three modes (0.1585–0.1594), as it should.
2. **Optical beats radar.** The paired difference s2 − s1 is **+0.0656**, with a 95% CI of [+0.0504, +0.0836].
3. **Adding radar to optical gives no measurable gain** on this cloud-free sample. The paired difference s1s2 − s2 is **+0.0029**, with a 95% CI of [−0.0083, +0.0134], which spans 0.
4. **Training is incomplete for radar only.** Over the last 5 epochs, a linear fit to validation mAP gives:
   - **s1:** best epoch 30/30, validation mAP 0.6050 → 0.6156 over epochs 26–30, slope +0.0025 per epoch. It is still rising, so it is under-trained.
   - **s2:** best epoch 28/30, last 5 epochs 0.6737–0.6789, slope +0.0007 per epoch. It is nearly flat.
   - **s1s2:** best epoch 27/30, last 5 epochs 0.6795–0.6841, slope −0.0003 per epoch. It has plateaued.

   The cosine schedule takes the learning rate to 0 at epoch 30, so the end-of-run trend is partly a schedule effect. A longer run would need a new schedule, not just more epochs. Radar's gap to optical may shrink with longer training; that has not been measured.

## Per-class AP (test), sorted by the s2 − s1 gap

| Class | Test positives | s1 | s2 | s1s2 | s2 − s1 |
|---|---|---|---|---|---|
| Permanent crops | 280 | 0.237 | 0.456 | 0.469 | +0.219 |
| Beaches, dunes, sands ⚠ | 12 | 0.009 | 0.192 | 0.079 | +0.183 |
| Coniferous forest | 1929 | 0.831 | 0.918 | 0.915 | +0.087 |
| Broad-leaved forest | 1853 | 0.740 | 0.826 | 0.832 | +0.086 |
| Agro-forestry areas | 524 | 0.714 | 0.797 | 0.798 | +0.084 |
| Land principally occupied by agriculture, with significant areas of natural vegetation | 1464 | 0.587 | 0.670 | 0.675 | +0.083 |
| Complex cultivation patterns | 1092 | 0.543 | 0.615 | 0.625 | +0.073 |
| Mixed forest | 2234 | 0.784 | 0.851 | 0.859 | +0.067 |
| Inland wetlands | 246 | 0.427 | 0.493 | 0.530 | +0.066 |
| Transitional woodland, shrub | 2025 | 0.649 | 0.710 | 0.722 | +0.061 |
| Moors, heathland and sclerophyllous vegetation | 186 | 0.339 | 0.398 | 0.411 | +0.059 |
| Pastures | 1346 | 0.772 | 0.815 | 0.828 | +0.043 |
| Natural grassland and sparsely vegetated areas | 90 | 0.159 | 0.201 | 0.256 | +0.042 |
| Arable land | 2443 | 0.862 | 0.903 | 0.902 | +0.041 |
| Coastal wetlands ⚠ | 8 | 0.108 | 0.131 | 0.093 | +0.023 |
| Inland waters | 871 | 0.862 | 0.880 | 0.886 | +0.018 |
| Marine waters | 591 | 0.975 | 0.989 | 0.985 | +0.014 |
| Urban fabric | 602 | 0.750 | 0.754 | 0.771 | +0.004 |
| Industrial or commercial units | 83 | 0.292 | 0.285 | 0.304 | −0.007 |

⚠ = fewer than 30 test positives, so the AP is unreliable.

**Observations:**
- Radar scores at least as high as optical (s1 ≥ s2) for only one class: Industrial or commercial units, by −0.007.
- Radar is closest to optical on water and built-up classes (Urban fabric +0.004, Marine waters +0.014, Inland waters +0.018).
- Radar is furthest behind on Permanent crops (+0.219) and on the forest and agro-forestry classes (+0.08 to +0.09).
- These are single-run point estimates. No per-class CIs were computed.

## Claim status

**Allowed:** "On 6,000 BigEarthNet v2 test patches (Europe, 10 m, official split), a ResNet-18 trained from scratch on 12,000 patches reaches land-cover macro mAP 0.560 from Sentinel-1 radar alone, 0.625 from Sentinel-2 optical (4 bands), 0.628 from both, vs 0.157 label-prior and ~0.159 with shuffled images. One seed."

**Not allowed:**
- Any claim about performance in India, or with RISAT or Cartosat.
- "Radar sees through clouds" as a measured result. The sample is cloud-free, so this was not tested.
- Comparisons to published BigEarthNet numbers, which use a different data size and setup.
- Calling these models final or deployed.

Micro-F1 and macro-F1 are recorded in the run JSONs, but they are not repeated here. The validation logits needed to recompute their thresholds were not saved, so the analysis code cannot reproduce them.
