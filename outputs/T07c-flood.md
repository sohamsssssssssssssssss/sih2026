# T07c flood Otsu — FAIL

India-test pass rule: point water IoU >= 0.65. No post-verdict tuning.

| Split | Arm | Water IoU | Chip-bootstrap CI95 | Chips |
|---|---|---:|---|---:|
| train | B1 | 0.451307 | [0.371530, 0.529336] | 252 |
| train | B2 | 0.178324 | [0.143851, 0.217162] | 252 |
| train | T07c | 0.339202 | [0.278408, 0.401539] | 252 |
| valid | B1 | 0.505620 | [0.404472, 0.599006] | 89 |
| valid | B2 | 0.207229 | [0.149587, 0.268402] | 89 |
| valid | T07c | 0.367210 | [0.281017, 0.449333] | 89 |
| test | B1 | 0.545846 | [0.417538, 0.653809] | 90 |
| test | B2 | 0.225132 | [0.153257, 0.295140] | 90 |
| test | T07c | 0.423390 | [0.305811, 0.521519] | 90 |
| bolivia | B1 | 0.697201 | [0.475943, 0.815673] | 15 |
| bolivia | B2 | 0.379014 | [0.164105, 0.585871] | 15 |
| bolivia | T07c | 0.636680 | [0.415325, 0.768786] | 15 |
| India-test | B1 | 0.704408 | [0.549551, 0.797494] | 14 |
| India-test | B2 | 0.379412 | [0.178096, 0.560738] | 14 |
| India-test | T07c | 0.631332 | [0.440758, 0.723997] | 14 |

All scoring uses label != -1. B1 is the shipped mask, B2 is the fixed historical
per-chip baseline. Non-finite VH is dry for T07c, including label-valid pixels.
Otsu pools all chips of the same event across splits, with imagery only; labels
are opened after thresholds/predictions are complete. This is transductive event
context, not independent-chip prediction. CI95 conditions on the event thresholds;
India-test has one event, so chip bootstrap does not estimate new-event uncertainty.

The clamp [-26,-14] was supplied by the user before this run, not learned from
test. Train-only raw/clamped thresholds (dB), audited first without any labels:

| Train event | Chips | Raw | Clamped |
|---|---:|---:|---:|
| Ghana | 31 | -16.5307 | -16.5307 |
| India | 40 | -16.9239 | -16.9239 |
| Mekong | 18 | -22.7528 | -22.7528 |
| Nigeria | 10 | -22.3818 | -22.3818 |
| Pakistan | 16 | -19.6057 | -19.6057 |
| Paraguay | 39 | -20.5934 | -20.5934 |
| Somalia | 15 | -20.0716 | -20.0716 |
| Spain | 18 | -24.6826 | -24.6826 |
| Sri-Lanka | 24 | -18.6261 | -18.6261 |
| USA | 41 | -19.9102 | -19.9102 |

CPU latency for one real patch (India_900498) including threshold computation
over 68 India-event chips and patch-mask assembly: **232.2 ms**.
The adapter mask exactly matches the evaluated mask for this patch.

Tool remains experimental_unvalidated whatever the verdict; no calibrated
confidence. This detects open water, not flood causality or water under vegetation.
Preregistration: docs/hypotheses/T07c.md at f5bd22d. JSON includes hashes of consumed
arrays, fixed method parameters, pool membership, thresholds and per-chip counts.
