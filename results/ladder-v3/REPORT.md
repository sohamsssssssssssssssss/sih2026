# LoveDA resolution ladder v3

Kaggle recorded Tesla T4 and commit `0c7b8787712585868265d6fb25de9009ba449560`. The gate and question-manifest SHA-256s matched the preregistration. Each real and grey condition covered the same 200 frozen tiles and all five rungs. All 21,880 predictions matched the registered tile, question type and gold tuples. The copy came from the extracted `ladder-v3` folder, so the source ZIP name is unavailable. [Preregistration](../../docs/gates/LADDER-V3.md) · [Raw file manifest](MANIFEST.json) · [Kaggle report](raw/REPORT.md) · [Raw predictions and full-precision scores](raw/results.json)

## Real-image accuracy by rung

Intervals use the registered 2,000 paired tile bootstrap draws with seed 17. Domain accuracy is the fraction of correct predictions on an all-Rural cohort. Urban recall and domain balanced accuracy are undefined.

| GSD (m) | Task | Base accuracy [95% CI] | B13 accuracy [95% CI] |
|---:|---|---:|---:|
| 0.3 | Pooled presence | 0.6715 [0.6325, 0.7077] | 0.5461 [0.5101, 0.5843] |
| 0.3 | Dominant class | 0.7050 [0.6400, 0.7700] | 0.5950 [0.5250, 0.6650] |
| 0.3 | Domain (Rural only) | 0.9950 [0.9850, 1.0000] | 0.9700 [0.9450, 0.9900] |
| 1 | Pooled presence | 0.5893 [0.5494, 0.6306] | 0.5576 [0.5206, 0.5952] |
| 1 | Dominant class | 0.6000 [0.5300, 0.6650] | 0.3500 [0.2850, 0.4150] |
| 1 | Domain (Rural only) | 0.9750 [0.9500, 0.9950] | 0.9500 [0.9200, 0.9800] |
| 2 | Pooled presence | 0.5461 [0.5022, 0.5858] | 0.5605 [0.5231, 0.6017] |
| 2 | Dominant class | 0.4000 [0.3350, 0.4700] | 0.1550 [0.1050, 0.2100] |
| 2 | Domain (Rural only) | 0.9950 [0.9850, 1.0000] | 0.9900 [0.9750, 1.0000] |
| 5 | Pooled presence | 0.5000 [0.4555, 0.5422] | 0.4914 [0.4526, 0.5272] |
| 5 | Dominant class | 0.2100 [0.1550, 0.2650] | 0.2000 [0.1450, 0.2600] |
| 5 | Domain (Rural only) | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] |
| 10 | Pooled presence | 0.5029 [0.4599, 0.5426] | 0.5043 [0.4605, 0.5440] |
| 10 | Dominant class | 0.1950 [0.1450, 0.2500] | 0.1900 [0.1350, 0.2450] |
| 10 | Domain (Rural only) | 0.9900 [0.9750, 1.0000] | 0.9750 [0.9500, 0.9950] |

## Reading and claim limits

At 5 m and 10 m, pooled presence was degenerate for both models and neither showed a pooled real-image advantage over grey. B13 also failed the pooled presence rule at 0.3 m and 1 m because its yes-rate exceeded 0.85. At 2 m it cleared the degeneracy rule, but its matched pooled interval still included zero. The building class showed a positive B13 image effect at 1 m and 2 m. The app and judges may cite those measured effects on synthetically coarsened LoveDA tiles and describe the near-chance result at 10 m. The experiment provides no evidence for reliable coarse-rung recognition, counting, Indian geography, real ISRO sensors or urban recall. [Raw predictions and scores](raw/results.json)

## Registered decisions

The preregistration defines cell-level decisions. It sets no overall model or release pass bar. Presence **PASS** means the cell is non-degenerate. **FAIL** means yes-rate is strictly above 0.85, the balanced-accuracy CI contains 0.5, or both. A CI endpoint equal to 0.5 counts as containing 0.5. For real minus grey, **PASS** means the 95% CI excludes zero in either direction. **FAIL** means the interval includes zero. A negative PASS means real imagery reduced accuracy. The tests are descriptive with no multiple-comparison adjustment. There were no amendments.

| GSD (m) | Base real presence gate | B13 real presence gate | Base real − grey accuracy gate | B13 real − grey accuracy gate |
|---:|---|---|---|---|
| 0.3 | PASS · yes 0.4568 · BA CI [0.6341, 0.7081] | FAIL · yes 0.9510 · BA CI [0.5282, 0.5652] | PASS · +0.1715 [0.1205, 0.2196] | FAIL · +0.0461 [-0.0272, 0.1225] |
| 1 | PASS · yes 0.2478 · BA CI [0.5555, 0.6240] | FAIL · yes 0.8876 · BA CI [0.5311, 0.5836] | PASS · +0.0893 [0.0505, 0.1281] | FAIL · +0.0576 [-0.0154, 0.1304] |
| 2 | PASS · yes 0.1268 · BA CI [0.5219, 0.5689] | PASS · yes 0.7406 · BA CI [0.5222, 0.5972] | PASS · +0.0461 [0.0213, 0.0697] | FAIL · +0.0605 [-0.0029, 0.1281] |
| 5 | FAIL · yes 0.0375 · BA CI [0.4857, 0.5135] | FAIL · yes 0.4899 · BA CI [0.4532, 0.5271] | FAIL · +0.0000 [-0.0142, 0.0131] | FAIL · -0.0086 [-0.0545, 0.0399] |
| 10 | FAIL · yes 0.0058 · BA CI [0.4974, 0.5087] | FAIL · yes 0.2032 · BA CI [0.4735, 0.5323] | FAIL · +0.0029 [-0.0028, 0.0086] | FAIL · +0.0043 [-0.0263, 0.0334] |

Grey pooled presence fails at every rung for both models (yes-rate 0.0000, balanced-accuracy CI [0.5000, 0.5000]). Across the 40 pooled and per-class presence decisions per condition, base real passed 15 and failed 25, B13 real passed 9 and failed 31, and both grey controls failed all 40. The 50 base image-dependence decisions included 15 passes (14 positive, 1 negative) and 35 failures. B13 had 8 passes (4 positive, 4 negative) and 42 failures.

## All preregistered metric cells

Balanced accuracy uses the observed gold classes for the dominant-class task. Both labels are represented for every frozen presence class. Each replicate with a required label absent is excluded from the corresponding balanced-accuracy interval. Malformed labels count as wrong and remain in the yes-rate denominator. Dashes indicate undefined or inapplicable metrics. Status uses the full-precision values.

| GSD (m) | Condition | Task | N | Accuracy [95% CI] | Balanced accuracy [95% CI] | Yes-rate | Malformed | Presence gate |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 0.3 | base_real | Pooled presence | 694 | 0.6715 [0.6325, 0.7077] | 0.6715 [0.6341, 0.7081] | 0.4568 | 0 | PASS |
| 0.3 | base_real | Dominant class | 200 | 0.7050 [0.6400, 0.7700] | 0.5383 [0.4672, 0.6534] | — | 0 | — |
| 0.3 | base_real | Domain (Rural only) | 200 | 0.9950 [0.9850, 1.0000] | — | — | 0 | — |
| 0.3 | base_real | Background | 14 | 0.4286 [0.1667, 0.7000] | 0.4286 [0.2500, 0.5000] | 0.0714 | 0 | FAIL |
| 0.3 | base_real | Building | 120 | 0.6583 [0.5739, 0.7427] | 0.6583 [0.5970, 0.7246] | 0.1917 | 0 | PASS |
| 0.3 | base_real | Road | 154 | 0.6299 [0.5506, 0.7055] | 0.6299 [0.5547, 0.7005] | 0.3766 | 0 | PASS |
| 0.3 | base_real | Water | 44 | 0.8409 [0.7291, 0.9389] | 0.8409 [0.7300, 0.9375] | 0.3864 | 0 | PASS |
| 0.3 | base_real | Barren | 160 | 0.5813 [0.5032, 0.6605] | 0.5813 [0.5045, 0.6599] | 0.5312 | 0 | PASS |
| 0.3 | base_real | Forest | 82 | 0.6585 [0.5568, 0.7586] | 0.6585 [0.5657, 0.7529] | 0.6951 | 0 | PASS |
| 0.3 | base_real | Agriculture | 120 | 0.8333 [0.7623, 0.8974] | 0.8333 [0.7698, 0.8898] | 0.6333 | 0 | PASS |
| 0.3 | base_constant | Pooled presence | 694 | 0.5000 [0.4576, 0.5395] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 0.3 | base_constant | Dominant class | 200 | 0.2200 [0.1650, 0.2750] | 0.1667 [0.1667, 0.1667] | — | 0 | — |
| 0.3 | base_constant | Domain (Rural only) | 200 | 1.0000 [1.0000, 1.0000] | — | — | 0 | — |
| 0.3 | base_constant | Background | 14 | 0.5000 [0.2222, 0.7692] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 0.3 | base_constant | Building | 120 | 0.5000 [0.4132, 0.5878] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 0.3 | base_constant | Road | 154 | 0.5000 [0.4214, 0.5780] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 0.3 | base_constant | Water | 44 | 0.5000 [0.3571, 0.6562] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 0.3 | base_constant | Barren | 160 | 0.5000 [0.4250, 0.5687] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 0.3 | base_constant | Forest | 82 | 0.5000 [0.3882, 0.6092] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 0.3 | base_constant | Agriculture | 120 | 0.5000 [0.4144, 0.5882] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 0.3 | b13_real | Pooled presence | 694 | 0.5461 [0.5101, 0.5843] | 0.5461 [0.5282, 0.5652] | 0.9510 | 0 | FAIL |
| 0.3 | b13_real | Dominant class | 200 | 0.5950 [0.5250, 0.6650] | 0.4832 [0.4322, 0.5707] | — | 0 | — |
| 0.3 | b13_real | Domain (Rural only) | 200 | 0.9700 [0.9450, 0.9900] | — | — | 0 | — |
| 0.3 | b13_real | Background | 14 | 0.5000 [0.2308, 0.7778] | 0.5000 [0.5000, 0.5000] | 1.0000 | 0 | FAIL |
| 0.3 | b13_real | Building | 120 | 0.6583 [0.5739, 0.7417] | 0.6583 [0.6000, 0.7178] | 0.8417 | 0 | PASS |
| 0.3 | b13_real | Road | 154 | 0.5195 [0.4444, 0.6000] | 0.5195 [0.5000, 0.5429] | 0.9805 | 0 | FAIL |
| 0.3 | b13_real | Water | 44 | 0.6591 [0.5094, 0.7964] | 0.6591 [0.5453, 0.7737] | 0.7955 | 0 | PASS |
| 0.3 | b13_real | Barren | 160 | 0.5000 [0.4313, 0.5750] | 0.5000 [0.5000, 0.5000] | 1.0000 | 0 | FAIL |
| 0.3 | b13_real | Forest | 82 | 0.5000 [0.3908, 0.6118] | 0.5000 [0.5000, 0.5000] | 1.0000 | 0 | FAIL |
| 0.3 | b13_real | Agriculture | 120 | 0.5250 [0.4390, 0.6107] | 0.5250 [0.5000, 0.5556] | 0.9750 | 0 | FAIL |
| 0.3 | b13_constant | Pooled presence | 694 | 0.5000 [0.4576, 0.5395] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 0.3 | b13_constant | Dominant class | 200 | 0.2200 [0.1650, 0.2750] | 0.1667 [0.1667, 0.1667] | — | 0 | — |
| 0.3 | b13_constant | Domain (Rural only) | 200 | 1.0000 [1.0000, 1.0000] | — | — | 0 | — |
| 0.3 | b13_constant | Background | 14 | 0.5000 [0.2222, 0.7692] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 0.3 | b13_constant | Building | 120 | 0.5000 [0.4132, 0.5878] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 0.3 | b13_constant | Road | 154 | 0.5000 [0.4214, 0.5780] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 0.3 | b13_constant | Water | 44 | 0.5000 [0.3571, 0.6562] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 0.3 | b13_constant | Barren | 160 | 0.5000 [0.4250, 0.5687] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 0.3 | b13_constant | Forest | 82 | 0.5000 [0.3882, 0.6092] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 0.3 | b13_constant | Agriculture | 120 | 0.5000 [0.4144, 0.5882] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 1 | base_real | Pooled presence | 694 | 0.5893 [0.5494, 0.6306] | 0.5893 [0.5555, 0.6240] | 0.2478 | 0 | PASS |
| 1 | base_real | Dominant class | 200 | 0.6000 [0.5300, 0.6650] | 0.3371 [0.2643, 0.4575] | — | 0 | — |
| 1 | base_real | Domain (Rural only) | 200 | 0.9750 [0.9500, 0.9950] | — | — | 0 | — |
| 1 | base_real | Background | 14 | 0.5000 [0.2222, 0.7692] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 1 | base_real | Building | 120 | 0.5167 [0.4297, 0.6066] | 0.5167 [0.5000, 0.5410] | 0.0167 | 0 | FAIL |
| 1 | base_real | Road | 154 | 0.5649 [0.4834, 0.6405] | 0.5649 [0.5112, 0.6208] | 0.1429 | 0 | PASS |
| 1 | base_real | Water | 44 | 0.5000 [0.3571, 0.6562] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 1 | base_real | Barren | 160 | 0.5563 [0.4817, 0.6329] | 0.5563 [0.4872, 0.6275] | 0.3187 | 0 | FAIL |
| 1 | base_real | Forest | 82 | 0.6707 [0.5634, 0.7683] | 0.6707 [0.5677, 0.7671] | 0.6341 | 0 | PASS |
| 1 | base_real | Agriculture | 120 | 0.7250 [0.6435, 0.8051] | 0.7250 [0.6453, 0.8031] | 0.3750 | 0 | PASS |
| 1 | base_constant | Pooled presence | 694 | 0.5000 [0.4576, 0.5395] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 1 | base_constant | Dominant class | 200 | 0.2200 [0.1650, 0.2750] | 0.1667 [0.1667, 0.1667] | — | 0 | — |
| 1 | base_constant | Domain (Rural only) | 200 | 1.0000 [1.0000, 1.0000] | — | — | 0 | — |
| 1 | base_constant | Background | 14 | 0.5000 [0.2222, 0.7692] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 1 | base_constant | Building | 120 | 0.5000 [0.4132, 0.5878] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 1 | base_constant | Road | 154 | 0.5000 [0.4214, 0.5780] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 1 | base_constant | Water | 44 | 0.5000 [0.3571, 0.6562] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 1 | base_constant | Barren | 160 | 0.5000 [0.4250, 0.5687] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 1 | base_constant | Forest | 82 | 0.5000 [0.3882, 0.6092] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 1 | base_constant | Agriculture | 120 | 0.5000 [0.4144, 0.5882] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 1 | b13_real | Pooled presence | 694 | 0.5576 [0.5206, 0.5952] | 0.5576 [0.5311, 0.5836] | 0.8876 | 0 | FAIL |
| 1 | b13_real | Dominant class | 200 | 0.3500 [0.2850, 0.4150] | 0.3554 [0.3281, 0.3825] | — | 0 | — |
| 1 | b13_real | Domain (Rural only) | 200 | 0.9500 [0.9200, 0.9800] | — | — | 0 | — |
| 1 | b13_real | Background | 14 | 0.5000 [0.2308, 0.7778] | 0.5000 [0.5000, 0.5000] | 1.0000 | 0 | FAIL |
| 1 | b13_real | Building | 120 | 0.6583 [0.5763, 0.7442] | 0.6583 [0.5903, 0.7273] | 0.7750 | 0 | PASS |
| 1 | b13_real | Road | 154 | 0.5584 [0.4839, 0.6346] | 0.5584 [0.5218, 0.5992] | 0.9286 | 0 | FAIL |
| 1 | b13_real | Water | 44 | 0.6591 [0.5111, 0.7941] | 0.6591 [0.5250, 0.7778] | 0.7045 | 0 | PASS |
| 1 | b13_real | Barren | 160 | 0.5125 [0.4375, 0.5915] | 0.5125 [0.4583, 0.5678] | 0.8625 | 0 | FAIL |
| 1 | b13_real | Forest | 82 | 0.5000 [0.3908, 0.6118] | 0.5000 [0.5000, 0.5000] | 1.0000 | 0 | FAIL |
| 1 | b13_real | Agriculture | 120 | 0.5250 [0.4380, 0.6111] | 0.5250 [0.4913, 0.5614] | 0.9583 | 0 | FAIL |
| 1 | b13_constant | Pooled presence | 694 | 0.5000 [0.4576, 0.5395] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 1 | b13_constant | Dominant class | 200 | 0.2200 [0.1650, 0.2750] | 0.1667 [0.1667, 0.1667] | — | 0 | — |
| 1 | b13_constant | Domain (Rural only) | 200 | 1.0000 [1.0000, 1.0000] | — | — | 0 | — |
| 1 | b13_constant | Background | 14 | 0.5000 [0.2222, 0.7692] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 1 | b13_constant | Building | 120 | 0.5000 [0.4132, 0.5878] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 1 | b13_constant | Road | 154 | 0.5000 [0.4214, 0.5780] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 1 | b13_constant | Water | 44 | 0.5000 [0.3571, 0.6562] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 1 | b13_constant | Barren | 160 | 0.5000 [0.4250, 0.5687] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 1 | b13_constant | Forest | 82 | 0.5000 [0.3882, 0.6092] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 1 | b13_constant | Agriculture | 120 | 0.5000 [0.4144, 0.5882] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 2 | base_real | Pooled presence | 694 | 0.5461 [0.5022, 0.5858] | 0.5461 [0.5219, 0.5689] | 0.1268 | 0 | PASS |
| 2 | base_real | Dominant class | 200 | 0.4000 [0.3350, 0.4700] | 0.2644 [0.1823, 0.3875] | — | 0 | — |
| 2 | base_real | Domain (Rural only) | 200 | 0.9950 [0.9850, 1.0000] | — | — | 0 | — |
| 2 | base_real | Background | 14 | 0.6429 [0.3636, 0.8889] | 0.6429 [0.5000, 0.8333] | 0.1429 | 0 | FAIL |
| 2 | base_real | Building | 120 | 0.5000 [0.4132, 0.5878] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 2 | base_real | Road | 154 | 0.5649 [0.4846, 0.6393] | 0.5649 [0.5260, 0.6069] | 0.0779 | 0 | PASS |
| 2 | base_real | Water | 44 | 0.5000 [0.3571, 0.6562] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 2 | base_real | Barren | 160 | 0.5188 [0.4444, 0.5938] | 0.5187 [0.4568, 0.5794] | 0.1938 | 0 | FAIL |
| 2 | base_real | Forest | 82 | 0.6098 [0.5000, 0.7105] | 0.6098 [0.5072, 0.7097] | 0.3780 | 0 | PASS |
| 2 | base_real | Agriculture | 120 | 0.5667 [0.4783, 0.6579] | 0.5667 [0.5155, 0.6242] | 0.1000 | 0 | PASS |
| 2 | base_constant | Pooled presence | 694 | 0.5000 [0.4576, 0.5395] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 2 | base_constant | Dominant class | 200 | 0.2200 [0.1650, 0.2750] | 0.1667 [0.1667, 0.1667] | — | 0 | — |
| 2 | base_constant | Domain (Rural only) | 200 | 1.0000 [1.0000, 1.0000] | — | — | 0 | — |
| 2 | base_constant | Background | 14 | 0.5000 [0.2222, 0.7692] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 2 | base_constant | Building | 120 | 0.5000 [0.4132, 0.5878] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 2 | base_constant | Road | 154 | 0.5000 [0.4214, 0.5780] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 2 | base_constant | Water | 44 | 0.5000 [0.3571, 0.6562] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 2 | base_constant | Barren | 160 | 0.5000 [0.4250, 0.5687] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 2 | base_constant | Forest | 82 | 0.5000 [0.3882, 0.6092] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 2 | base_constant | Agriculture | 120 | 0.5000 [0.4144, 0.5882] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 2 | b13_real | Pooled presence | 694 | 0.5605 [0.5231, 0.6017] | 0.5605 [0.5222, 0.5972] | 0.7406 | 0 | PASS |
| 2 | b13_real | Dominant class | 200 | 0.1550 [0.1050, 0.2100] | 0.2582 [0.2295, 0.2891] | — | 0 | — |
| 2 | b13_real | Domain (Rural only) | 200 | 0.9900 [0.9750, 1.0000] | — | — | 0 | — |
| 2 | b13_real | Background | 14 | 0.5000 [0.2308, 0.7778] | 0.5000 [0.5000, 0.5000] | 1.0000 | 0 | FAIL |
| 2 | b13_real | Building | 120 | 0.6833 [0.6000, 0.7652] | 0.6833 [0.6000, 0.7653] | 0.4833 | 0 | PASS |
| 2 | b13_real | Road | 154 | 0.5649 [0.4902, 0.6447] | 0.5649 [0.5038, 0.6283] | 0.7792 | 0 | PASS |
| 2 | b13_real | Water | 44 | 0.6591 [0.5111, 0.7917] | 0.6591 [0.5089, 0.7931] | 0.5682 | 0 | PASS |
| 2 | b13_real | Barren | 160 | 0.4938 [0.4193, 0.5723] | 0.4938 [0.4348, 0.5544] | 0.8063 | 0 | FAIL |
| 2 | b13_real | Forest | 82 | 0.5000 [0.3947, 0.6047] | 0.5000 [0.4659, 0.5341] | 0.9756 | 0 | FAIL |
| 2 | b13_real | Agriculture | 120 | 0.5333 [0.4421, 0.6250] | 0.5333 [0.4536, 0.6121] | 0.7333 | 0 | FAIL |
| 2 | b13_constant | Pooled presence | 694 | 0.5000 [0.4576, 0.5395] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 2 | b13_constant | Dominant class | 200 | 0.2200 [0.1650, 0.2750] | 0.1667 [0.1667, 0.1667] | — | 0 | — |
| 2 | b13_constant | Domain (Rural only) | 200 | 1.0000 [1.0000, 1.0000] | — | — | 0 | — |
| 2 | b13_constant | Background | 14 | 0.5000 [0.2222, 0.7692] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 2 | b13_constant | Building | 120 | 0.5000 [0.4132, 0.5878] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 2 | b13_constant | Road | 154 | 0.5000 [0.4214, 0.5780] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 2 | b13_constant | Water | 44 | 0.5000 [0.3571, 0.6562] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 2 | b13_constant | Barren | 160 | 0.5000 [0.4250, 0.5687] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 2 | b13_constant | Forest | 82 | 0.5000 [0.3882, 0.6092] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 2 | b13_constant | Agriculture | 120 | 0.5000 [0.4144, 0.5882] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 5 | base_real | Pooled presence | 694 | 0.5000 [0.4555, 0.5422] | 0.5000 [0.4857, 0.5135] | 0.0375 | 0 | FAIL |
| 5 | base_real | Dominant class | 200 | 0.2100 [0.1550, 0.2650] | 0.1956 [0.1170, 0.3174] | — | 0 | — |
| 5 | base_real | Domain (Rural only) | 200 | 1.0000 [1.0000, 1.0000] | — | — | 0 | — |
| 5 | base_real | Background | 14 | 0.5000 [0.2222, 0.7692] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 5 | base_real | Building | 120 | 0.5000 [0.4132, 0.5878] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 5 | base_real | Road | 154 | 0.5000 [0.4214, 0.5780] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 5 | base_real | Water | 44 | 0.5000 [0.3571, 0.6562] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 5 | base_real | Barren | 160 | 0.5000 [0.4233, 0.5696] | 0.5000 [0.4823, 0.5181] | 0.0125 | 0 | FAIL |
| 5 | base_real | Forest | 82 | 0.5000 [0.3882, 0.6068] | 0.5000 [0.3991, 0.5954] | 0.2927 | 0 | FAIL |
| 5 | base_real | Agriculture | 120 | 0.5000 [0.4144, 0.5882] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 5 | base_constant | Pooled presence | 694 | 0.5000 [0.4576, 0.5395] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 5 | base_constant | Dominant class | 200 | 0.2200 [0.1650, 0.2750] | 0.1667 [0.1667, 0.1667] | — | 0 | — |
| 5 | base_constant | Domain (Rural only) | 200 | 1.0000 [1.0000, 1.0000] | — | — | 0 | — |
| 5 | base_constant | Background | 14 | 0.5000 [0.2222, 0.7692] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 5 | base_constant | Building | 120 | 0.5000 [0.4132, 0.5878] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 5 | base_constant | Road | 154 | 0.5000 [0.4214, 0.5780] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 5 | base_constant | Water | 44 | 0.5000 [0.3571, 0.6562] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 5 | base_constant | Barren | 160 | 0.5000 [0.4250, 0.5687] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 5 | base_constant | Forest | 82 | 0.5000 [0.3882, 0.6092] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 5 | base_constant | Agriculture | 120 | 0.5000 [0.4144, 0.5882] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 5 | b13_real | Pooled presence | 694 | 0.4914 [0.4526, 0.5272] | 0.4914 [0.4532, 0.5271] | 0.4899 | 0 | FAIL |
| 5 | b13_real | Dominant class | 200 | 0.2000 [0.1450, 0.2600] | 0.2025 [0.1336, 0.3195] | — | 0 | — |
| 5 | b13_real | Domain (Rural only) | 200 | 1.0000 [1.0000, 1.0000] | — | — | 0 | — |
| 5 | b13_real | Background | 14 | 0.5000 [0.2308, 0.7778] | 0.5000 [0.5000, 0.5000] | 1.0000 | 0 | FAIL |
| 5 | b13_real | Building | 120 | 0.5333 [0.4444, 0.6250] | 0.5333 [0.4977, 0.5763] | 0.0500 | 0 | FAIL |
| 5 | b13_real | Road | 154 | 0.5065 [0.4268, 0.5844] | 0.5065 [0.4564, 0.5564] | 0.1104 | 0 | FAIL |
| 5 | b13_real | Water | 44 | 0.5000 [0.3500, 0.6487] | 0.5000 [0.3789, 0.6167] | 0.1818 | 0 | FAIL |
| 5 | b13_real | Barren | 160 | 0.5500 [0.4762, 0.6289] | 0.5500 [0.5084, 0.5953] | 0.9125 | 0 | FAIL |
| 5 | b13_real | Forest | 82 | 0.4878 [0.3750, 0.6000] | 0.4878 [0.4595, 0.5000] | 0.9878 | 0 | FAIL |
| 5 | b13_real | Agriculture | 120 | 0.3500 [0.2636, 0.4318] | 0.3500 [0.2638, 0.4319] | 0.5667 | 0 | PASS |
| 5 | b13_constant | Pooled presence | 694 | 0.5000 [0.4576, 0.5395] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 5 | b13_constant | Dominant class | 200 | 0.2200 [0.1650, 0.2750] | 0.1667 [0.1667, 0.1667] | — | 0 | — |
| 5 | b13_constant | Domain (Rural only) | 200 | 1.0000 [1.0000, 1.0000] | — | — | 0 | — |
| 5 | b13_constant | Background | 14 | 0.5000 [0.2222, 0.7692] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 5 | b13_constant | Building | 120 | 0.5000 [0.4132, 0.5878] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 5 | b13_constant | Road | 154 | 0.5000 [0.4214, 0.5780] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 5 | b13_constant | Water | 44 | 0.5000 [0.3571, 0.6562] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 5 | b13_constant | Barren | 160 | 0.5000 [0.4250, 0.5687] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 5 | b13_constant | Forest | 82 | 0.5000 [0.3882, 0.6092] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 5 | b13_constant | Agriculture | 120 | 0.5000 [0.4144, 0.5882] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 10 | base_real | Pooled presence | 694 | 0.5029 [0.4599, 0.5426] | 0.5029 [0.4974, 0.5087] | 0.0058 | 0 | FAIL |
| 10 | base_real | Dominant class | 200 | 0.1950 [0.1450, 0.2500] | 0.1462 [0.1256, 0.1648] | — | 0 | — |
| 10 | base_real | Domain (Rural only) | 200 | 0.9900 [0.9750, 1.0000] | — | — | 0 | — |
| 10 | base_real | Background | 14 | 0.5000 [0.2222, 0.7692] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 10 | base_real | Building | 120 | 0.5000 [0.4132, 0.5878] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 10 | base_real | Road | 154 | 0.5000 [0.4214, 0.5780] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 10 | base_real | Water | 44 | 0.5000 [0.3571, 0.6562] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 10 | base_real | Barren | 160 | 0.5000 [0.4250, 0.5687] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 10 | base_real | Forest | 82 | 0.5244 [0.4105, 0.6322] | 0.5244 [0.4810, 0.5714] | 0.0488 | 0 | FAIL |
| 10 | base_real | Agriculture | 120 | 0.5000 [0.4144, 0.5882] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 10 | base_constant | Pooled presence | 694 | 0.5000 [0.4576, 0.5395] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 10 | base_constant | Dominant class | 200 | 0.2200 [0.1650, 0.2750] | 0.1667 [0.1667, 0.1667] | — | 0 | — |
| 10 | base_constant | Domain (Rural only) | 200 | 1.0000 [1.0000, 1.0000] | — | — | 0 | — |
| 10 | base_constant | Background | 14 | 0.5000 [0.2222, 0.7692] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 10 | base_constant | Building | 120 | 0.5000 [0.4132, 0.5878] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 10 | base_constant | Road | 154 | 0.5000 [0.4214, 0.5780] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 10 | base_constant | Water | 44 | 0.5000 [0.3571, 0.6562] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 10 | base_constant | Barren | 160 | 0.5000 [0.4250, 0.5687] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 10 | base_constant | Forest | 82 | 0.5000 [0.3882, 0.6092] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 10 | base_constant | Agriculture | 120 | 0.5000 [0.4144, 0.5882] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 10 | b13_real | Pooled presence | 694 | 0.5043 [0.4605, 0.5440] | 0.5043 [0.4735, 0.5323] | 0.2032 | 0 | FAIL |
| 10 | b13_real | Dominant class | 200 | 0.1900 [0.1350, 0.2450] | 0.2992 [0.2778, 0.3182] | — | 0 | — |
| 10 | b13_real | Domain (Rural only) | 200 | 0.9750 [0.9500, 0.9950] | — | — | 0 | — |
| 10 | b13_real | Background | 14 | 0.5000 [0.2308, 0.7778] | 0.5000 [0.5000, 0.5000] | 1.0000 | 0 | FAIL |
| 10 | b13_real | Building | 120 | 0.5167 [0.4312, 0.6063] | 0.5167 [0.4869, 0.5532] | 0.0333 | 0 | FAIL |
| 10 | b13_real | Road | 154 | 0.5000 [0.4214, 0.5780] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 10 | b13_real | Water | 44 | 0.5000 [0.3571, 0.6552] | 0.5000 [0.4348, 0.5625] | 0.0455 | 0 | FAIL |
| 10 | b13_real | Barren | 160 | 0.5312 [0.4516, 0.6037] | 0.5312 [0.4715, 0.5869] | 0.1812 | 0 | FAIL |
| 10 | b13_real | Forest | 82 | 0.5000 [0.3908, 0.6118] | 0.5000 [0.5000, 0.5000] | 1.0000 | 0 | FAIL |
| 10 | b13_real | Agriculture | 120 | 0.4667 [0.3801, 0.5574] | 0.4667 [0.4153, 0.5163] | 0.0833 | 0 | FAIL |
| 10 | b13_constant | Pooled presence | 694 | 0.5000 [0.4576, 0.5395] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 10 | b13_constant | Dominant class | 200 | 0.2200 [0.1650, 0.2750] | 0.1667 [0.1667, 0.1667] | — | 0 | — |
| 10 | b13_constant | Domain (Rural only) | 200 | 1.0000 [1.0000, 1.0000] | — | — | 0 | — |
| 10 | b13_constant | Background | 14 | 0.5000 [0.2222, 0.7692] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 10 | b13_constant | Building | 120 | 0.5000 [0.4132, 0.5878] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 10 | b13_constant | Road | 154 | 0.5000 [0.4214, 0.5780] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 10 | b13_constant | Water | 44 | 0.5000 [0.3571, 0.6562] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 10 | b13_constant | Barren | 160 | 0.5000 [0.4250, 0.5687] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 10 | b13_constant | Forest | 82 | 0.5000 [0.3882, 0.6092] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |
| 10 | b13_constant | Agriculture | 120 | 0.5000 [0.4144, 0.5882] | 0.5000 [0.5000, 0.5000] | 0.0000 | 0 | FAIL |

## All real-minus-grey gates

| GSD (m) | Task | Base difference [95% CI] and outcome | B13 difference [95% CI] and outcome |
|---:|---|---|---|
| 0.3 | Pooled presence | PASS · +0.1715 [0.1205, 0.2196] | FAIL · +0.0461 [-0.0272, 0.1225] |
| 0.3 | Dominant class | PASS · +0.4850 [0.3699, 0.5950] | PASS · +0.3750 [0.2600, 0.4850] |
| 0.3 | Domain (Rural only) | FAIL · -0.0050 [-0.0150, 0.0000] | PASS · -0.0300 [-0.0550, -0.0100] |
| 0.3 | Background | FAIL · -0.0714 [-0.2381, 0.0000] | FAIL · +0.0000 [-0.5385, 0.5556] |
| 0.3 | Building | PASS · +0.1583 [0.0901, 0.2366] | FAIL · +0.1583 [-0.0079, 0.3171] |
| 0.3 | Road | PASS · +0.1299 [0.0327, 0.2250] | FAIL · +0.0195 [-0.1338, 0.1742] |
| 0.3 | Water | PASS · +0.3409 [0.1860, 0.4884] | FAIL · +0.1591 [-0.1087, 0.4130] |
| 0.3 | Barren | FAIL · +0.0813 [-0.0311, 0.1948] | FAIL · +0.0000 [-0.1373, 0.1500] |
| 0.3 | Forest | FAIL · +0.1585 [-0.0227, 0.3378] | FAIL · +0.0000 [-0.2184, 0.2236] |
| 0.3 | Agriculture | PASS · +0.3333 [0.2018, 0.4552] | FAIL · +0.0250 [-0.1485, 0.1947] |
| 1 | Pooled presence | PASS · +0.0893 [0.0505, 0.1281] | FAIL · +0.0576 [-0.0154, 0.1304] |
| 1 | Dominant class | PASS · +0.3800 [0.2699, 0.4850] | PASS · +0.1300 [0.0250, 0.2250] |
| 1 | Domain (Rural only) | PASS · -0.0250 [-0.0500, -0.0050] | PASS · -0.0500 [-0.0800, -0.0200] |
| 1 | Background | FAIL · +0.0000 [0.0000, 0.0000] | FAIL · +0.0000 [-0.5385, 0.5556] |
| 1 | Building | FAIL · +0.0167 [0.0000, 0.0417] | PASS · +0.1583 [0.0075, 0.3145] |
| 1 | Road | PASS · +0.0649 [0.0065, 0.1258] | FAIL · +0.0584 [-0.0886, 0.2089] |
| 1 | Water | FAIL · +0.0000 [0.0000, 0.0000] | FAIL · +0.1591 [-0.0981, 0.4000] |
| 1 | Barren | FAIL · +0.0563 [-0.0256, 0.1448] | FAIL · +0.0125 [-0.1195, 0.1544] |
| 1 | Forest | FAIL · +0.1707 [0.0000, 0.3425] | FAIL · +0.0000 [-0.2184, 0.2236] |
| 1 | Agriculture | PASS · +0.2250 [0.1261, 0.3274] | FAIL · +0.0250 [-0.1488, 0.1901] |
| 2 | Pooled presence | PASS · +0.0461 [0.0213, 0.0697] | FAIL · +0.0605 [-0.0029, 0.1281] |
| 2 | Dominant class | PASS · +0.1800 [0.0850, 0.2800] | FAIL · -0.0650 [-0.1350, 0.0000] |
| 2 | Domain (Rural only) | FAIL · -0.0050 [-0.0150, 0.0000] | FAIL · -0.0100 [-0.0250, 0.0000] |
| 2 | Background | FAIL · +0.1429 [0.0000, 0.3333] | FAIL · +0.0000 [-0.5385, 0.5556] |
| 2 | Building | FAIL · +0.0000 [0.0000, 0.0000] | PASS · +0.1833 [0.0687, 0.3040] |
| 2 | Road | PASS · +0.0649 [0.0253, 0.1081] | FAIL · +0.0649 [-0.0671, 0.2096] |
| 2 | Water | FAIL · +0.0000 [0.0000, 0.0000] | FAIL · +0.1591 [-0.0667, 0.3674] |
| 2 | Barren | FAIL · +0.0188 [-0.0464, 0.0881] | FAIL · -0.0062 [-0.1345, 0.1329] |
| 2 | Forest | FAIL · +0.1098 [-0.0130, 0.2341] | FAIL · +0.0000 [-0.2105, 0.2118] |
| 2 | Agriculture | PASS · +0.0667 [0.0159, 0.1261] | FAIL · +0.0333 [-0.1167, 0.1865] |
| 5 | Pooled presence | FAIL · +0.0000 [-0.0142, 0.0131] | FAIL · -0.0086 [-0.0545, 0.0399] |
| 5 | Dominant class | FAIL · -0.0100 [-0.0700, 0.0500] | FAIL · -0.0200 [-0.0500, 0.0050] |
| 5 | Domain (Rural only) | FAIL · +0.0000 [0.0000, 0.0000] | FAIL · +0.0000 [0.0000, 0.0000] |
| 5 | Background | FAIL · +0.0000 [0.0000, 0.0000] | FAIL · +0.0000 [-0.5385, 0.5556] |
| 5 | Building | FAIL · +0.0000 [0.0000, 0.0000] | FAIL · +0.0333 [0.0000, 0.0763] |
| 5 | Road | FAIL · +0.0000 [0.0000, 0.0000] | FAIL · +0.0065 [-0.0435, 0.0592] |
| 5 | Water | FAIL · +0.0000 [0.0000, 0.0000] | FAIL · +0.0000 [-0.1321, 0.1229] |
| 5 | Barren | FAIL · +0.0000 [-0.0185, 0.0186] | FAIL · +0.0500 [-0.0859, 0.1923] |
| 5 | Forest | FAIL · +0.0000 [-0.1154, 0.1081] | FAIL · -0.0122 [-0.2299, 0.2060] |
| 5 | Agriculture | FAIL · +0.0000 [0.0000, 0.0000] | PASS · -0.1500 [-0.2807, -0.0254] |
| 10 | Pooled presence | FAIL · +0.0029 [-0.0028, 0.0086] | FAIL · +0.0043 [-0.0263, 0.0334] |
| 10 | Dominant class | FAIL · -0.0250 [-0.0550, 0.0050] | FAIL · -0.0300 [-0.0650, 0.0050] |
| 10 | Domain (Rural only) | FAIL · -0.0100 [-0.0250, 0.0000] | PASS · -0.0250 [-0.0500, -0.0050] |
| 10 | Background | FAIL · +0.0000 [0.0000, 0.0000] | FAIL · +0.0000 [-0.5385, 0.5556] |
| 10 | Building | FAIL · +0.0000 [0.0000, 0.0000] | FAIL · +0.0167 [-0.0155, 0.0531] |
| 10 | Road | FAIL · +0.0000 [0.0000, 0.0000] | FAIL · +0.0000 [0.0000, 0.0000] |
| 10 | Water | FAIL · +0.0000 [0.0000, 0.0000] | FAIL · +0.0000 [-0.0667, 0.0613] |
| 10 | Barren | FAIL · +0.0000 [0.0000, 0.0000] | FAIL · +0.0312 [-0.0307, 0.0915] |
| 10 | Forest | FAIL · +0.0244 [-0.0218, 0.0732] | FAIL · +0.0000 [-0.2184, 0.2236] |
| 10 | Agriculture | FAIL · +0.0000 [0.0000, 0.0000] | FAIL · -0.0333 [-0.0870, 0.0167] |
