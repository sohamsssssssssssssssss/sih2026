# Ladder v3

Git commit `0c7b8787712585868265d6fb25de9009ba449560`. Gate SHA-256 `8ce9ab89cf313258e666a63cb90750481869575df23fcc54cc24c68a586f3544`.

See results.json for intervals and per-question results.

| Condition | GSD | Task | Accuracy | Balanced accuracy | Yes rate | Degenerate |
|---|---:|---|---:|---:|---:|---|
| base_real | 0.3 | presence:forest | 0.659 | 0.659 | 0.695 | False |
| base_real | 0.3 | presence | 0.671 | 0.671 | 0.457 | False |
| base_real | 0.3 | presence:agriculture | 0.833 | 0.833 | 0.633 | False |
| base_real | 0.3 | dominant | 0.705 | 0.538 | — | None |
| base_real | 0.3 | domain | 0.995 | — | — | None |
| base_constant | 0.3 | presence:forest | 0.500 | 0.500 | 0.000 | True |
| base_constant | 0.3 | presence | 0.500 | 0.500 | 0.000 | True |
| base_constant | 0.3 | presence:agriculture | 0.500 | 0.500 | 0.000 | True |
| base_constant | 0.3 | dominant | 0.220 | 0.167 | — | None |
| base_constant | 0.3 | domain | 1.000 | — | — | None |
| base_real | 1 | presence:forest | 0.671 | 0.671 | 0.634 | False |
| base_real | 1 | presence | 0.589 | 0.589 | 0.248 | False |
| base_real | 1 | presence:agriculture | 0.725 | 0.725 | 0.375 | False |
| base_real | 1 | dominant | 0.600 | 0.337 | — | None |
| base_real | 1 | domain | 0.975 | — | — | None |
| base_constant | 1 | presence:forest | 0.500 | 0.500 | 0.000 | True |
| base_constant | 1 | presence | 0.500 | 0.500 | 0.000 | True |
| base_constant | 1 | presence:agriculture | 0.500 | 0.500 | 0.000 | True |
| base_constant | 1 | dominant | 0.220 | 0.167 | — | None |
| base_constant | 1 | domain | 1.000 | — | — | None |
| base_real | 2 | presence:forest | 0.610 | 0.610 | 0.378 | False |
| base_real | 2 | presence | 0.546 | 0.546 | 0.127 | False |
| base_real | 2 | presence:agriculture | 0.567 | 0.567 | 0.100 | False |
| base_real | 2 | dominant | 0.400 | 0.264 | — | None |
| base_real | 2 | domain | 0.995 | — | — | None |
| base_constant | 2 | presence:forest | 0.500 | 0.500 | 0.000 | True |
| base_constant | 2 | presence | 0.500 | 0.500 | 0.000 | True |
| base_constant | 2 | presence:agriculture | 0.500 | 0.500 | 0.000 | True |
| base_constant | 2 | dominant | 0.220 | 0.167 | — | None |
| base_constant | 2 | domain | 1.000 | — | — | None |
| base_real | 5 | presence:forest | 0.500 | 0.500 | 0.293 | True |
| base_real | 5 | presence | 0.500 | 0.500 | 0.037 | True |
| base_real | 5 | presence:agriculture | 0.500 | 0.500 | 0.000 | True |
| base_real | 5 | dominant | 0.210 | 0.196 | — | None |
| base_real | 5 | domain | 1.000 | — | — | None |
| base_constant | 5 | presence:forest | 0.500 | 0.500 | 0.000 | True |
| base_constant | 5 | presence | 0.500 | 0.500 | 0.000 | True |
| base_constant | 5 | presence:agriculture | 0.500 | 0.500 | 0.000 | True |
| base_constant | 5 | dominant | 0.220 | 0.167 | — | None |
| base_constant | 5 | domain | 1.000 | — | — | None |
| base_real | 10 | presence:forest | 0.524 | 0.524 | 0.049 | True |
| base_real | 10 | presence | 0.503 | 0.503 | 0.006 | True |
| base_real | 10 | presence:agriculture | 0.500 | 0.500 | 0.000 | True |
| base_real | 10 | dominant | 0.195 | 0.146 | — | None |
| base_real | 10 | domain | 0.990 | — | — | None |
| base_constant | 10 | presence:forest | 0.500 | 0.500 | 0.000 | True |
| base_constant | 10 | presence | 0.500 | 0.500 | 0.000 | True |
| base_constant | 10 | presence:agriculture | 0.500 | 0.500 | 0.000 | True |
| base_constant | 10 | dominant | 0.220 | 0.167 | — | None |
| base_constant | 10 | domain | 1.000 | — | — | None |
| base_real | 0.3 | presence:building | 0.658 | 0.658 | 0.192 | False |
| base_real | 0.3 | presence:road | 0.630 | 0.630 | 0.377 | False |
| base_real | 0.3 | presence:barren | 0.581 | 0.581 | 0.531 | False |
| base_constant | 0.3 | presence:building | 0.500 | 0.500 | 0.000 | True |
| base_constant | 0.3 | presence:road | 0.500 | 0.500 | 0.000 | True |
| base_constant | 0.3 | presence:barren | 0.500 | 0.500 | 0.000 | True |
| base_real | 1 | presence:building | 0.517 | 0.517 | 0.017 | True |
| base_real | 1 | presence:road | 0.565 | 0.565 | 0.143 | False |
| base_real | 1 | presence:barren | 0.556 | 0.556 | 0.319 | True |
| base_constant | 1 | presence:building | 0.500 | 0.500 | 0.000 | True |
| base_constant | 1 | presence:road | 0.500 | 0.500 | 0.000 | True |
| base_constant | 1 | presence:barren | 0.500 | 0.500 | 0.000 | True |
| base_real | 2 | presence:building | 0.500 | 0.500 | 0.000 | True |
| base_real | 2 | presence:road | 0.565 | 0.565 | 0.078 | False |
| base_real | 2 | presence:barren | 0.519 | 0.519 | 0.194 | True |
| base_constant | 2 | presence:building | 0.500 | 0.500 | 0.000 | True |
| base_constant | 2 | presence:road | 0.500 | 0.500 | 0.000 | True |
| base_constant | 2 | presence:barren | 0.500 | 0.500 | 0.000 | True |
| base_real | 5 | presence:building | 0.500 | 0.500 | 0.000 | True |
| base_real | 5 | presence:road | 0.500 | 0.500 | 0.000 | True |
| base_real | 5 | presence:barren | 0.500 | 0.500 | 0.013 | True |
| base_constant | 5 | presence:building | 0.500 | 0.500 | 0.000 | True |
| base_constant | 5 | presence:road | 0.500 | 0.500 | 0.000 | True |
| base_constant | 5 | presence:barren | 0.500 | 0.500 | 0.000 | True |
| base_real | 10 | presence:building | 0.500 | 0.500 | 0.000 | True |
| base_real | 10 | presence:road | 0.500 | 0.500 | 0.000 | True |
| base_real | 10 | presence:barren | 0.500 | 0.500 | 0.000 | True |
| base_constant | 10 | presence:building | 0.500 | 0.500 | 0.000 | True |
| base_constant | 10 | presence:road | 0.500 | 0.500 | 0.000 | True |
| base_constant | 10 | presence:barren | 0.500 | 0.500 | 0.000 | True |
| base_real | 0.3 | presence:water | 0.841 | 0.841 | 0.386 | False |
| base_constant | 0.3 | presence:water | 0.500 | 0.500 | 0.000 | True |
| base_real | 1 | presence:water | 0.500 | 0.500 | 0.000 | True |
| base_constant | 1 | presence:water | 0.500 | 0.500 | 0.000 | True |
| base_real | 2 | presence:water | 0.500 | 0.500 | 0.000 | True |
| base_constant | 2 | presence:water | 0.500 | 0.500 | 0.000 | True |
| base_real | 5 | presence:water | 0.500 | 0.500 | 0.000 | True |
| base_constant | 5 | presence:water | 0.500 | 0.500 | 0.000 | True |
| base_real | 10 | presence:water | 0.500 | 0.500 | 0.000 | True |
| base_constant | 10 | presence:water | 0.500 | 0.500 | 0.000 | True |
| base_real | 0.3 | presence:background | 0.429 | 0.429 | 0.071 | True |
| base_constant | 0.3 | presence:background | 0.500 | 0.500 | 0.000 | True |
| base_real | 1 | presence:background | 0.500 | 0.500 | 0.000 | True |
| base_constant | 1 | presence:background | 0.500 | 0.500 | 0.000 | True |
| base_real | 2 | presence:background | 0.643 | 0.643 | 0.143 | True |
| base_constant | 2 | presence:background | 0.500 | 0.500 | 0.000 | True |
| base_real | 5 | presence:background | 0.500 | 0.500 | 0.000 | True |
| base_constant | 5 | presence:background | 0.500 | 0.500 | 0.000 | True |
| base_real | 10 | presence:background | 0.500 | 0.500 | 0.000 | True |
| base_constant | 10 | presence:background | 0.500 | 0.500 | 0.000 | True |
| b13_real | 0.3 | presence:forest | 0.500 | 0.500 | 1.000 | True |
| b13_real | 0.3 | presence | 0.546 | 0.546 | 0.951 | True |
| b13_real | 0.3 | presence:agriculture | 0.525 | 0.525 | 0.975 | True |
| b13_real | 0.3 | dominant | 0.595 | 0.483 | — | None |
| b13_real | 0.3 | domain | 0.970 | — | — | None |
| b13_constant | 0.3 | presence:forest | 0.500 | 0.500 | 0.000 | True |
| b13_constant | 0.3 | presence | 0.500 | 0.500 | 0.000 | True |
| b13_constant | 0.3 | presence:agriculture | 0.500 | 0.500 | 0.000 | True |
| b13_constant | 0.3 | dominant | 0.220 | 0.167 | — | None |
| b13_constant | 0.3 | domain | 1.000 | — | — | None |
| b13_real | 1 | presence:forest | 0.500 | 0.500 | 1.000 | True |
| b13_real | 1 | presence | 0.558 | 0.558 | 0.888 | True |
| b13_real | 1 | presence:agriculture | 0.525 | 0.525 | 0.958 | True |
| b13_real | 1 | dominant | 0.350 | 0.355 | — | None |
| b13_real | 1 | domain | 0.950 | — | — | None |
| b13_constant | 1 | presence:forest | 0.500 | 0.500 | 0.000 | True |
| b13_constant | 1 | presence | 0.500 | 0.500 | 0.000 | True |
| b13_constant | 1 | presence:agriculture | 0.500 | 0.500 | 0.000 | True |
| b13_constant | 1 | dominant | 0.220 | 0.167 | — | None |
| b13_constant | 1 | domain | 1.000 | — | — | None |
| b13_real | 2 | presence:forest | 0.500 | 0.500 | 0.976 | True |
| b13_real | 2 | presence | 0.561 | 0.561 | 0.741 | False |
| b13_real | 2 | presence:agriculture | 0.533 | 0.533 | 0.733 | True |
| b13_real | 2 | dominant | 0.155 | 0.258 | — | None |
| b13_real | 2 | domain | 0.990 | — | — | None |
| b13_constant | 2 | presence:forest | 0.500 | 0.500 | 0.000 | True |
| b13_constant | 2 | presence | 0.500 | 0.500 | 0.000 | True |
| b13_constant | 2 | presence:agriculture | 0.500 | 0.500 | 0.000 | True |
| b13_constant | 2 | dominant | 0.220 | 0.167 | — | None |
| b13_constant | 2 | domain | 1.000 | — | — | None |
| b13_real | 5 | presence:forest | 0.488 | 0.488 | 0.988 | True |
| b13_real | 5 | presence | 0.491 | 0.491 | 0.490 | True |
| b13_real | 5 | presence:agriculture | 0.350 | 0.350 | 0.567 | False |
| b13_real | 5 | dominant | 0.200 | 0.203 | — | None |
| b13_real | 5 | domain | 1.000 | — | — | None |
| b13_constant | 5 | presence:forest | 0.500 | 0.500 | 0.000 | True |
| b13_constant | 5 | presence | 0.500 | 0.500 | 0.000 | True |
| b13_constant | 5 | presence:agriculture | 0.500 | 0.500 | 0.000 | True |
| b13_constant | 5 | dominant | 0.220 | 0.167 | — | None |
| b13_constant | 5 | domain | 1.000 | — | — | None |
| b13_real | 10 | presence:forest | 0.500 | 0.500 | 1.000 | True |
| b13_real | 10 | presence | 0.504 | 0.504 | 0.203 | True |
| b13_real | 10 | presence:agriculture | 0.467 | 0.467 | 0.083 | True |
| b13_real | 10 | dominant | 0.190 | 0.299 | — | None |
| b13_real | 10 | domain | 0.975 | — | — | None |
| b13_constant | 10 | presence:forest | 0.500 | 0.500 | 0.000 | True |
| b13_constant | 10 | presence | 0.500 | 0.500 | 0.000 | True |
| b13_constant | 10 | presence:agriculture | 0.500 | 0.500 | 0.000 | True |
| b13_constant | 10 | dominant | 0.220 | 0.167 | — | None |
| b13_constant | 10 | domain | 1.000 | — | — | None |
| b13_real | 0.3 | presence:building | 0.658 | 0.658 | 0.842 | False |
| b13_real | 0.3 | presence:road | 0.519 | 0.519 | 0.981 | True |
| b13_real | 0.3 | presence:barren | 0.500 | 0.500 | 1.000 | True |
| b13_constant | 0.3 | presence:building | 0.500 | 0.500 | 0.000 | True |
| b13_constant | 0.3 | presence:road | 0.500 | 0.500 | 0.000 | True |
| b13_constant | 0.3 | presence:barren | 0.500 | 0.500 | 0.000 | True |
| b13_real | 1 | presence:building | 0.658 | 0.658 | 0.775 | False |
| b13_real | 1 | presence:road | 0.558 | 0.558 | 0.929 | True |
| b13_real | 1 | presence:barren | 0.512 | 0.512 | 0.863 | True |
| b13_constant | 1 | presence:building | 0.500 | 0.500 | 0.000 | True |
| b13_constant | 1 | presence:road | 0.500 | 0.500 | 0.000 | True |
| b13_constant | 1 | presence:barren | 0.500 | 0.500 | 0.000 | True |
| b13_real | 2 | presence:building | 0.683 | 0.683 | 0.483 | False |
| b13_real | 2 | presence:road | 0.565 | 0.565 | 0.779 | False |
| b13_real | 2 | presence:barren | 0.494 | 0.494 | 0.806 | True |
| b13_constant | 2 | presence:building | 0.500 | 0.500 | 0.000 | True |
| b13_constant | 2 | presence:road | 0.500 | 0.500 | 0.000 | True |
| b13_constant | 2 | presence:barren | 0.500 | 0.500 | 0.000 | True |
| b13_real | 5 | presence:building | 0.533 | 0.533 | 0.050 | True |
| b13_real | 5 | presence:road | 0.506 | 0.506 | 0.110 | True |
| b13_real | 5 | presence:barren | 0.550 | 0.550 | 0.912 | True |
| b13_constant | 5 | presence:building | 0.500 | 0.500 | 0.000 | True |
| b13_constant | 5 | presence:road | 0.500 | 0.500 | 0.000 | True |
| b13_constant | 5 | presence:barren | 0.500 | 0.500 | 0.000 | True |
| b13_real | 10 | presence:building | 0.517 | 0.517 | 0.033 | True |
| b13_real | 10 | presence:road | 0.500 | 0.500 | 0.000 | True |
| b13_real | 10 | presence:barren | 0.531 | 0.531 | 0.181 | True |
| b13_constant | 10 | presence:building | 0.500 | 0.500 | 0.000 | True |
| b13_constant | 10 | presence:road | 0.500 | 0.500 | 0.000 | True |
| b13_constant | 10 | presence:barren | 0.500 | 0.500 | 0.000 | True |
| b13_real | 0.3 | presence:water | 0.659 | 0.659 | 0.795 | False |
| b13_constant | 0.3 | presence:water | 0.500 | 0.500 | 0.000 | True |
| b13_real | 1 | presence:water | 0.659 | 0.659 | 0.705 | False |
| b13_constant | 1 | presence:water | 0.500 | 0.500 | 0.000 | True |
| b13_real | 2 | presence:water | 0.659 | 0.659 | 0.568 | False |
| b13_constant | 2 | presence:water | 0.500 | 0.500 | 0.000 | True |
| b13_real | 5 | presence:water | 0.500 | 0.500 | 0.182 | True |
| b13_constant | 5 | presence:water | 0.500 | 0.500 | 0.000 | True |
| b13_real | 10 | presence:water | 0.500 | 0.500 | 0.045 | True |
| b13_constant | 10 | presence:water | 0.500 | 0.500 | 0.000 | True |
| b13_real | 0.3 | presence:background | 0.500 | 0.500 | 1.000 | True |
| b13_constant | 0.3 | presence:background | 0.500 | 0.500 | 0.000 | True |
| b13_real | 1 | presence:background | 0.500 | 0.500 | 1.000 | True |
| b13_constant | 1 | presence:background | 0.500 | 0.500 | 0.000 | True |
| b13_real | 2 | presence:background | 0.500 | 0.500 | 1.000 | True |
| b13_constant | 2 | presence:background | 0.500 | 0.500 | 0.000 | True |
| b13_real | 5 | presence:background | 0.500 | 0.500 | 1.000 | True |
| b13_constant | 5 | presence:background | 0.500 | 0.500 | 0.000 | True |
| b13_real | 10 | presence:background | 0.500 | 0.500 | 1.000 | True |
| b13_constant | 10 | presence:background | 0.500 | 0.500 | 0.000 | True |

| Real condition | GSD | Task | Real minus grey [95% CI] | Image-dependent |
|---|---:|---|---|---|
| base_real | 0.3 | presence:forest | +0.159 [-0.02272727272727271, 0.33783783783783783] | False |
| base_real | 0.3 | presence | +0.171 [0.12050658980357697, 0.2196142576565644] | True |
| base_real | 0.3 | presence:agriculture | +0.333 [0.2018348623853211, 0.4552253974032278] | True |
| base_real | 0.3 | dominant | +0.485 [0.369875, 0.595] | True |
| base_real | 0.3 | domain | -0.005 [-0.015000000000000013, 0.0] | False |
| base_real | 1 | presence:forest | +0.171 [0.0, 0.3424657534246575] | False |
| base_real | 1 | presence | +0.089 [0.050517864865906495, 0.1280621669811125] | True |
| base_real | 1 | presence:agriculture | +0.225 [0.12605042016806722, 0.3274374427830333] | True |
| base_real | 1 | dominant | +0.380 [0.26987500000000003, 0.48500000000000004] | True |
| base_real | 1 | domain | -0.025 [-0.050000000000000044, -0.0050000000000000044] | True |
| base_real | 2 | presence:forest | +0.110 [-0.012987012987012938, 0.2340556868925663] | False |
| base_real | 2 | presence | +0.046 [0.021306818181818232, 0.06966871434839836] | True |
| base_real | 2 | presence:agriculture | +0.067 [0.015869891263592, 0.12614757614757607] | True |
| base_real | 2 | dominant | +0.180 [0.08500000000000002, 0.27999999999999997] | True |
| base_real | 2 | domain | -0.005 [-0.015000000000000013, 0.0] | False |
| base_real | 5 | presence:forest | +0.000 [-0.11538461538461542, 0.10810810810810811] | False |
| base_real | 5 | presence | +0.000 [-0.01420759800729358, 0.013081395348837233] | False |
| base_real | 5 | presence:agriculture | +0.000 [0.0, 0.0] | False |
| base_real | 5 | dominant | -0.010 [-0.07, 0.04999999999999999] | False |
| base_real | 5 | domain | +0.000 [0.0, 0.0] | False |
| base_real | 10 | presence:forest | +0.024 [-0.021757449926722017, 0.07317073170731703] | False |
| base_real | 10 | presence | +0.003 [-0.0028012186641837565, 0.008645533141210361] | False |
| base_real | 10 | presence:agriculture | +0.000 [0.0, 0.0] | False |
| base_real | 10 | dominant | -0.025 [-0.05500000000000002, 0.004999999999999977] | False |
| base_real | 10 | domain | -0.010 [-0.025000000000000022, 0.0] | False |
| base_real | 0.3 | presence:building | +0.158 [0.09014197892271657, 0.23664624347127364] | True |
| base_real | 0.3 | presence:road | +0.130 [0.032679738562091554, 0.22499999999999998] | True |
| base_real | 0.3 | presence:barren | +0.081 [-0.03106075310559012, 0.19480927877154297] | False |
| base_real | 1 | presence:building | +0.017 [0.0, 0.04166666666666663] | False |
| base_real | 1 | presence:road | +0.065 [0.0064924591537494246, 0.12578616352201255] | True |
| base_real | 1 | presence:barren | +0.056 [-0.025641025641025605, 0.14475478468899516] | False |
| base_real | 2 | presence:building | +0.000 [0.0, 0.0] | False |
| base_real | 2 | presence:road | +0.065 [0.025316455696202556, 0.10810810810810806] | True |
| base_real | 2 | presence:barren | +0.019 [-0.04636534216335531, 0.08805031446540879] | False |
| base_real | 5 | presence:building | +0.000 [0.0, 0.0] | False |
| base_real | 5 | presence:road | +0.000 [0.0, 0.0] | False |
| base_real | 5 | presence:barren | +0.000 [-0.018518518518518545, 0.018633540372670843] | False |
| base_real | 10 | presence:building | +0.000 [0.0, 0.0] | False |
| base_real | 10 | presence:road | +0.000 [0.0, 0.0] | False |
| base_real | 10 | presence:barren | +0.000 [0.0, 0.0] | False |
| base_real | 0.3 | presence:water | +0.341 [0.18600061199510404, 0.48837209302325574] | True |
| base_real | 1 | presence:water | +0.000 [0.0, 0.0] | False |
| base_real | 2 | presence:water | +0.000 [0.0, 0.0] | False |
| base_real | 5 | presence:water | +0.000 [0.0, 0.0] | False |
| base_real | 10 | presence:water | +0.000 [0.0, 0.0] | False |
| base_real | 0.3 | presence:background | -0.071 [-0.23809523809523814, 0.0] | False |
| base_real | 1 | presence:background | +0.000 [0.0, 0.0] | False |
| base_real | 2 | presence:background | +0.143 [0.0, 0.33333333333333337] | False |
| base_real | 5 | presence:background | +0.000 [0.0, 0.0] | False |
| base_real | 10 | presence:background | +0.000 [0.0, 0.0] | False |
| b13_real | 0.3 | presence:forest | +0.000 [-0.21839080459770116, 0.22355342136854733] | False |
| b13_real | 0.3 | presence | +0.046 [-0.027221606722219308, 0.12251330671989348] | False |
| b13_real | 0.3 | presence:agriculture | +0.025 [-0.14850807510504202, 0.19469589020548972] | False |
| b13_real | 0.3 | dominant | +0.375 [0.26, 0.485] | True |
| b13_real | 0.3 | domain | -0.030 [-0.05500000000000005, -0.010000000000000009] | True |
| b13_real | 1 | presence:forest | +0.000 [-0.21839080459770116, 0.22355342136854733] | False |
| b13_real | 1 | presence | +0.058 [-0.015391180654338603, 0.13037727547124892] | False |
| b13_real | 1 | presence:agriculture | +0.025 [-0.1487603305785124, 0.1900924832743014] | False |
| b13_real | 1 | dominant | +0.130 [0.025000000000000022, 0.225] | True |
| b13_real | 1 | domain | -0.050 [-0.07999999999999996, -0.020000000000000018] | True |
| b13_real | 2 | presence:forest | +0.000 [-0.2105409356725146, 0.2117897371714642] | False |
| b13_real | 2 | presence | +0.061 [-0.002900303626839219, 0.1280602879842177] | False |
| b13_real | 2 | presence:agriculture | +0.033 [-0.11671610169491528, 0.18645255821321077] | False |
| b13_real | 2 | dominant | -0.065 [-0.135, 0.0] | False |
| b13_real | 2 | domain | -0.010 [-0.025000000000000022, 0.0] | False |
| b13_real | 5 | presence:forest | -0.012 [-0.22990716180371357, 0.20599845201238332] | False |
| b13_real | 5 | presence | -0.009 [-0.05448466054350912, 0.0399408284023669] | False |
| b13_real | 5 | presence:agriculture | -0.150 [-0.28072966507177033, -0.02542372881355931] | True |
| b13_real | 5 | dominant | -0.020 [-0.04999999999999999, 0.0050000000000000044] | False |
| b13_real | 5 | domain | +0.000 [0.0, 0.0] | False |
| b13_real | 10 | presence:forest | +0.000 [-0.21839080459770116, 0.22355342136854733] | False |
| b13_real | 10 | presence | +0.004 [-0.026278332693046476, 0.03338781308617451] | False |
| b13_real | 10 | presence:agriculture | -0.033 [-0.08695652173913038, 0.016666666666666607] | False |
| b13_real | 10 | dominant | -0.030 [-0.065, 0.0050000000000000044] | False |
| b13_real | 10 | domain | -0.025 [-0.050000000000000044, -0.0050000000000000044] | True |
| b13_real | 0.3 | presence:building | +0.158 [-0.007874015748031482, 0.3171204793944489] | False |
| b13_real | 0.3 | presence:road | +0.019 [-0.13378824586768784, 0.17420933904804864] | False |
| b13_real | 0.3 | presence:barren | +0.000 [-0.1373045420699925, 0.15004746835443009] | False |
| b13_real | 1 | presence:building | +0.158 [0.007499999999999984, 0.31451612903225806] | True |
| b13_real | 1 | presence:road | +0.058 [-0.0886217044263485, 0.2088679986762637] | False |
| b13_real | 1 | presence:barren | +0.012 [-0.11951576307618818, 0.15435517853182507] | False |
| b13_real | 2 | presence:building | +0.183 [0.06870229007633588, 0.304] | True |
| b13_real | 2 | presence:road | +0.065 [-0.06707419381240792, 0.20958823094551637] | False |
| b13_real | 2 | presence:barren | -0.006 [-0.13450573549257758, 0.13291139240506328] | False |
| b13_real | 5 | presence:building | +0.033 [0.0, 0.07627118644067798] | False |
| b13_real | 5 | presence:road | +0.006 [-0.043485054347826055, 0.05922032938306021] | False |
| b13_real | 5 | presence:barren | +0.050 [-0.085902825115504, 0.19231366459627325] | False |
| b13_real | 10 | presence:building | +0.017 [-0.015513104466592811, 0.05309734513274339] | False |
| b13_real | 10 | presence:road | +0.000 [0.0, 0.0] | False |
| b13_real | 10 | presence:barren | +0.031 [-0.03067484662576681, 0.0914644109676391] | False |
| b13_real | 0.3 | presence:water | +0.159 [-0.10869565217391308, 0.4130434782608696] | False |
| b13_real | 1 | presence:water | +0.159 [-0.09808823529411763, 0.39999999999999997] | False |
| b13_real | 2 | presence:water | +0.159 [-0.06666666666666665, 0.3674023957409047] | False |
| b13_real | 5 | presence:water | +0.000 [-0.13207547169811318, 0.12286184210526291] | False |
| b13_real | 10 | presence:water | +0.000 [-0.06666666666666665, 0.061256377551020244] | False |
| b13_real | 0.3 | presence:background | +0.000 [-0.5384615384615385, 0.5555555555555556] | False |
| b13_real | 1 | presence:background | +0.000 [-0.5384615384615385, 0.5555555555555556] | False |
| b13_real | 2 | presence:background | +0.000 [-0.5384615384615385, 0.5555555555555556] | False |
| b13_real | 5 | presence:background | +0.000 [-0.5384615384615385, 0.5555555555555556] | False |
| b13_real | 10 | presence:background | +0.000 [-0.5384615384615385, 0.5555555555555556] | False |
