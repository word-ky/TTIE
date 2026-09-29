# Ours ceiling metrics: LSRW (N=50, G=50) — ceiling reference: Ours knobs and kappa selected on this target's test GT (not held-out)

| Row | Mean PSNR | Median PSNR | Mean RGB-SSIM |
|---|---|---|---|
| retinexformer | 17.1954 | 17.3106 | 0.5093 |
| snr_aware | 16.4998 | 16.2919 | 0.5135 |
| promptir | 9.1702 | 8.9125 | 0.2095 |
| promptir_dctta | 9.1616 | 8.9031 | 0.2078 |
| mr_illuminate | 17.6663 | 17.6762 | 0.5194 |
| quadprior | 16.9687 | 17.3096 | 0.5646 |
| ours_step0 | 9.1588 | 8.9132 | 0.2018 |
| ours_ttt | 15.4957 | 14.8493 | 0.4342 |
| ours_ttt_sdsd_knobs | 16.3319 | 16.1502 | 0.4440 |
| ours_v2 | 15.6614 | 15.0276 | 0.4941 |
| ours_v2_sdsd_knobs | 16.5937 | 16.4733 | 0.5151 |
| retinexformer_plus_D | 17.2559 | 17.3174 | 0.5419 |
| snr_aware_plus_D | 16.6289 | 16.4483 | 0.5405 |
| promptir_plus_D | 9.1252 | 8.8782 | 0.2041 |
| promptir_dctta_plus_D | 9.1166 | 8.8701 | 0.2025 |
| mr_illuminate_plus_D | 17.8100 | 17.8144 | 0.5653 |
| quadprior_plus_D | 16.9658 | 17.3128 | 0.5696 |
| ours_step0_plus_D | 9.1117 | 8.8727 | 0.1966 |
| ours_ttt_target_tuned (tuned on test GT) | 16.3007 | 16.0003 | 0.4484 |
| ours_v2_ceiling (tuned on test GT) | 16.5389 | 16.3455 | 0.5156 |

| Comparison | Metric | Mean Δ | Win fraction | 95% cluster CI |
|---|---|---|---|---|
| ours_v2_ceiling - retinexformer | psnr | -0.6565 | 0.420 | [-1.5467, +0.2672] |
| ours_v2_ceiling - retinexformer | rgb_ssim | +0.0062 | 0.580 | [-0.0182, +0.0300] |
| ours_v2_ceiling - snr_aware | psnr | +0.0391 | 0.380 | [-1.2693, +1.4330] |
| ours_v2_ceiling - snr_aware | rgb_ssim | +0.0021 | 0.540 | [-0.0219, +0.0259] |
| ours_v2_ceiling - promptir | psnr | +7.3687 | 1.000 | [+6.1844, +8.5610] |
| ours_v2_ceiling - promptir | rgb_ssim | +0.3061 | 0.980 | [+0.2611, +0.3512] |
| ours_v2_ceiling - promptir_dctta | psnr | +7.3773 | 1.000 | [+6.1931, +8.5700] |
| ours_v2_ceiling - promptir_dctta | rgb_ssim | +0.3078 | 0.980 | [+0.2624, +0.3533] |
| ours_v2_ceiling - mr_illuminate | psnr | -1.1274 | 0.400 | [-2.0988, -0.1565] |
| ours_v2_ceiling - mr_illuminate | rgb_ssim | -0.0038 | 0.540 | [-0.0288, +0.0198] |
| ours_v2_ceiling - quadprior | psnr | -0.4298 | 0.480 | [-1.3553, +0.5321] |
| ours_v2_ceiling - quadprior | rgb_ssim | -0.0491 | 0.320 | [-0.0691, -0.0296] |
| ours_v2_ceiling - ours_step0 | psnr | +7.3801 | 1.000 | [+6.1927, +8.5751] |
| ours_v2_ceiling - ours_step0 | rgb_ssim | +0.3138 | 0.980 | [+0.2684, +0.3593] |
| ours_v2_ceiling - ours_ttt | psnr | +1.0432 | 0.800 | [+0.5733, +1.5991] |
| ours_v2_ceiling - ours_ttt | rgb_ssim | +0.0814 | 0.900 | [+0.0652, +0.0971] |
| ours_v2_ceiling - ours_ttt_sdsd_knobs | psnr | +0.2070 | 0.560 | [-0.1587, +0.5455] |
| ours_v2_ceiling - ours_ttt_sdsd_knobs | rgb_ssim | +0.0716 | 0.900 | [+0.0555, +0.0874] |
| ours_v2_ceiling - ours_v2 | psnr | +0.8775 | 0.780 | [+0.3912, +1.4407] |
| ours_v2_ceiling - ours_v2 | rgb_ssim | +0.0215 | 0.740 | [+0.0116, +0.0326] |
| ours_v2_ceiling - ours_v2_sdsd_knobs | psnr | -0.0548 | 0.420 | [-0.4083, +0.2529] |
| ours_v2_ceiling - ours_v2_sdsd_knobs | rgb_ssim | +0.0005 | 0.520 | [-0.0023, +0.0034] |
| ours_v2_ceiling - retinexformer_plus_D | psnr | -0.7171 | 0.380 | [-1.6040, +0.2034] |
| ours_v2_ceiling - retinexformer_plus_D | rgb_ssim | -0.0263 | 0.380 | [-0.0471, -0.0062] |
| ours_v2_ceiling - snr_aware_plus_D | psnr | -0.0900 | 0.360 | [-1.4068, +1.3069] |
| ours_v2_ceiling - snr_aware_plus_D | rgb_ssim | -0.0250 | 0.380 | [-0.0461, -0.0037] |
| ours_v2_ceiling - promptir_plus_D | psnr | +7.4137 | 1.000 | [+6.2294, +8.6042] |
| ours_v2_ceiling - promptir_plus_D | rgb_ssim | +0.3115 | 1.000 | [+0.2653, +0.3581] |
| ours_v2_ceiling - promptir_dctta_plus_D | psnr | +7.4223 | 1.000 | [+6.2393, +8.6137] |
| ours_v2_ceiling - promptir_dctta_plus_D | rgb_ssim | +0.3131 | 1.000 | [+0.2667, +0.3604] |
| ours_v2_ceiling - mr_illuminate_plus_D | psnr | -1.2711 | 0.380 | [-2.2378, -0.3038] |
| ours_v2_ceiling - mr_illuminate_plus_D | rgb_ssim | -0.0497 | 0.220 | [-0.0689, -0.0318] |
| ours_v2_ceiling - quadprior_plus_D | psnr | -0.4269 | 0.480 | [-1.3482, +0.5289] |
| ours_v2_ceiling - quadprior_plus_D | rgb_ssim | -0.0540 | 0.260 | [-0.0728, -0.0356] |
| ours_v2_ceiling - ours_step0_plus_D | psnr | +7.4272 | 1.000 | [+6.2431, +8.6205] |
| ours_v2_ceiling - ours_step0_plus_D | rgb_ssim | +0.3190 | 1.000 | [+0.2723, +0.3664] |
| ours_v2_ceiling - ours_ttt_target_tuned | psnr | +0.2382 | 0.760 | [+0.1499, +0.3376] |
| ours_v2_ceiling - ours_ttt_target_tuned | rgb_ssim | +0.0671 | 0.900 | [+0.0508, +0.0829] |
