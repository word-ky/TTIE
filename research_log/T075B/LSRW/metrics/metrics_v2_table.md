# T075-A v2 metrics: LSRW (N=50, G=50)

| Row | Mean PSNR | Mean RGB-SSIM |
|---|---|---|
| retinexformer | 17.1954 | 0.5093 |
| snr_aware | 16.4998 | 0.5135 |
| promptir | 9.1702 | 0.2095 |
| promptir_dctta | 9.1616 | 0.2078 |
| mr_illuminate | 17.6663 | 0.5194 |
| quadprior | 16.9687 | 0.5646 |
| ours_step0 | 9.1588 | 0.2018 |
| ours_ttt | 15.4957 | 0.4342 |
| ours_ttt_sdsd_knobs | 16.3319 | 0.4440 |
| ours_v2 | 15.6614 | 0.4941 |
| ours_v2_sdsd_knobs | 16.5937 | 0.5151 |
| retinexformer_plus_D | 17.2559 | 0.5419 |
| snr_aware_plus_D | 16.6289 | 0.5405 |
| promptir_plus_D | 9.1252 | 0.2041 |
| promptir_dctta_plus_D | 9.1166 | 0.2025 |
| mr_illuminate_plus_D | 17.8100 | 0.5653 |
| quadprior_plus_D | 16.9658 | 0.5696 |
| ours_step0_plus_D | 9.1117 | 0.1966 |

| Comparison | Metric | Mean Δ | Win fraction | 95% cluster CI |
|---|---|---|---|---|
| ours_v2 - retinexformer | psnr | -1.5340 | 0.300 | [-2.4317, -0.5685] |
| ours_v2 - retinexformer | rgb_ssim | -0.0153 | 0.440 | [-0.0399, +0.0098] |
| ours_v2 - snr_aware | psnr | -0.8384 | 0.340 | [-2.2606, +0.6546] |
| ours_v2 - snr_aware | rgb_ssim | -0.0195 | 0.420 | [-0.0443, +0.0057] |
| ours_v2 - promptir | psnr | +6.4911 | 0.980 | [+5.2210, +7.7954] |
| ours_v2 - promptir | rgb_ssim | +0.2846 | 0.980 | [+0.2369, +0.3329] |
| ours_v2 - promptir_dctta | psnr | +6.4998 | 0.980 | [+5.2283, +7.8044] |
| ours_v2 - promptir_dctta | rgb_ssim | +0.2863 | 0.980 | [+0.2382, +0.3349] |
| ours_v2 - mr_illuminate | psnr | -2.0049 | 0.280 | [-2.9775, -0.9854] |
| ours_v2 - mr_illuminate | rgb_ssim | -0.0253 | 0.440 | [-0.0519, +0.0008] |
| ours_v2 - quadprior | psnr | -1.3073 | 0.360 | [-2.2992, -0.2478] |
| ours_v2 - quadprior | rgb_ssim | -0.0706 | 0.240 | [-0.0920, -0.0492] |
| ours_v2 - retinexformer_plus_D | psnr | -1.5946 | 0.240 | [-2.4808, -0.6323] |
| ours_v2 - retinexformer_plus_D | rgb_ssim | -0.0478 | 0.260 | [-0.0689, -0.0270] |
| ours_v2 - snr_aware_plus_D | psnr | -0.9676 | 0.320 | [-2.3916, +0.5367] |
| ours_v2 - snr_aware_plus_D | rgb_ssim | -0.0465 | 0.280 | [-0.0702, -0.0231] |
| ours_v2 - promptir_plus_D | psnr | +6.5362 | 1.000 | [+5.2654, +7.8408] |
| ours_v2 - promptir_plus_D | rgb_ssim | +0.2900 | 0.980 | [+0.2412, +0.3396] |
| ours_v2 - promptir_dctta_plus_D | psnr | +6.5447 | 1.000 | [+5.2737, +7.8507] |
| ours_v2 - promptir_dctta_plus_D | rgb_ssim | +0.2916 | 0.980 | [+0.2426, +0.3416] |
| ours_v2 - mr_illuminate_plus_D | psnr | -2.1486 | 0.260 | [-3.1087, -1.1346] |
| ours_v2 - mr_illuminate_plus_D | rgb_ssim | -0.0713 | 0.180 | [-0.0918, -0.0510] |
| ours_v2 - quadprior_plus_D | psnr | -1.3044 | 0.360 | [-2.2912, -0.2499] |
| ours_v2 - quadprior_plus_D | rgb_ssim | -0.0755 | 0.160 | [-0.0959, -0.0551] |
| ours_v2 - ours_ttt | psnr | +0.1657 | 0.640 | [+0.0884, +0.2553] |
| ours_v2 - ours_ttt | rgb_ssim | +0.0599 | 0.900 | [+0.0456, +0.0744] |
| ours_v2_sdsd_knobs - retinexformer | psnr | -0.6017 | 0.380 | [-1.4995, +0.3383] |
| ours_v2_sdsd_knobs - retinexformer | rgb_ssim | +0.0058 | 0.620 | [-0.0180, +0.0285] |
| ours_v2_sdsd_knobs - snr_aware | psnr | +0.0939 | 0.400 | [-1.1932, +1.4653] |
| ours_v2_sdsd_knobs - snr_aware | rgb_ssim | +0.0016 | 0.560 | [-0.0222, +0.0248] |
| ours_v2_sdsd_knobs - promptir | psnr | +7.4235 | 0.980 | [+6.1390, +8.7136] |
| ours_v2_sdsd_knobs - promptir | rgb_ssim | +0.3056 | 1.000 | [+0.2603, +0.3512] |
| ours_v2_sdsd_knobs - promptir_dctta | psnr | +7.4321 | 0.980 | [+6.1475, +8.7225] |
| ours_v2_sdsd_knobs - promptir_dctta | rgb_ssim | +0.3073 | 1.000 | [+0.2617, +0.3532] |
| ours_v2_sdsd_knobs - mr_illuminate | psnr | -1.0726 | 0.460 | [-2.1194, -0.0035] |
| ours_v2_sdsd_knobs - mr_illuminate | rgb_ssim | -0.0043 | 0.600 | [-0.0292, +0.0188] |
| ours_v2_sdsd_knobs - quadprior | psnr | -0.3750 | 0.460 | [-1.3706, +0.6410] |
| ours_v2_sdsd_knobs - quadprior | rgb_ssim | -0.0496 | 0.340 | [-0.0696, -0.0301] |
| ours_v2_sdsd_knobs - retinexformer_plus_D | psnr | -0.6623 | 0.380 | [-1.5534, +0.2746] |
| ours_v2_sdsd_knobs - retinexformer_plus_D | rgb_ssim | -0.0268 | 0.360 | [-0.0477, -0.0070] |
| ours_v2_sdsd_knobs - snr_aware_plus_D | psnr | -0.0353 | 0.400 | [-1.3358, +1.3425] |
| ours_v2_sdsd_knobs - snr_aware_plus_D | rgb_ssim | -0.0254 | 0.360 | [-0.0472, -0.0039] |
| ours_v2_sdsd_knobs - promptir_plus_D | psnr | +7.4685 | 0.980 | [+6.1870, +8.7564] |
| ours_v2_sdsd_knobs - promptir_plus_D | rgb_ssim | +0.3110 | 1.000 | [+0.2646, +0.3580] |
| ours_v2_sdsd_knobs - promptir_dctta_plus_D | psnr | +7.4771 | 0.980 | [+6.1951, +8.7648] |
| ours_v2_sdsd_knobs - promptir_dctta_plus_D | rgb_ssim | +0.3126 | 1.000 | [+0.2659, +0.3600] |
| ours_v2_sdsd_knobs - mr_illuminate_plus_D | psnr | -1.2163 | 0.460 | [-2.2645, -0.1448] |
| ours_v2_sdsd_knobs - mr_illuminate_plus_D | rgb_ssim | -0.0502 | 0.260 | [-0.0697, -0.0320] |
| ours_v2_sdsd_knobs - quadprior_plus_D | psnr | -0.3721 | 0.460 | [-1.3601, +0.6430] |
| ours_v2_sdsd_knobs - quadprior_plus_D | rgb_ssim | -0.0545 | 0.260 | [-0.0736, -0.0359] |
| ours_v2_sdsd_knobs - ours_ttt | psnr | +1.0980 | 0.760 | [+0.4938, +1.8070] |
| ours_v2_sdsd_knobs - ours_ttt | rgb_ssim | +0.0809 | 0.900 | [+0.0651, +0.0965] |
| ours_v2_sdsd_knobs - ours_ttt_sdsd_knobs | psnr | +0.2618 | 0.840 | [+0.1789, +0.3548] |
| ours_v2_sdsd_knobs - ours_ttt_sdsd_knobs | rgb_ssim | +0.0711 | 0.900 | [+0.0559, +0.0860] |
| ours_ttt_sdsd_knobs - ours_ttt | psnr | +0.8363 | 0.760 | [+0.2406, +1.5414] |
| ours_ttt_sdsd_knobs - ours_ttt | rgb_ssim | +0.0098 | 0.480 | [+0.0015, +0.0188] |
