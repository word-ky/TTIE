# T075-A v2 metrics: SDSD_indoor (N=180, G=6) — DEVELOPMENT (v2 designed on this GT)

| Row | Mean PSNR | Mean RGB-SSIM |
|---|---|---|
| retinexformer | 19.1301 | 0.7874 |
| snr_aware | 18.0519 | 0.7695 |
| promptir | 18.3279 | 0.7644 |
| promptir_dctta | 18.6923 | 0.7741 |
| mr_illuminate | 17.7976 | 0.7867 |
| quadprior | 17.5352 | 0.7895 |
| ours_step0 | 7.3951 | 0.3189 |
| ours_ttt | 18.2385 | 0.6408 |
| ours_ttt_sdsd_knobs | 19.0342 | 0.6252 |
| ours_v2 | 18.4021 | 0.8184 |
| ours_v2_sdsd_knobs | 19.3349 | 0.8296 |
| retinexformer_plus_D | 19.1823 | 0.8228 |
| snr_aware_plus_D | 18.0793 | 0.7960 |
| promptir_plus_D | 18.3274 | 0.8007 |
| promptir_dctta_plus_D | 18.7440 | 0.8214 |
| mr_illuminate_plus_D | 17.7947 | 0.8160 |
| quadprior_plus_D | 17.5122 | 0.8253 |
| ours_step0_plus_D | 7.3630 | 0.3094 |

| Comparison | Metric | Mean Δ | Win fraction | 95% cluster CI |
|---|---|---|---|---|
| ours_v2 - retinexformer | psnr | -0.7280 | 0.328 | [-2.6308, +1.2262] |
| ours_v2 - retinexformer | rgb_ssim | +0.0310 | 0.750 | [+0.0070, +0.0548] |
| ours_v2 - snr_aware | psnr | +0.3502 | 0.494 | [-2.5990, +3.6428] |
| ours_v2 - snr_aware | rgb_ssim | +0.0489 | 0.789 | [+0.0093, +0.1020] |
| ours_v2 - promptir | psnr | +0.0742 | 0.550 | [-1.1630, +1.2342] |
| ours_v2 - promptir | rgb_ssim | +0.0539 | 0.956 | [+0.0383, +0.0687] |
| ours_v2 - promptir_dctta | psnr | -0.2903 | 0.428 | [-1.7245, +1.2702] |
| ours_v2 - promptir_dctta | rgb_ssim | +0.0443 | 0.911 | [+0.0248, +0.0638] |
| ours_v2 - mr_illuminate | psnr | +0.6045 | 0.644 | [-0.4626, +1.7085] |
| ours_v2 - mr_illuminate | rgb_ssim | +0.0317 | 0.911 | [+0.0160, +0.0474] |
| ours_v2 - quadprior | psnr | +0.8669 | 0.850 | [+0.4955, +1.2325] |
| ours_v2 - quadprior | rgb_ssim | +0.0289 | 0.806 | [+0.0116, +0.0457] |
| ours_v2 - retinexformer_plus_D | psnr | -0.7803 | 0.328 | [-2.6727, +1.1623] |
| ours_v2 - retinexformer_plus_D | rgb_ssim | -0.0044 | 0.578 | [-0.0283, +0.0192] |
| ours_v2 - snr_aware_plus_D | psnr | +0.3228 | 0.494 | [-2.6160, +3.6550] |
| ours_v2 - snr_aware_plus_D | rgb_ssim | +0.0224 | 0.722 | [-0.0219, +0.0784] |
| ours_v2 - promptir_plus_D | psnr | +0.0747 | 0.556 | [-1.1784, +1.2522] |
| ours_v2 - promptir_plus_D | rgb_ssim | +0.0177 | 0.600 | [-0.0034, +0.0388] |
| ours_v2 - promptir_dctta_plus_D | psnr | -0.3420 | 0.428 | [-1.7692, +1.1883] |
| ours_v2 - promptir_dctta_plus_D | rgb_ssim | -0.0031 | 0.606 | [-0.0267, +0.0173] |
| ours_v2 - mr_illuminate_plus_D | psnr | +0.6074 | 0.650 | [-0.4446, +1.6789] |
| ours_v2 - mr_illuminate_plus_D | rgb_ssim | +0.0024 | 0.728 | [-0.0159, +0.0181] |
| ours_v2 - quadprior_plus_D | psnr | +0.8898 | 0.850 | [+0.5330, +1.2427] |
| ours_v2 - quadprior_plus_D | rgb_ssim | -0.0069 | 0.428 | [-0.0192, +0.0059] |
| ours_v2 - ours_ttt | psnr | +0.1636 | 0.811 | [+0.0372, +0.3140] |
| ours_v2 - ours_ttt | rgb_ssim | +0.1775 | 1.000 | [+0.1570, +0.1972] |
| ours_v2_sdsd_knobs - retinexformer | psnr | +0.2048 | 0.500 | [-0.6730, +1.1099] |
| ours_v2_sdsd_knobs - retinexformer | rgb_ssim | +0.0422 | 0.928 | [+0.0219, +0.0599] |
| ours_v2_sdsd_knobs - snr_aware | psnr | +1.2830 | 0.578 | [-0.6951, +3.8671] |
| ours_v2_sdsd_knobs - snr_aware | rgb_ssim | +0.0601 | 0.983 | [+0.0251, +0.1081] |
| ours_v2_sdsd_knobs - promptir | psnr | +1.0070 | 0.589 | [-0.1984, +2.2859] |
| ours_v2_sdsd_knobs - promptir | rgb_ssim | +0.0651 | 1.000 | [+0.0412, +0.0906] |
| ours_v2_sdsd_knobs - promptir_dctta | psnr | +0.6425 | 0.700 | [-0.1930, +1.3811] |
| ours_v2_sdsd_knobs - promptir_dctta | rgb_ssim | +0.0555 | 1.000 | [+0.0432, +0.0685] |
| ours_v2_sdsd_knobs - mr_illuminate | psnr | +1.5373 | 0.839 | [+0.7059, +2.1941] |
| ours_v2_sdsd_knobs - mr_illuminate | rgb_ssim | +0.0429 | 1.000 | [+0.0294, +0.0565] |
| ours_v2_sdsd_knobs - quadprior | psnr | +1.7997 | 0.839 | [+0.6608, +2.8586] |
| ours_v2_sdsd_knobs - quadprior | rgb_ssim | +0.0401 | 0.844 | [+0.0159, +0.0643] |
| ours_v2_sdsd_knobs - retinexformer_plus_D | psnr | +0.1525 | 0.500 | [-0.7023, +1.0433] |
| ours_v2_sdsd_knobs - retinexformer_plus_D | rgb_ssim | +0.0067 | 0.611 | [-0.0087, +0.0221] |
| ours_v2_sdsd_knobs - snr_aware_plus_D | psnr | +1.2556 | 0.544 | [-0.7161, +3.8784] |
| ours_v2_sdsd_knobs - snr_aware_plus_D | rgb_ssim | +0.0335 | 0.728 | [-0.0006, +0.0837] |
| ours_v2_sdsd_knobs - promptir_plus_D | psnr | +1.0075 | 0.578 | [-0.2191, +2.3189] |
| ours_v2_sdsd_knobs - promptir_plus_D | rgb_ssim | +0.0288 | 0.667 | [+0.0013, +0.0558] |
| ours_v2_sdsd_knobs - promptir_dctta_plus_D | psnr | +0.5908 | 0.678 | [-0.2534, +1.3432] |
| ours_v2_sdsd_knobs - promptir_dctta_plus_D | rgb_ssim | +0.0081 | 0.722 | [-0.0017, +0.0180] |
| ours_v2_sdsd_knobs - mr_illuminate_plus_D | psnr | +1.5402 | 0.839 | [+0.7188, +2.1984] |
| ours_v2_sdsd_knobs - mr_illuminate_plus_D | rgb_ssim | +0.0136 | 0.794 | [+0.0044, +0.0223] |
| ours_v2_sdsd_knobs - quadprior_plus_D | psnr | +1.8227 | 0.839 | [+0.6816, +2.8738] |
| ours_v2_sdsd_knobs - quadprior_plus_D | rgb_ssim | +0.0042 | 0.511 | [-0.0128, +0.0209] |
| ours_v2_sdsd_knobs - ours_ttt | psnr | +1.0964 | 0.733 | [-0.1324, +2.2468] |
| ours_v2_sdsd_knobs - ours_ttt | rgb_ssim | +0.1887 | 1.000 | [+0.1582, +0.2235] |
| ours_v2_sdsd_knobs - ours_ttt_sdsd_knobs | psnr | +0.3007 | 0.878 | [+0.1240, +0.4877] |
| ours_v2_sdsd_knobs - ours_ttt_sdsd_knobs | rgb_ssim | +0.2043 | 1.000 | [+0.1758, +0.2306] |
| ours_ttt_sdsd_knobs - ours_ttt | psnr | +0.7957 | 0.572 | [-0.3606, +1.9423] |
| ours_ttt_sdsd_knobs - ours_ttt | rgb_ssim | -0.0156 | 0.089 | [-0.0311, +0.0003] |
