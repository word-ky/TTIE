# T075-A v2 metrics: SMID (N=1470, G=49)

| Row | Mean PSNR | Mean RGB-SSIM |
|---|---|---|
| retinexformer | 16.2141 | 0.5921 |
| snr_aware | 13.8494 | 0.5235 |
| promptir | 13.7448 | 0.4194 |
| promptir_dctta | 3.5169 | 0.1757 |
| mr_illuminate | 16.6179 | 0.6240 |
| quadprior | 15.6503 | 0.6060 |
| ours_step0 | 13.3824 | 0.4245 |
| ours_ttt | 15.0232 | 0.5824 |
| ours_ttt_sdsd_knobs | 14.3613 | 0.5736 |
| ours_v2 | 15.0831 | 0.5810 |
| ours_v2_sdsd_knobs | 14.4172 | 0.5715 |
| retinexformer_plus_D | 16.1203 | 0.5562 |
| snr_aware_plus_D | 13.8533 | 0.5119 |
| promptir_plus_D | 13.7235 | 0.4178 |
| promptir_dctta_plus_D | 3.5541 | 0.1762 |
| mr_illuminate_plus_D | 16.6228 | 0.6189 |
| quadprior_plus_D | 15.6674 | 0.5915 |
| ours_step0_plus_D | 13.3316 | 0.4132 |

| Comparison | Metric | Mean Δ | Win fraction | 95% cluster CI |
|---|---|---|---|---|
| ours_v2 - retinexformer | psnr | -1.1310 | 0.368 | [-1.9340, -0.3299] |
| ours_v2 - retinexformer | rgb_ssim | -0.0111 | 0.392 | [-0.0338, +0.0120] |
| ours_v2 - snr_aware | psnr | +1.2337 | 0.659 | [+0.2563, +2.1832] |
| ours_v2 - snr_aware | rgb_ssim | +0.0575 | 0.681 | [+0.0277, +0.0877] |
| ours_v2 - promptir | psnr | +1.3383 | 0.656 | [+0.3047, +2.3827] |
| ours_v2 - promptir | rgb_ssim | +0.1616 | 0.825 | [+0.1228, +0.2007] |
| ours_v2 - promptir_dctta | psnr | +11.5662 | 1.000 | [+10.8848, +12.2266] |
| ours_v2 - promptir_dctta | rgb_ssim | +0.4053 | 1.000 | [+0.3759, +0.4333] |
| ours_v2 - mr_illuminate | psnr | -1.5348 | 0.353 | [-2.5280, -0.5427] |
| ours_v2 - mr_illuminate | rgb_ssim | -0.0430 | 0.236 | [-0.0677, -0.0174] |
| ours_v2 - quadprior | psnr | -0.5672 | 0.438 | [-1.2548, +0.1103] |
| ours_v2 - quadprior | rgb_ssim | -0.0250 | 0.316 | [-0.0425, -0.0078] |
| ours_v2 - retinexformer_plus_D | psnr | -1.0372 | 0.370 | [-1.8364, -0.2352] |
| ours_v2 - retinexformer_plus_D | rgb_ssim | +0.0248 | 0.522 | [+0.0028, +0.0483] |
| ours_v2 - snr_aware_plus_D | psnr | +1.2298 | 0.659 | [+0.2484, +2.1829] |
| ours_v2 - snr_aware_plus_D | rgb_ssim | +0.0691 | 0.727 | [+0.0405, +0.0984] |
| ours_v2 - promptir_plus_D | psnr | +1.3596 | 0.656 | [+0.3124, +2.4124] |
| ours_v2 - promptir_plus_D | rgb_ssim | +0.1632 | 0.796 | [+0.1220, +0.2043] |
| ours_v2 - promptir_dctta_plus_D | psnr | +11.5290 | 1.000 | [+10.8481, +12.1892] |
| ours_v2 - promptir_dctta_plus_D | rgb_ssim | +0.4048 | 1.000 | [+0.3755, +0.4328] |
| ours_v2 - mr_illuminate_plus_D | psnr | -1.5397 | 0.350 | [-2.5184, -0.5565] |
| ours_v2 - mr_illuminate_plus_D | rgb_ssim | -0.0379 | 0.239 | [-0.0588, -0.0163] |
| ours_v2 - quadprior_plus_D | psnr | -0.5843 | 0.437 | [-1.2682, +0.0909] |
| ours_v2 - quadprior_plus_D | rgb_ssim | -0.0105 | 0.444 | [-0.0244, +0.0033] |
| ours_v2 - ours_ttt | psnr | +0.0599 | 0.797 | [+0.0417, +0.0785] |
| ours_v2 - ours_ttt | rgb_ssim | -0.0014 | 0.468 | [-0.0136, +0.0104] |
| ours_v2_sdsd_knobs - retinexformer | psnr | -1.7968 | 0.265 | [-2.7308, -0.8585] |
| ours_v2_sdsd_knobs - retinexformer | rgb_ssim | -0.0206 | 0.407 | [-0.0454, +0.0043] |
| ours_v2_sdsd_knobs - snr_aware | psnr | +0.5679 | 0.601 | [-0.5111, +1.6016] |
| ours_v2_sdsd_knobs - snr_aware | rgb_ssim | +0.0480 | 0.596 | [+0.0158, +0.0801] |
| ours_v2_sdsd_knobs - promptir | psnr | +0.6724 | 0.629 | [-0.4639, +1.7721] |
| ours_v2_sdsd_knobs - promptir | rgb_ssim | +0.1521 | 0.772 | [+0.1129, +0.1910] |
| ours_v2_sdsd_knobs - promptir_dctta | psnr | +10.9003 | 1.000 | [+10.2720, +11.5122] |
| ours_v2_sdsd_knobs - promptir_dctta | rgb_ssim | +0.3958 | 1.000 | [+0.3669, +0.4226] |
| ours_v2_sdsd_knobs - mr_illuminate | psnr | -2.2007 | 0.333 | [-3.2829, -1.1235] |
| ours_v2_sdsd_knobs - mr_illuminate | rgb_ssim | -0.0525 | 0.207 | [-0.0781, -0.0267] |
| ours_v2_sdsd_knobs - quadprior | psnr | -1.2331 | 0.363 | [-2.0337, -0.4622] |
| ours_v2_sdsd_knobs - quadprior | rgb_ssim | -0.0344 | 0.305 | [-0.0532, -0.0165] |
| ours_v2_sdsd_knobs - retinexformer_plus_D | psnr | -1.7031 | 0.269 | [-2.6366, -0.7701] |
| ours_v2_sdsd_knobs - retinexformer_plus_D | rgb_ssim | +0.0153 | 0.474 | [-0.0083, +0.0398] |
| ours_v2_sdsd_knobs - snr_aware_plus_D | psnr | +0.5639 | 0.602 | [-0.5167, +1.5991] |
| ours_v2_sdsd_knobs - snr_aware_plus_D | rgb_ssim | +0.0597 | 0.659 | [+0.0289, +0.0912] |
| ours_v2_sdsd_knobs - promptir_plus_D | psnr | +0.6938 | 0.631 | [-0.4508, +1.8028] |
| ours_v2_sdsd_knobs - promptir_plus_D | rgb_ssim | +0.1537 | 0.761 | [+0.1118, +0.1949] |
| ours_v2_sdsd_knobs - promptir_dctta_plus_D | psnr | +10.8631 | 1.000 | [+10.2346, +11.4752] |
| ours_v2_sdsd_knobs - promptir_dctta_plus_D | rgb_ssim | +0.3954 | 1.000 | [+0.3665, +0.4222] |
| ours_v2_sdsd_knobs - mr_illuminate_plus_D | psnr | -2.2056 | 0.324 | [-3.2772, -1.1380] |
| ours_v2_sdsd_knobs - mr_illuminate_plus_D | rgb_ssim | -0.0474 | 0.223 | [-0.0694, -0.0248] |
| ours_v2_sdsd_knobs - quadprior_plus_D | psnr | -1.2501 | 0.356 | [-2.0494, -0.4814] |
| ours_v2_sdsd_knobs - quadprior_plus_D | rgb_ssim | -0.0200 | 0.371 | [-0.0354, -0.0053] |
| ours_v2_sdsd_knobs - ours_ttt | psnr | -0.6059 | 0.161 | [-0.9161, -0.2580] |
| ours_v2_sdsd_knobs - ours_ttt | rgb_ssim | -0.0109 | 0.457 | [-0.0246, +0.0027] |
| ours_v2_sdsd_knobs - ours_ttt_sdsd_knobs | psnr | +0.0559 | 0.839 | [+0.0360, +0.0749] |
| ours_v2_sdsd_knobs - ours_ttt_sdsd_knobs | rgb_ssim | -0.0021 | 0.469 | [-0.0151, +0.0105] |
| ours_ttt_sdsd_knobs - ours_ttt | psnr | -0.6618 | 0.117 | [-0.9773, -0.3045] |
| ours_ttt_sdsd_knobs - ours_ttt | rgb_ssim | -0.0087 | 0.150 | [-0.0161, -0.0001] |
