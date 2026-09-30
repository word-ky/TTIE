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
| ours_ttt_target_tuned (tuned on test GT) | 16.5740 | 16.4002 | 0.4557 |
| ours_v2_ceiling (tuned on test GT) | 16.8302 | 16.9498 | 0.5227 |

| Comparison | Metric | Mean Δ | Win fraction | 95% cluster CI |
|---|---|---|---|---|
| ours_v2_ceiling - retinexformer | psnr | -0.3652 | 0.440 | [-1.2697, +0.5641] |
| ours_v2_ceiling - retinexformer | rgb_ssim | +0.0134 | 0.600 | [-0.0112, +0.0374] |
| ours_v2_ceiling - snr_aware | psnr | +0.3304 | 0.400 | [-0.9987, +1.7239] |
| ours_v2_ceiling - snr_aware | rgb_ssim | +0.0092 | 0.580 | [-0.0147, +0.0325] |
| ours_v2_ceiling - promptir | psnr | +7.6599 | 1.000 | [+6.4989, +8.8170] |
| ours_v2_ceiling - promptir | rgb_ssim | +0.3132 | 1.000 | [+0.2693, +0.3572] |
| ours_v2_ceiling - promptir_dctta | psnr | +7.6686 | 1.000 | [+6.5077, +8.8260] |
| ours_v2_ceiling - promptir_dctta | rgb_ssim | +0.3149 | 1.000 | [+0.2706, +0.3595] |
| ours_v2_ceiling - mr_illuminate | psnr | -0.8361 | 0.400 | [-1.7317, +0.0606] |
| ours_v2_ceiling - mr_illuminate | rgb_ssim | +0.0034 | 0.600 | [-0.0201, +0.0255] |
| ours_v2_ceiling - quadprior | psnr | -0.1385 | 0.520 | [-1.0964, +0.8467] |
| ours_v2_ceiling - quadprior | rgb_ssim | -0.0419 | 0.400 | [-0.0614, -0.0233] |
| ours_v2_ceiling - ours_step0 | psnr | +7.6714 | 1.000 | [+6.5063, +8.8309] |
| ours_v2_ceiling - ours_step0 | rgb_ssim | +0.3209 | 1.000 | [+0.2767, +0.3655] |
| ours_v2_ceiling - ours_ttt | psnr | +1.3345 | 0.820 | [+0.7808, +1.9572] |
| ours_v2_ceiling - ours_ttt | rgb_ssim | +0.0885 | 0.920 | [+0.0719, +0.1048] |
| ours_v2_ceiling - ours_ttt_sdsd_knobs | psnr | +0.4982 | 0.640 | [+0.0531, +0.9926] |
| ours_v2_ceiling - ours_ttt_sdsd_knobs | rgb_ssim | +0.0787 | 0.900 | [+0.0623, +0.0949] |
| ours_v2_ceiling - ours_v2 | psnr | +1.1688 | 0.780 | [+0.5977, +1.8139] |
| ours_v2_ceiling - ours_v2 | rgb_ssim | +0.0287 | 0.740 | [+0.0169, +0.0416] |
| ours_v2_ceiling - ours_v2_sdsd_knobs | psnr | +0.2365 | 0.480 | [-0.2029, +0.7314] |
| ours_v2_ceiling - ours_v2_sdsd_knobs | rgb_ssim | +0.0076 | 0.600 | [+0.0024, +0.0137] |
| ours_v2_ceiling - retinexformer_plus_D | psnr | -0.4258 | 0.400 | [-1.3218, +0.4998] |
| ours_v2_ceiling - retinexformer_plus_D | rgb_ssim | -0.0192 | 0.400 | [-0.0404, +0.0012] |
| ours_v2_ceiling - snr_aware_plus_D | psnr | +0.2012 | 0.380 | [-1.1332, +1.6046] |
| ours_v2_ceiling - snr_aware_plus_D | rgb_ssim | -0.0178 | 0.380 | [-0.0390, +0.0031] |
| ours_v2_ceiling - promptir_plus_D | psnr | +7.7050 | 1.000 | [+6.5479, +8.8591] |
| ours_v2_ceiling - promptir_plus_D | rgb_ssim | +0.3186 | 1.000 | [+0.2735, +0.3641] |
| ours_v2_ceiling - promptir_dctta_plus_D | psnr | +7.7135 | 1.000 | [+6.5557, +8.8681] |
| ours_v2_ceiling - promptir_dctta_plus_D | rgb_ssim | +0.3203 | 1.000 | [+0.2747, +0.3663] |
| ours_v2_ceiling - mr_illuminate_plus_D | psnr | -0.9798 | 0.380 | [-1.8762, -0.0820] |
| ours_v2_ceiling - mr_illuminate_plus_D | rgb_ssim | -0.0426 | 0.240 | [-0.0612, -0.0255] |
| ours_v2_ceiling - quadprior_plus_D | psnr | -0.1356 | 0.540 | [-1.0902, +0.8435] |
| ours_v2_ceiling - quadprior_plus_D | rgb_ssim | -0.0469 | 0.320 | [-0.0656, -0.0292] |
| ours_v2_ceiling - ours_step0_plus_D | psnr | +7.7185 | 1.000 | [+6.5557, +8.8747] |
| ours_v2_ceiling - ours_step0_plus_D | rgb_ssim | +0.3261 | 1.000 | [+0.2804, +0.3725] |
| ours_v2_ceiling - ours_ttt_target_tuned | psnr | +0.2562 | 0.760 | [+0.1692, +0.3563] |
| ours_v2_ceiling - ours_ttt_target_tuned | rgb_ssim | +0.0670 | 0.900 | [+0.0498, +0.0834] |
