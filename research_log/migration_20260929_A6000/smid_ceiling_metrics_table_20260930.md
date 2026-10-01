# Ours ceiling metrics: SMID (N=1470, G=49) — ceiling reference: Ours knobs and kappa selected on this target's test GT (not held-out)

| Row | Mean PSNR | Median PSNR | Mean RGB-SSIM |
|---|---|---|---|
| retinexformer | 16.2141 | 15.8777 | 0.5921 |
| snr_aware | 13.8494 | 13.4081 | 0.5235 |
| promptir | 13.7448 | 13.9775 | 0.4194 |
| promptir_dctta | 3.5169 | 3.3537 | 0.1757 |
| mr_illuminate | 16.6179 | 16.3702 | 0.6240 |
| quadprior | 15.6503 | 15.7193 | 0.6060 |
| ours_step0 | 13.3824 | 12.8718 | 0.4245 |
| ours_ttt | 15.0232 | 15.2568 | 0.5824 |
| ours_ttt_sdsd_knobs | 14.3613 | 14.7536 | 0.5736 |
| ours_v2 | 15.0831 | 15.3523 | 0.5810 |
| ours_v2_sdsd_knobs | 14.4172 | 14.7964 | 0.5715 |
| retinexformer_plus_D | 16.1203 | 15.6343 | 0.5562 |
| snr_aware_plus_D | 13.8533 | 13.3883 | 0.5119 |
| promptir_plus_D | 13.7235 | 13.9044 | 0.4178 |
| promptir_dctta_plus_D | 3.5541 | 3.3932 | 0.1762 |
| mr_illuminate_plus_D | 16.6228 | 16.3204 | 0.6189 |
| quadprior_plus_D | 15.6674 | 15.7442 | 0.5915 |
| ours_step0_plus_D | 13.3316 | 12.8148 | 0.4132 |
| ours_ttt_target_tuned (tuned on test GT) | 18.1099 | 18.8286 | 0.6407 |
| ours_v2_ceiling (tuned on test GT) | 18.1253 | 18.9012 | 0.6479 |

| Comparison | Metric | Mean Δ | Win fraction | 95% cluster CI |
|---|---|---|---|---|
| ours_v2_ceiling - retinexformer | psnr | +1.9112 | 0.675 | [+0.9813, +2.9313] |
| ours_v2_ceiling - retinexformer | rgb_ssim | +0.0558 | 0.725 | [+0.0336, +0.0795] |
| ours_v2_ceiling - snr_aware | psnr | +4.2759 | 0.940 | [+3.2707, +5.3372] |
| ours_v2_ceiling - snr_aware | rgb_ssim | +0.1243 | 0.965 | [+0.0971, +0.1534] |
| ours_v2_ceiling - promptir | psnr | +4.3805 | 0.978 | [+3.7129, +5.0221] |
| ours_v2_ceiling - promptir | rgb_ssim | +0.2285 | 0.979 | [+0.1954, +0.2606] |
| ours_v2_ceiling - promptir_dctta | psnr | +14.6084 | 1.000 | [+13.4731, +15.6833] |
| ours_v2_ceiling - promptir_dctta | rgb_ssim | +0.4722 | 1.000 | [+0.4418, +0.5008] |
| ours_v2_ceiling - mr_illuminate | psnr | +1.5074 | 0.537 | [-0.0086, +2.9996] |
| ours_v2_ceiling - mr_illuminate | rgb_ssim | +0.0238 | 0.529 | [-0.0082, +0.0578] |
| ours_v2_ceiling - quadprior | psnr | +2.4750 | 0.705 | [+1.3256, +3.6367] |
| ours_v2_ceiling - quadprior | rgb_ssim | +0.0419 | 0.723 | [+0.0172, +0.0675] |
| ours_v2_ceiling - ours_step0 | psnr | +4.7429 | 0.959 | [+4.0252, +5.4006] |
| ours_v2_ceiling - ours_step0 | rgb_ssim | +0.2234 | 0.954 | [+0.1881, +0.2580] |
| ours_v2_ceiling - ours_ttt | psnr | +3.1021 | 0.805 | [+2.2666, +3.9338] |
| ours_v2_ceiling - ours_ttt | rgb_ssim | +0.0655 | 0.857 | [+0.0468, +0.0844] |
| ours_v2_ceiling - ours_ttt_sdsd_knobs | psnr | +3.7640 | 0.833 | [+2.7518, +4.7392] |
| ours_v2_ceiling - ours_ttt_sdsd_knobs | rgb_ssim | +0.0742 | 0.839 | [+0.0508, +0.0966] |
| ours_v2_ceiling - ours_v2 | psnr | +3.0422 | 0.804 | [+2.2165, +3.8629] |
| ours_v2_ceiling - ours_v2 | rgb_ssim | +0.0669 | 0.875 | [+0.0506, +0.0831] |
| ours_v2_ceiling - ours_v2_sdsd_knobs | psnr | +3.7081 | 0.828 | [+2.7065, +4.6770] |
| ours_v2_ceiling - ours_v2_sdsd_knobs | rgb_ssim | +0.0764 | 0.852 | [+0.0562, +0.0958] |
| ours_v2_ceiling - retinexformer_plus_D | psnr | +2.0050 | 0.696 | [+1.0892, +3.0088] |
| ours_v2_ceiling - retinexformer_plus_D | rgb_ssim | +0.0916 | 0.879 | [+0.0687, +0.1158] |
| ours_v2_ceiling - snr_aware_plus_D | psnr | +4.2720 | 0.944 | [+3.2699, +5.3301] |
| ours_v2_ceiling - snr_aware_plus_D | rgb_ssim | +0.1360 | 0.955 | [+0.1083, +0.1650] |
| ours_v2_ceiling - promptir_plus_D | psnr | +4.4018 | 0.976 | [+3.7303, +5.0513] |
| ours_v2_ceiling - promptir_plus_D | rgb_ssim | +0.2300 | 0.977 | [+0.1946, +0.2648] |
| ours_v2_ceiling - promptir_dctta_plus_D | psnr | +14.5712 | 1.000 | [+13.4371, +15.6453] |
| ours_v2_ceiling - promptir_dctta_plus_D | rgb_ssim | +0.4717 | 1.000 | [+0.4414, +0.5003] |
| ours_v2_ceiling - mr_illuminate_plus_D | psnr | +1.5025 | 0.539 | [-0.0017, +2.9811] |
| ours_v2_ceiling - mr_illuminate_plus_D | rgb_ssim | +0.0289 | 0.568 | [-0.0011, +0.0604] |
| ours_v2_ceiling - quadprior_plus_D | psnr | +2.4579 | 0.705 | [+1.3128, +3.6092] |
| ours_v2_ceiling - quadprior_plus_D | rgb_ssim | +0.0564 | 0.776 | [+0.0328, +0.0808] |
| ours_v2_ceiling - ours_step0_plus_D | psnr | +4.7937 | 0.959 | [+4.0703, +5.4551] |
| ours_v2_ceiling - ours_step0_plus_D | rgb_ssim | +0.2347 | 0.954 | [+0.1979, +0.2706] |
| ours_v2_ceiling - ours_ttt_target_tuned | psnr | +0.0154 | 0.548 | [-0.0032, +0.0337] |
| ours_v2_ceiling - ours_ttt_target_tuned | rgb_ssim | +0.0072 | 0.649 | [+0.0013, +0.0128] |
