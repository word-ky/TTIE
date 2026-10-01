# T075-A v2 metrics: SID (N=598, G=50)

| Row | Mean PSNR | Mean RGB-SSIM |
|---|---|---|
| retinexformer | 14.7816 | 0.3154 |
| snr_aware | 14.9077 | 0.2994 |
| promptir | 13.9788 | 0.3270 |
| promptir_dctta | 3.0333 | 0.2707 |
| mr_illuminate | 15.7460 | 0.4000 |
| quadprior | 15.8107 | 0.5108 |
| ours_step0 | 11.5039 | 0.1505 |
| ours_ttt | 13.1121 | 0.1516 |
| ours_ttt_sdsd_knobs | 13.3526 | 0.1521 |
| ours_v2 | 15.1166 | 0.4264 |
| ours_v2_sdsd_knobs | 15.5591 | 0.4382 |
| retinexformer_plus_D | 14.8584 | 0.4138 |
| snr_aware_plus_D | 15.2831 | 0.4495 |
| promptir_plus_D | 13.9903 | 0.3734 |
| promptir_dctta_plus_D | 3.0797 | 0.2713 |
| mr_illuminate_plus_D | 15.7570 | 0.4789 |
| quadprior_plus_D | 15.7231 | 0.5162 |
| ours_step0_plus_D | 11.5833 | 0.2585 |

| Comparison | Metric | Mean Δ | Win fraction | 95% cluster CI |
|---|---|---|---|---|
| ours_v2 - retinexformer | psnr | +0.3350 | 0.662 | [-0.3472, +0.9557] |
| ours_v2 - retinexformer | rgb_ssim | +0.1110 | 0.751 | [+0.0619, +0.1566] |
| ours_v2 - snr_aware | psnr | +0.2089 | 0.500 | [-1.0043, +1.4205] |
| ours_v2 - snr_aware | rgb_ssim | +0.1270 | 0.751 | [+0.0701, +0.1801] |
| ours_v2 - promptir | psnr | +1.1378 | 0.692 | [-0.2064, +2.3586] |
| ours_v2 - promptir | rgb_ssim | +0.0994 | 0.771 | [+0.0466, +0.1473] |
| ours_v2 - promptir_dctta | psnr | +12.0833 | 1.000 | [+11.0811, +13.1146] |
| ours_v2 - promptir_dctta | rgb_ssim | +0.1557 | 0.871 | [+0.1181, +0.1949] |
| ours_v2 - mr_illuminate | psnr | -0.6294 | 0.413 | [-1.7813, +0.5224] |
| ours_v2 - mr_illuminate | rgb_ssim | +0.0264 | 0.644 | [-0.0243, +0.0741] |
| ours_v2 - quadprior | psnr | -0.6941 | 0.500 | [-1.4183, +0.0050] |
| ours_v2 - quadprior | rgb_ssim | -0.0844 | 0.303 | [-0.1250, -0.0452] |
| ours_v2 - retinexformer_plus_D | psnr | +0.2582 | 0.639 | [-0.4036, +0.8667] |
| ours_v2 - retinexformer_plus_D | rgb_ssim | +0.0126 | 0.726 | [-0.0211, +0.0427] |
| ours_v2 - snr_aware_plus_D | psnr | -0.1665 | 0.440 | [-1.3683, +1.0348] |
| ours_v2 - snr_aware_plus_D | rgb_ssim | -0.0231 | 0.396 | [-0.0657, +0.0169] |
| ours_v2 - promptir_plus_D | psnr | +1.1263 | 0.691 | [-0.2503, +2.3689] |
| ours_v2 - promptir_plus_D | rgb_ssim | +0.0530 | 0.716 | [-0.0044, +0.1038] |
| ours_v2 - promptir_dctta_plus_D | psnr | +12.0369 | 1.000 | [+11.0334, +13.0687] |
| ours_v2 - promptir_dctta_plus_D | rgb_ssim | +0.1551 | 0.870 | [+0.1174, +0.1943] |
| ours_v2 - mr_illuminate_plus_D | psnr | -0.6404 | 0.410 | [-1.7612, +0.4934] |
| ours_v2 - mr_illuminate_plus_D | rgb_ssim | -0.0525 | 0.242 | [-0.0905, -0.0140] |
| ours_v2 - quadprior_plus_D | psnr | -0.6065 | 0.503 | [-1.3001, +0.0611] |
| ours_v2 - quadprior_plus_D | rgb_ssim | -0.0898 | 0.176 | [-0.1249, -0.0565] |
| ours_v2 - ours_ttt | psnr | +2.0045 | 0.796 | [+1.4744, +2.5174] |
| ours_v2 - ours_ttt | rgb_ssim | +0.2748 | 0.977 | [+0.2289, +0.3183] |
| ours_v2_sdsd_knobs - retinexformer | psnr | +0.7775 | 0.694 | [+0.0433, +1.4601] |
| ours_v2_sdsd_knobs - retinexformer | rgb_ssim | +0.1229 | 0.759 | [+0.0735, +0.1679] |
| ours_v2_sdsd_knobs - snr_aware | psnr | +0.6515 | 0.512 | [-0.5585, +1.8583] |
| ours_v2_sdsd_knobs - snr_aware | rgb_ssim | +0.1388 | 0.776 | [+0.0841, +0.1895] |
| ours_v2_sdsd_knobs - promptir | psnr | +1.5803 | 0.711 | [+0.2216, +2.8112] |
| ours_v2_sdsd_knobs - promptir | rgb_ssim | +0.1112 | 0.794 | [+0.0620, +0.1560] |
| ours_v2_sdsd_knobs - promptir_dctta | psnr | +12.5258 | 1.000 | [+11.5756, +13.4953] |
| ours_v2_sdsd_knobs - promptir_dctta | rgb_ssim | +0.1676 | 0.905 | [+0.1325, +0.2045] |
| ours_v2_sdsd_knobs - mr_illuminate | psnr | -0.1868 | 0.420 | [-1.2854, +0.9295] |
| ours_v2_sdsd_knobs - mr_illuminate | rgb_ssim | +0.0383 | 0.635 | [-0.0072, +0.0818] |
| ours_v2_sdsd_knobs - quadprior | psnr | -0.2516 | 0.530 | [-0.9335, +0.4313] |
| ours_v2_sdsd_knobs - quadprior | rgb_ssim | -0.0726 | 0.313 | [-0.1103, -0.0364] |
| ours_v2_sdsd_knobs - retinexformer_plus_D | psnr | +0.7008 | 0.681 | [-0.0279, +1.3874] |
| ours_v2_sdsd_knobs - retinexformer_plus_D | rgb_ssim | +0.0244 | 0.756 | [-0.0110, +0.0574] |
| ours_v2_sdsd_knobs - snr_aware_plus_D | psnr | +0.2760 | 0.463 | [-0.9308, +1.4803] |
| ours_v2_sdsd_knobs - snr_aware_plus_D | rgb_ssim | -0.0112 | 0.433 | [-0.0528, +0.0276] |
| ours_v2_sdsd_knobs - promptir_plus_D | psnr | +1.5688 | 0.711 | [+0.1636, +2.8227] |
| ours_v2_sdsd_knobs - promptir_plus_D | rgb_ssim | +0.0649 | 0.739 | [+0.0112, +0.1137] |
| ours_v2_sdsd_knobs - promptir_dctta_plus_D | psnr | +12.4794 | 1.000 | [+11.5283, +13.4515] |
| ours_v2_sdsd_knobs - promptir_dctta_plus_D | rgb_ssim | +0.1669 | 0.903 | [+0.1318, +0.2038] |
| ours_v2_sdsd_knobs - mr_illuminate_plus_D | psnr | -0.1978 | 0.423 | [-1.2739, +0.9071] |
| ours_v2_sdsd_knobs - mr_illuminate_plus_D | rgb_ssim | -0.0407 | 0.263 | [-0.0750, -0.0051] |
| ours_v2_sdsd_knobs - quadprior_plus_D | psnr | -0.1639 | 0.545 | [-0.8338, +0.4943] |
| ours_v2_sdsd_knobs - quadprior_plus_D | rgb_ssim | -0.0779 | 0.174 | [-0.1092, -0.0477] |
| ours_v2_sdsd_knobs - ours_ttt | psnr | +2.4470 | 0.878 | [+1.9645, +2.9066] |
| ours_v2_sdsd_knobs - ours_ttt | rgb_ssim | +0.2866 | 0.977 | [+0.2422, +0.3285] |
| ours_v2_sdsd_knobs - ours_ttt_sdsd_knobs | psnr | +2.2066 | 0.860 | [+1.6856, +2.7040] |
| ours_v2_sdsd_knobs - ours_ttt_sdsd_knobs | rgb_ssim | +0.2862 | 0.977 | [+0.2407, +0.3287] |
| ours_ttt_sdsd_knobs - ours_ttt | psnr | +0.2404 | 0.505 | [+0.0148, +0.5110] |
| ours_ttt_sdsd_knobs - ours_ttt | rgb_ssim | +0.0005 | 0.159 | [-0.0027, +0.0043] |
