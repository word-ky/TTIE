# T074-C metrics: SDSD_indoor (N=180, G=6, low-power: CIs unreliable)

Gate receipt `a21eb87d10c36fce75988b4414bd8c74fe687b6fe4c8965c47b8c5580eb70360`. Outputs clipped to [0,1]; frozen T071-A metrics.

| Row | N | Mean PSNR | Median PSNR | Mean RGB-SSIM | No-active abstentions |
|---|---|---|---|---|---|
| retinexformer | 180 | 19.1301 | 18.7717 | 0.7874 | N/A |
| snr_aware | 180 | 18.0519 | 18.4885 | 0.7695 | N/A |
| promptir | 180 | 18.3279 | 18.8477 | 0.7644 | N/A |
| promptir_dctta | 180 | 18.6923 | 18.9790 | 0.7741 | N/A |
| mr_illuminate | 180 | 17.7976 | 17.8000 | 0.7867 | N/A |
| quadprior | 180 | 17.5352 | 17.2112 | 0.7895 | N/A |
| ours_step0 | 180 | 7.3951 | 7.1149 | 0.3189 | N/A |
| ours_ttt | 180 | 18.2385 | 18.0246 | 0.6408 | 0 |

| Comparison | Metric | Mean Δ | Median Δ | Win fraction | 95% cluster CI | Families |
|---|---|---|---|---|---|---|
| ours_ttt - retinexformer | psnr | -0.8916 | -0.9029 | 0.328 (59/180) | [-2.6940, +0.9786] | headline |
| ours_ttt - retinexformer | rgb_ssim | -0.1466 | -0.1445 | 0.000 (0/180) | [-0.1758, -0.1208] | headline |
| ours_ttt - snr_aware | psnr | +0.1866 | -1.1752 | 0.494 (89/180) | [-2.6897, +3.4942] | headline |
| ours_ttt - snr_aware | rgb_ssim | -0.1287 | -0.1387 | 0.006 (1/180) | [-0.1689, -0.0860] | headline |
| ours_ttt - promptir | psnr | -0.0894 | +0.2064 | 0.533 (96/180) | [-1.3384, +1.0946] | headline |
| ours_ttt - promptir | rgb_ssim | -0.1236 | -0.1200 | 0.000 (0/180) | [-0.1564, -0.0901] | headline |
| ours_ttt - promptir_dctta | psnr | -0.4538 | -0.5232 | 0.428 (77/180) | [-1.8802, +1.0796] | headline |
| ours_ttt - promptir_dctta | rgb_ssim | -0.1333 | -0.1176 | 0.000 (0/180) | [-0.1644, -0.1054] | headline |
| ours_ttt - mr_illuminate | psnr | +0.4409 | +0.7562 | 0.650 (117/180) | [-0.5879, +1.4495] | headline |
| ours_ttt - mr_illuminate | rgb_ssim | -0.1458 | -0.1415 | 0.000 (0/180) | [-0.1714, -0.1229] | headline |
| ours_ttt - quadprior | psnr | +0.7033 | +0.9625 | 0.861 (155/180) | [+0.4415, +0.9595] | headline |
| ours_ttt - quadprior | rgb_ssim | -0.1487 | -0.1427 | 0.000 (0/180) | [-0.1612, -0.1388] | headline |
| ours_ttt - ours_step0 | psnr | +10.8434 | +10.6383 | 1.000 (180/180) | [+8.9236, +12.6957] | headline, adaptation_gain |
| ours_ttt - ours_step0 | rgb_ssim | +0.3220 | +0.3133 | 1.000 (180/180) | [+0.2742, +0.3773] | headline, adaptation_gain |
| promptir_dctta - promptir | psnr | +0.3644 | +0.2886 | 0.633 (114/180) | [-0.5331, +1.2310] | adaptation_gain |
| promptir_dctta - promptir | rgb_ssim | +0.0097 | -0.0008 | 0.456 (82/180) | [-0.0200, +0.0373] | adaptation_gain |
