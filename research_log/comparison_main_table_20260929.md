# Verified comparison table — 2026-09-29 (ceiling rows pending)

Each cell is mean PSNR (dB) / mean RGB-SSIM, both higher is better. All rows use their full stated cohort and the frozen T071-A metric implementation. SDSD-indoor (180 frames) is the v2 **development** dataset; LSRW (50 images), SMID (1470 frames), and SID (598 images) are post-freeze comparisons. Ordinary target runs used no test GT for adaptation/selection; references were opened only after each target's required output gate. `+D` applies the same denoising post-process to the named frozen source output. `sdsd_knobs` parameters were developed on SDSD and transferred unchanged to the other targets.

| Method ID | SDSD-indoor (dev) | LSRW | SMID | SID |
|---|---:|---:|---:|---:|
| retinexformer | 19.1301 / 0.7874 | 17.1954 / 0.5093 | 16.2141 / 0.5921 | 14.7816 / 0.3154 |
| snr_aware | 18.0519 / 0.7695 | 16.4998 / 0.5135 | 13.8494 / 0.5235 | 14.9077 / 0.2994 |
| promptir | 18.3279 / 0.7644 | 9.1702 / 0.2095 | 13.7448 / 0.4194 | 13.9788 / 0.3270 |
| promptir_dctta | 18.6923 / 0.7741 | 9.1616 / 0.2078 | 3.5169 / 0.1757 | 3.0333 / 0.2707 |
| mr_illuminate | 17.7976 / 0.7867 | 17.6663 / 0.5194 | 16.6179 / 0.6240 | 15.7460 / 0.4000 |
| quadprior | 17.5352 / 0.7895 | 16.9687 / 0.5646 | 15.6503 / 0.6060 | 15.8107 / 0.5108 |
| ours_step0 | 7.3951 / 0.3189 | 9.1588 / 0.2018 | 13.3824 / 0.4245 | 11.5039 / 0.1505 |
| ours_ttt | 18.2385 / 0.6408 | 15.4957 / 0.4342 | 15.0232 / 0.5824 | 13.1121 / 0.1516 |
| ours_ttt_sdsd_knobs | 19.0342 / 0.6252 | 16.3319 / 0.4440 | 14.3613 / 0.5736 | 13.3526 / 0.1521 |
| ours_v2 | 18.4021 / 0.8184 | 15.6614 / 0.4941 | 15.0831 / 0.5810 | 15.1166 / 0.4264 |
| ours_v2_sdsd_knobs | 19.3349 / 0.8296 | 16.5937 / 0.5151 | 14.4172 / 0.5715 | 15.5591 / 0.4382 |
| retinexformer_plus_D | 19.1823 / 0.8228 | 17.2559 / 0.5419 | 16.1203 / 0.5562 | 14.8584 / 0.4138 |
| snr_aware_plus_D | 18.0793 / 0.7960 | 16.6289 / 0.5405 | 13.8533 / 0.5119 | 15.2831 / 0.4495 |
| promptir_plus_D | 18.3274 / 0.8007 | 9.1252 / 0.2041 | 13.7235 / 0.4178 | 13.9903 / 0.3734 |
| promptir_dctta_plus_D | 18.7440 / 0.8214 | 9.1166 / 0.2025 | 3.5541 / 0.1762 | 3.0797 / 0.2713 |
| mr_illuminate_plus_D | 17.7947 / 0.8160 | 17.8100 / 0.5653 | 16.6228 / 0.6189 | 15.7570 / 0.4789 |
| quadprior_plus_D | 17.5122 / 0.8253 | 16.9658 / 0.5696 | 15.6674 / 0.5915 | 15.7231 / 0.5162 |
| ours_step0_plus_D | 7.3630 / 0.3094 | 9.1117 / 0.1966 | 13.3316 / 0.4132 | 11.5833 / 0.2585 |

## Separate target-test-GT ceiling references (not held-out)

| Method ID | SDSD-indoor (dev) | LSRW | SMID | SID |
|---|---:|---:|---:|---:|
| ours_ttt_target_tuned | 19.0342 / 0.6252 | 16.3007 / 0.4484 | pending | pending |
| ours_v2_ceiling | 19.3349 / 0.8296 | 16.5389 / 0.5156 | pending | pending |

The ceiling uses that dataset's **test GT for tuning and κ selection**. It is the result of a finite, declared parameter search, **not a proven mathematical upper bound** on Ours-v2. It cannot support a held-out generalization claim or replace the ordinary `ours_v2`/`ours_v2_sdsd_knobs` rows. On LSRW, the completed ceiling does **not** lead: MR. Illuminate+D is 17.8100 dB / 0.5653, while QuadPrior+D reaches 0.5696 SSIM. It is even 0.0548 dB below the frozen Ours-v2 SDSD-parameter transfer row (16.5389 versus 16.5937), despite slightly higher RGB-SSIM (0.5156 versus 0.5151). A post-hoc per-image GT oracle over six *existing* Ours outputs still reaches only 16.9272 dB / 0.5114; see `T075A/LSRW_ceiling/oracle_diagnostic.md`. This is diagnostic, not a method row.

Source per-image results and read logs: `T075A/sdsd_v2/metrics/`, `T075B/LSRW/metrics/`, `T075B/SMID/metrics/`, `T075B/SID/metrics/`; LSRW ceiling: `T075A/LSRW_ceiling/`. The older 100-image official LOL-v2 Real test has only its separate frozen three-row comparison and is not mixed into this table. UHD-LL references remain sealed, so no UHD-LL PSNR/SSIM is reported. The severe PromptIR+DCTTA failures on SMID/SID are retained as observed rows, not used to infer an overall Ours win over baselines.
