# T068-A — DONE / ABS_STEP_CAP_DEV_NO_GAIN

The complete development-only absolute-step cap family selects K=27 (the no-cap control). Close this family under the predeclared stop rule. No cap is advanced to transfer or fresh qualification.

Source: `3f0f2d5cbca0ab13efbd7ddd1bdf5570c74cbd38`; branch: `codex/T068A-dev-step-cap`; authorization: `b3ba8989dc0071891f7992a4d22f6bc92da0609b`.

Implementation reuses accepted T066-A model/input/renderer and T067-B interpolation, with lambda=0.875, rho=0.9857470621423519, p_safe threshold=0.5. The only new rule is k_K=max(k_FS,min(k_lambda,K)) for every integer K=0..27. No optimizer rerun or model fit.

All 2,800 choices were frozen at `2026-09-20T22:29:15.369214+00:00`, with reference_reads=0 and SHA256 `bf0168154550137bfe540e5f9149fd865cb2bff8a0817d8d6c01e6ba364af2c1`. The first development quality-access boundary is `2026-09-20T22:29:15.401483+00:00`; evaluation binding/quality were read only after the freeze. Every row binds low/trace/images, state/render, model and rule hashes. K=27 reproduces prior T067-B choices/state/output identities before evaluation and all prior metrics exactly after evaluation.

Ranking among all-five-gate-pass candidates maximizes worst paired T026 PSNR, then mean T036 PSNR, then median T036 PSNR, then larger K on exact ties. Passing order: [27, 26, 25, 24, 23, 22, 21, 20]. K=25/26/27 produce identical choices and metrics; the specified larger-K tie break selects 27.

| K | Mean ΔPSNR vs T036 | Median ΔPSNR vs T036 | Regressions vs T026 | Worst ΔPSNR vs T026 | Mean ΔSSIM vs T036 | All gates |
|---:|---:|---:|---:|---:|---:|:---:|
| 0 | -2.7553243375 | -2.4675441866 | 80 | -5.5386899816 | -0.1355450430 | FAIL |
| 1 | -2.7269444986 | -2.4439005765 | 80 | -5.5386899816 | -0.1327935208 | FAIL |
| 2 | -2.6498584938 | -2.4176167806 | 78 | -5.5386899816 | -0.1285591655 | FAIL |
| 3 | -2.5684893918 | -2.3884913806 | 76 | -5.5386899816 | -0.1240718924 | FAIL |
| 4 | -2.4521573421 | -2.3444161600 | 75 | -5.2896226680 | -0.1175455905 | FAIL |
| 5 | -2.3237064550 | -2.2315915243 | 74 | -5.2476588532 | -0.1105227495 | FAIL |
| 6 | -2.1905149325 | -2.1109820413 | 74 | -5.2476588532 | -0.1032198021 | FAIL |
| 7 | -2.0391895836 | -1.9827612953 | 71 | -5.1578011671 | -0.0944423833 | FAIL |
| 8 | -1.8715099807 | -1.8472743752 | 70 | -5.1578011671 | -0.0847469935 | FAIL |
| 9 | -1.6878750836 | -1.6273089709 | 70 | -5.1578011671 | -0.0743962009 | FAIL |
| 10 | -1.4829984288 | -1.5629707563 | 67 | -5.1578011671 | -0.0634462830 | FAIL |
| 11 | -1.2315087513 | -1.4048143477 | 64 | -4.4461963907 | -0.0511748462 | FAIL |
| 12 | -0.9532974127 | -1.1147218222 | 64 | -3.9213170728 | -0.0390353949 | FAIL |
| 13 | -0.6601040003 | -0.8502120354 | 59 | -3.8622762616 | -0.0278072816 | FAIL |
| 14 | -0.3366432428 | -0.4072016947 | 45 | -3.7721936217 | -0.0175669114 | FAIL |
| 15 | 0.0214412417 | -0.1900405971 | 22 | -3.6525511867 | -0.0084672325 | FAIL |
| 16 | 0.4037271146 | 0.0760071940 | 7 | -3.5068847303 | -0.0007395055 | FAIL |
| 17 | 0.8058250990 | 0.4048491908 | 2 | -3.3413666854 | 0.0055488750 | FAIL |
| 18 | 1.2225232436 | 0.6221519112 | 1 | -3.1627577966 | 0.0104858176 | FAIL |
| 19 | 1.6474127334 | 1.0705276386 | 1 | -2.9731603418 | 0.0142039061 | FAIL |
| 20 | 2.0065245618 | 1.5662029888 | 1 | -2.7534756632 | 0.0168747654 | PASS |
| 21 | 2.2862139040 | 2.0086444944 | 1 | -2.5656754908 | 0.0185976566 | PASS |
| 22 | 2.5009989998 | 2.2330290554 | 1 | -2.4273991686 | 0.0195673855 | PASS |
| 23 | 2.5949487916 | 2.2396107203 | 1 | -2.4273991686 | 0.0201111727 | PASS |
| 24 | 2.6337663976 | 2.2431391512 | 1 | -2.4273991686 | 0.0204121775 | PASS |
| 25 | 2.6360346285 | 2.3057191323 | 1 | -2.4273991686 | 0.0204407487 | PASS |
| 26 | 2.6360346285 | 2.3057191323 | 1 | -2.4273991686 | 0.0204407487 | PASS |
| 27 | 2.6360346285 | 2.3057191323 | 1 | -2.4273991686 | 0.0204407487 | PASS |

Selected K=27 changes **0/100** choices vs uncapped lambda=0.875. Step histogram: `{"16": 1, "17": 3, "18": 1, "19": 3, "20": 2, "21": 18, "22": 22, "23": 28, "24": 20, "25": 1, "9": 1}`. Complete selected rows and per-K metrics/gates are in evidence.

Validation: baseline3 passed12.58s; cap increment3 passed13.93s; final affected10 passed12.01s locally; imports PASS; remote10 passed1.83s. Commands: `python -B -m pytest research_log/T068A research_log/T067B research_log/T063C -q`, then `python -B -m research_log.T068A.run --out /media/wenchang/F/wjq/TTIE/runs/T068A-dev-step-cap`, then the same with `research_log.T068A.verify`.

Independent verifier PASS: 2,800 candidate choices, 2041 unique frozen-state renders on A6000 GPU1, and 2,800 independently recomputed development PSNR/SSIM states using CPU. Maximum discrepancy 1.0871303857129533e-12. Recomputed selection formula, gates, exact ranking, control identity, selected histogram and frozen manifest. optimizer_runs=0; model_fits=0.

Run `20260921-062905-ttie-t068a-step-cap`, exit0 at 2026-09-21T06:30:02+08:00. Primary scalar calibration 0.8729428109945729 seconds. Release `/media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t068a-step-cap`; output `/media/wenchang/F/wjq/TTIE/runs/T068A-dev-step-cap`. Environment: existing Python3.12/Torch2.4+cu121, GPU1, TF32 disabled in verifier, OMP/MKL/OpenBLAS threads1 with sequential MKL. Source binding contains 280 files; separate evaluation binding contains development quality and prior control metrics.

Failures/deviations: local git whitespace preflight rejected generated CRLF files; normalized only task-owned files to LF and refreshed hashes before scientific source commit. One SSH log read timed out and succeeded on retry. Neither issue caused an experiment rerun. Home had only ~320MB free, so the release and outputs use F storage; small recovery copy retained under project home. No scientific deviations.

Raw archive: `/media/wenchang/F/wjq/TTIE/shared/t068a/T068A_raw.tar`, 2611200 bytes, SHA256 `47da7ea937334e4e9e763b9259dee90118eb8fd6217d3e8117d94fd0c59c6cef`. Recovery archive: `/home/wenchang/asdasdsad/wjq/TTIE/shared/t068a/T068A_recovery.tar.gz` (F backup `/media/wenchang/F/wjq/TTIE/shared/t068a/T068A_recovery.tar.gz`), 2271545 bytes, SHA256 `cfd2236fb02c5fee3ca3d2dba7d68feece6b431386f93af9200d7f0dcb581712`; downloaded and hash-verified locally.

Scope/next step: this is original-development, model-in-sample calibration only. No transfer cohort/reference, fresh set, official test, LSRW or UHD-LL was opened. No final benchmark claim. Absolute cap family is closed as NO_GAIN; await a new bounded research-lead task. No second rule or transfer evaluation was attempted.
