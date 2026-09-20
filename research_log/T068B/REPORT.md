# T068-B — DONE / OBJECTIVE_MOTION_KNEE_DEV_NO_GAIN

The exact objective-motion knee changes 98/100 development choices but fails mean PSNR, median PSNR and mean SSIM gates. Close this exact rule under the predeclared stop criterion. No parameter change or transfer evaluation.

Source: `86b99ad7261bc9ac4b1b4fad089f616137de0d27`; branch: `codex/T068B-objective-motion-knee`; authorization: `c33806afc811938efc3ba2648ec9724e1494cea7`.

Reuse: T066-A model/input/state/renderer, T067-B frozen lambda=0.875 interpolation/endpoints, T063B exact gates/metrics, T066A independent development scoring. New logic is only the specified float64 objective-motion knee and audit orchestration. No optimizer/model fit. Fixed rho=0.9857470621423519, probability threshold0.5, epsilon1e-12; larger step only on exact advantage ties.

Primary RMS motion uses GPU1 float64 after promoting exact frozen float32 images before subtraction. Cumulative motion, clipped progress, advantage and argmax are float64. Complete inclusive u/d/s/v/a curves and model/rule/input/selected hashes are frozen for all100 images. For mathematically undefined degenerate ratios the predeclared serialization uses null entries and the specified endpoint; d_FS=0. Actual reason counts: `{"maximum_advantage": 100}`.

Freeze UTC `2026-09-20T23:12:49.384434+00:00`, SHA256 `90620e0596fedd2e5c16ec7867719b53eeaf540500d5b75ccfa5831bcc3ada71`, reference_reads=0 in table and every row. First development-quality access boundary UTC `2026-09-20T23:12:49.397428+00:00`; evaluation-binding hashing and quality access occur only afterward. Frozen T067-B endpoints/state/output identities and exact control metrics reproduced.

| Metric | Knee | Frozen lambda=.875 control | Knee gate |
|---|---:|---:|:---:|
| Mean PSNR delta vs T036 | -0.0056981598218439 | 2.636034628456087 | FAIL |
| Median PSNR delta vs T036 | -0.11985705550209236 | 2.3057191322756285 | FAIL |
| Regressions vs T026 | 25 | 1 | PASS |
| Worst paired PSNR delta vs T026 | -2.42739916855475 | -2.42739916855475 | PASS |
| Mean RGB-SSIM delta vs T036 | -0.011646022623018221 | 0.02044074873815783 | FAIL |

Paired knee minus exact interpolation control: mean PSNR `-2.6417327882779307 dB`, median PSNR `-2.304739135268329 dB`, mean RGB-SSIM `-0.032086771361176054`. These are reported only; no post-outcome tuning.

Changed choices: 98/100. Selected-step histogram: `{"4": 1, "10": 5, "11": 2, "12": 6, "13": 19, "14": 25, "15": 16, "16": 20, "17": 1, "18": 3, "22": 1, "24": 1}`. Histogram of changed selections: `{"4": 1, "10": 5, "11": 2, "12": 6, "13": 19, "14": 25, "15": 16, "16": 20, "17": 1, "18": 3}`. All changes move earlier inside the frozen interval. Complete100 rows are in candidate_freeze/selected_choices and paired_control evidence.

Validation: focused core4 passed3.42s; final affected11 passed,1skipped9.85s locally; skipped GPU-only test ran remotely; imports PASS. Remote12 passed1.83s, including the actual float64 GPU RMS numerical test. Commands: `python -B -m pytest research_log/T068B research_log/T067B research_log/T063C -q`; then `python -B -m research_log.T068B.run --out /media/wenchang/F/wjq/TTIE/runs/T068B-dev-motion-knee`; then same with `research_log.T068B.verify`.

Independent verifier PASS: 2041 interval states rerendered on A6000 GPU1; independent CPU float64 RMS (dot-product MSE), cumulative motion, progress, advantage and exact chosen steps; 2,800 independently recomputed development PSNR/SSIM states; gates/control/paired deltas/histograms/manifests checked. Curve max discrepancy `2.5701663020072374e-14`; metric max discrepancy `1.0871303857129533e-12`. optimizer_runs=0, model_fits=0.

Run `20260921-071214-ttie-t068b-motion-knee`, exit0 at2026-09-21T07:13:43+08:00; primary duration 27.146354639000492 seconds. Release `/media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t068b-motion-knee`; output `/media/wenchang/F/wjq/TTIE/runs/T068B-dev-motion-knee`. Existing Python3.12/Torch2.4+cu121; CUDA_VISIBLE_DEVICES=1; TF32 disabled; sequential MKL and OMP/MKL/OpenBLAS threads1. 280 source/input bindings with development quality separately bound post-freeze.

Failures/deviations: none. No scientific changes after source freeze. Release/output on F storage as in T068-A.

Raw archive: `/media/wenchang/F/wjq/TTIE/shared/t068b/T068B_raw.tar`, 1187840 bytes, SHA256 `d0b1acde2cdf116031f789fe73033e1ce315ca2ca142c574d9b0f84b4802b9ee`. Recovery: `/home/wenchang/asdasdsad/wjq/TTIE/shared/t068b/T068B_recovery.tar.gz`, F backup `/media/wenchang/F/wjq/TTIE/shared/t068b/T068B_recovery.tar.gz`, 2445362 bytes, SHA256 `f4a78b2699b32df59cd3f9aa9e0f63f86e34bc595dcc329cf9c61d6900ce466b`; downloaded/hash-verified locally.

Interpretation: the exact knee triggers substantially earlier and loses utility without improving the development worst-tail metric. This closes only this exact formula. Original-development/model-in-sample audit, not final benchmark evidence. No transfer/fresh/test/cross-dataset reference was opened. Stop here and await the research lead; do not adjust epsilon, tie rule, motion metric or add a selector.
