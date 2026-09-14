# T034-A — WB capacity audit

**DONE — substantial chromatic action headroom.** Paired PSNR mean +2.447957702532162 dB and median +1.9091729772217718 dB meet both fixed thresholds. However, SSIM falls on 90/100 images (mean -0.02328043361919761). This is MSE/PSNR capacity evidence, not an across-metric restoration improvement.

## Fixed implementation and bindings

Source `caed8ad78b42c377ebe02ddc149afe2ccaba86e6`, branch `codex/T034A-wb-oracle`, PR #59. The new isolated `research_log/T034A_oracle/core.py` imports the exact accepted T028 `optimize_start` function. There is no change to deployable `ttie/` code. `WBRegion2` replaces the parameter tensor with shape 1×5×2×2 and supplies existing TTIE physical parameters: EV, gamma and RGB WB, with contrast fixed to identity. Hard Region2 geometry, the inactive output bypass and renderer ordering are inherited unchanged. The original Gamma05Box projection operates on the EV/gamma slice, while inactive WB raw values remain zero. Active WB uses `exp(log(2)*tanh(raw))`, physically bounded to [0.5,2] and initialized at gain 1.

Original 100-image T022/T026 development validation split SHA256 `b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b`. Accepted T026 source `b2359721c89db732d17e03be273e0bdb71bb377a`, execution `20260914-113853-ttie-t026a-gamma05`. T028 source `80006657e9b270644f8e0ad9fdf1cb6309966807`, execution `20260914-163238-ttie-t028a-oracle`; its freeze SHA256 is `e22d9cacb45cc634ca812588927f15efb0e0bed2c89faef2691178bbd268b2a2`, and read-only per-image CSV SHA256 is `caa23530153cf45f7b70958f8eca1e1de2675da64aff7322c05eb2cc4756d3f7`. Per-image accepted gate, starts, outputs and trajectory hashes are preserved in the preflight receipt. All 300 prior T028 output/state/history file hashes were checked before this run. All 26 staged file hashes match the deployed release.

The identity-WB regression covered all 100 validation low images at both fixed starts, plus reconstruction of all accepted selected outputs: maximum absolute error **0.0**. Preflight completed `2026-09-14T17:42:09.587465+00:00`, with zero normal-image decodes and SHA256 `725581dcae9b185e9dc6b69f4cefb5511effcf3392c94e2311501f441263bb2c`. Bound starting tensors are retained alongside the receipt. This exceeds the requested fixed smoke-set coverage without changing the <=1e-6 acceptance threshold.

## Experiment and tests

Sole full run `20260915-014747-ttie-t034a-oracle`, deployed release `20260915-014124-ttie-t034a-wb`, A6000 physical GPU 1. Torch 2.4.0+cu121 / CUDA 12.1 / Python 3.12.12. Two starts per image (identity and frozen T026 selected); Adam lr 0.05; exactly 500 updates per start; full-frame RGB MSE with float64 accumulation through the float32 renderer. Earliest minimum MSE within each start; winner tie-break uses MSE, earliest step and identity start. Every start retains all 501 raw states and objective values, plus its best float output. Image scoring follows accepted T028 native float32 RGB decoding promoted to float64, without crop, resizing, luminance conversion or PNG quantization.

Local original-oracle baseline test: **1 passed, 9.53 s**. Local new focused tests: **2 passed, 7.88 s**. Remote old+new tests in preflight run `20260915-014157-ttie-t034a-preflight`: **3 passed, 2.62 s**. Tests cover identity-WB renderer equivalence, mixed active/inactive regions, WB bounds, WB gradients and literal optimizer-function reuse. Preflight exited 0.

One SSH connection timed out while resuming, before launching the full experiment. The retry connected successfully and confirmed no full T034 run or tmux session existed. This connectivity failure did not execute any optimization. The existing NVML warning is nonblocking; actual CUDA computation completed successfully. The full experiment has not been restarted or tuned.

## Interpretation boundary

This is **REFERENCE_ORACLE_ONLY**, not deployable TTT. Paired normals may guide this isolated capacity diagnostic; none of its states, metrics, best steps or bound patterns may feed the deployed energy, gate, selector or training. The official test remains unopened. T028 and Retinexformer are not rerun, and no method is promoted.

The test measures the finite-search reachability of the WB-expanded family, not a certified global optimum. RGB gains include a common intensity component, so an improvement would demonstrate added WB-family capacity without uniquely attributing the gain to chromatic correction. EV/gamma parameter bounds remain unchanged, but a common RGB gain can still change total rendered brightness. No extra experiment or parameter restriction was introduced to separate these effects.

Scientific classification remains exactly the predeclared paired expanded-minus-T028 rule: substantial if mean >=1.50 dB and median >=1.00 dB; limited if mean <0.50 dB; mixed otherwise.


## Final results

| Family/start | Mean PSNR | Median PSNR | Mean RGB-SSIM | Median RGB-SSIM |
|---|---:|---:|---:|---:|
| Read-only T028 | 17.459991778212 | 16.342409074030 | 0.431715829791 | 0.480401006108 |
| Identity-start best | 19.874705897565 | 19.537139401993 | 0.406828717924 | 0.429211561418 |
| T026-selected-start best | 19.907764821207 | 19.568847169741 | 0.408503753316 | 0.429932909013 |
| Expanded winning state | 19.907949480744 | 19.568847169741 | 0.408435396171 | 0.429932909013 |
| Paired expanded minus T028 | 2.447957702532 | 1.909172977222 | -0.023280433619 | -0.017232476929 |

PSNR win/equal/loss **100/0/0**; SSIM **10/0/90**. PSNR deltas range +0.005025533990 to +9.701675106358 dB; SSIM deltas range -0.098749002724 to +0.031736316482. The MSE objective and selection do not optimize SSIM. No objective or threshold changed in response to this tradeoff.

Winner starts: 83 T026 selected, 17 identity. Winning step 500 occurs on 87/100 images; full both-start histograms and per-image values are in `T034A_result/evidence/summary.json` and `per_image.csv`. Frequent budget-end minima reinforce the finite-search interpretation.

## Winning-state parameter distributions

392 active and 8 inactive region instances. Table: active regions only; full active/inactive/all means, medians, min/max and p05/p95 by channel and region are in `summary.json`; all 2000 scalar records are in `bound_values.json`. Boundary hits use physical constraints and absolute tolerance 1e-6.

| Channel | Region | Count | Mean | Median | Min | Max | Lower/upper hits |
|---|---|---:|---:|---:|---:|---:|---:|
| ev | 00 | 96 | 1.114868 | 1.443233 | -0.500000 | 1.997843 | 3/7 |
| ev | 01 | 100 | 1.234366 | 1.707553 | -0.500000 | 1.997321 | 2/5 |
| ev | 10 | 98 | 1.287216 | 1.730352 | -0.500000 | 1.999377 | 4/4 |
| ev | 11 | 98 | 1.109869 | 1.447336 | -0.500000 | 1.997155 | 3/3 |
| ev | all | 392 | 1.187189 | 1.531377 | -0.500000 | 1.999377 | 12/19 |
| gamma | 00 | 96 | 0.549039 | 0.510490 | 0.500211 | 0.892786 | 0/0 |
| gamma | 01 | 100 | 0.549706 | 0.505755 | 0.500425 | 0.891434 | 0/0 |
| gamma | 10 | 98 | 0.558151 | 0.512539 | 0.500409 | 0.991066 | 0/0 |
| gamma | 11 | 98 | 0.564766 | 0.510621 | 0.500253 | 0.980870 | 0/0 |
| gamma | all | 392 | 0.555419 | 0.510172 | 0.500211 | 0.991066 | 0/0 |
| wb_r | 00 | 96 | 1.474613 | 1.439730 | 0.753186 | 1.998196 | 0/0 |
| wb_r | 01 | 100 | 1.391515 | 1.337521 | 0.731828 | 1.996880 | 0/0 |
| wb_r | 10 | 98 | 1.383816 | 1.359654 | 0.660499 | 1.998165 | 0/0 |
| wb_r | 11 | 98 | 1.471793 | 1.463986 | 0.759564 | 1.997305 | 0/0 |
| wb_r | all | 392 | 1.430010 | 1.388076 | 0.660499 | 1.998196 | 0/0 |
| wb_g | 00 | 96 | 1.510570 | 1.469255 | 0.819954 | 1.998742 | 0/0 |
| wb_g | 01 | 100 | 1.427586 | 1.349838 | 0.772666 | 1.997481 | 0/0 |
| wb_g | 10 | 98 | 1.418975 | 1.335347 | 0.794399 | 1.998575 | 0/0 |
| wb_g | 11 | 98 | 1.511064 | 1.458284 | 0.955175 | 1.997685 | 0/0 |
| wb_g | all | 392 | 1.466626 | 1.415871 | 0.772666 | 1.998742 | 0/0 |
| wb_b | 00 | 96 | 1.504352 | 1.492179 | 0.570567 | 1.998352 | 0/0 |
| wb_b | 01 | 100 | 1.421044 | 1.394853 | 0.561419 | 1.997485 | 0/0 |
| wb_b | 10 | 98 | 1.346113 | 1.274911 | 0.608067 | 1.998683 | 0/0 |
| wb_b | 11 | 98 | 1.450513 | 1.427735 | 0.566724 | 1.997862 | 0/0 |
| wb_b | all | 392 | 1.430080 | 1.415446 | 0.561419 | 1.998683 | 0/0 |

All WB channels have zero 1e-6 boundary hits, but maxima near 1.999 and tanh bounds mean zero exact hits do not establish absence of near-bound behavior. Active EV has 12 lower/19 upper hits; active gamma 0/0. All 8 inactive regions keep WB=gamma=1 and EV=0 throughout histories. Inactive EV/gamma have coincident lower/upper identity constraints; their 8/8 hits are separated from active counts.

## Completion and numerical verification

Single run completed 100 images, 200 starts, **100000 updates**, 501 states/start, exit 0. All outputs froze at **2026-09-14T18:07:03.199003+00:00**, SHA256 `8a1f60005329e75c362c0f37b7ba84c555059c8b813d9ac157ad6fcf2482be04`, before PSNR/SSIM scoring; evaluation completed **2026-09-14T18:07:51.644805+00:00**. Reference MSE was deliberately permitted during this isolated optimization. All 400 output/state/history/start-output file hashes, 200 finite/bounded histories, earliest minima and winning states were verified. Saved winning images exactly match the best-start image. All decoded paths are specified validation lows/normals; official test access false.

Independent dot-product PSNR and independent RGB SSIM for both start-best images and winners: maximum absolute error **1.7763568394002505e-14**. Local stdlib replay recomputed all metric/runtime/distribution summaries, paired deltas, histograms and classification: maximum difference **3.552713678800501e-15**. All 200 retained history/state hashes match freeze. Full image arrays remain on server/F and were checked by server evaluation; compact local evidence contains histories, states and scalars.

Per-image timing includes hashing/decoding, both optimizations and saving retained files. Mean/median/p95 **11.481785686037 /11.537162375025 /11.655964512471 s**, total **1148.178568603704 s**. Postfreeze evaluation took about 48 s. No scientific failure, rerun or parameter deviation. Operational issues: one pre-launch SSH timeout, nonblocking NVML warning; local report generation first failed on Windows GBK decoding and was retried with UTF-8, without changing experiment files.

## Archives and next action

Full F backup `/media/wenchang/F/wjq/TTIE/shared/t034a/T034A_execution.tar`, **875827200 bytes**, SHA256 `361b28ff7f42437ed7881e29bc9b29090165b42a8d9b862509be7506e5d220f2`. Compact **7672142 bytes**, SHA256 `f8d408939171ccf89b446ca5f4c7c21290d3ecacb82e1c0a79003cfb2cf2e74b`, verified on both remote roots and locally. Exact command/run/preflight logs are in `T034A_result/`; 26 staged-file hashes in `T034A_source_binding.json`. Report and delivery receipts are mirrored to outer project research_log.

Return the positive fixed PSNR classification with SSIM decline and common-gain attribution limitation. Stop awaiting research-lead review/new OPEN task; no deployable WB adaptation, retraining, selector change, extra start or test access. PR #59 remains for review without self-merge.

substantial chromatic action headroom
