# T035-A — shared-gain attribution control

**DONE — common-mode explains most WB gain.** Shared gain yields paired PSNR mean +2.0930311111438713 dB / median +1.5620560810430515 dB versus T028, exceeding both frozen thresholds. SSIM still declines by mean0.02324888738373933 (91/100 images).

## Implementation and frozen protocol

Source `82dc0b00dbf1bd234727670698c796e1f7eb19e6`, branch `codex/T035A-common-gain`, PR #60. Only the isolated oracle and its test were added. Deployable `ttie/` is unchanged. `CommonRegion2` inherits accepted T034 rendering, replaces raw parameters with 1×3×2×2 (12 scalar parameters: EV, gamma and one gain for each of four regions), and expands the one raw gain identically across RGB before applying existing TTIE physical parameters. Gain `exp(log(2)*tanh(raw))` is bounded [0.5,2] and starts at 1. Contrast remains 1. Existing hard masks, gate, inactive identity, EV/gamma projection and renderer ordering are unchanged. The exact accepted T028 `optimize_start` function is imported, not reimplemented.

Original frozen validation100; identity and accepted T026 selected starts; full RGB-MSE against paired normals; Adam lr0.05; exactly 500 updates/start; earliest strict minimum within each start, then same MSE/step/start tie-break. T028 and T034 are read-only. No setting changes, sweep, third start, new cohort, independent RGB freedom, other operators or official test. This is REFERENCE_ORACLE_ONLY, not deployable TTT; no oracle quantity may feed energy, gate, selector, training or future low-only inference.

Split SHA256 `b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b`. Accepted T026 execution `20260914-113853-ttie-t026a-gamma05`, frozen starts from accepted T028 preflight. T028 freeze SHA `e22d9cacb45cc634ca812588927f15efb0e0bed2c89faef2691178bbd268b2a2`, CSV SHA `caa23530153cf45f7b70958f8eca1e1de2675da64aff7322c05eb2cc4756d3f7`. T034 freeze SHA `8a1f60005329e75c362c0f37b7ba84c555059c8b813d9ac157ad6fcf2482be04`, CSV SHA `686a1497a6a52b3f3ed51e2fd5f4b6d916c3015aebd7dd95a31879c858db869b`. All 300 T028 and 400 T034 artifacts were hash-checked read-only, as were T026 bound per-image decisions/outputs/trajectories. All 31 staged source files match the deployed release; hashes are retained in `T035A_source_binding.json`.

## Regression, tests and execution

Preflight run `20260915-023543-ttie-t035a-preflight` completed `2026-09-14T18:35:56.702622+00:00`, exit0, zero normal decodes. Preflight SHA256 `77d6b0c4628aedf6202d2a4ec06d6141f5505fc9c8f0001abe2ad2f904b439fb`. On all100 lows at both original starts, identity gain reproduces T028 rendering with **max absolute error0.0**, including accepted selected-output reconstruction. On all100 lows with two predeclared nonidentity raw gain patterns `[-.8,.2,.7,-.3]` and `[.9,-.6,.1,.5]` (original inactive projection applied), shared-gain rendering matches T034 with tied RGB gains at **max absolute error0.0**. Both meet the unchanged <=1e-6 threshold before reference optimization. These smoke settings are not optimization starts or selected hyperparameters.

Local accepted baseline tests: **2passed8.43s**. New focused tests: **2passed7.71s**, covering identity/tied-RGB equivalence, inactive identity, 12-parameter shape, exact optimizer-function reuse, gradient updates and bounded tied gains. Server old+new: **4passed2.50s**. Existing NVML warning is nonblocking; CUDA is used for the real run.

Sole oracle/evaluation run `20260915-023649-ttie-t035a-oracle`, release `20260915-023511-ttie-t035a-common`, A6000 physicalGPU1, Torch2.4.0+cu121/CUDA12.1/Python3.12.12. `run.py` executes before `evaluate.py` through `&&`; all outputs freeze before PSNR/SSIM scoring. Reference MSE is deliberately allowed within the isolated optimization. Every start retains its 501 raw states/MSEs and its best float output. Same accepted T028 native float32 RGB/255 decoding then float64 metric computation; no crop, resize, Y conversion or quantization.

Fixed classification: common-minus-T028 PSNR mean >=1.835968277 dB and median >=1.431879733 dB yields `common-mode explains most WB gain`; mean <0.50 yields `chromatic degrees essential`; otherwise `mixed attribution`. These diagnostic thresholds cannot promote an action family. Both PSNR and SSIM differences will be reported regardless of direction. Finite-budget oracle reachability is not a certified global optimum.


## Final results

| Family/start/comparison | Mean PSNR | Median PSNR | Mean RGB-SSIM | Median RGB-SSIM |
|---|---:|---:|---:|---:|
| Accepted T028 | 17.459991778212 | 16.342409074030 | 0.431715829791 | 0.480401006108 |
| Common, identity-start best | 19.522654899810 | 18.870980771106 | 0.406857432145 | 0.429514660032 |
| Common, T026-start best | 19.552850720711 | 18.898213118998 | 0.408566640615 | 0.431895772367 |
| Common winning state | 19.553022889356 | 18.898213118998 | 0.408466942407 | 0.431895772367 |
| Accepted T034 full WB | 19.907949480744 | 19.568847169741 | 0.408435396171 | 0.429932909013 |
| Paired common minus T028 | 2.093031111144 | 1.562056081043 | -0.023248887384 | -0.019093323574 |
| Paired full WB minus common | 0.354926591388 | 0.182969701740 | -0.000031546235 | -0.000081781239 |

Shared gain accounts for **85.501114% of T034's aggregate mean PSNR gain** and **81.818468% of its aggregate median gain**. These ratios compare frozen aggregates; they are not per-image fractions or a formal causal variance decomposition. The control supports the common-mode attribution under the declared thresholds, while full channel-specific WB still provides mean +0.3549265913882905 dB / median +0.1829697017399461 dB residual benefit.

Common-minus-T028 PSNR win/equal/loss **95/0/5**; SSIM **9/0/91**. Five PSNR declines are small (worst -0.000341969339043402 dB), consistent with finite fixed-budget optimization rather than a certified optimum of the nested family. Full-WB-minus-common win/equal/loss: {"psnr": {"win": 100, "equal": 0, "loss": 0}, "ssim": {"win": 45, "equal": 0, "loss": 55}}. Full WB does not materially reverse the mean SSIM decline: its mean difference versus common is -0.00003154623545827462. No objective/budget was changed in response.

Winner starts: **84 T026 selected /16 identity**. **85/100** winning states are at step500; all both-start step histograms retained. Full both-start/winner scores and both paired comparisons are in `T035A_result/evidence/per_image.csv` and `summary.json`; `T035A_paired_comparisons.json` additionally contains all100 paired rows in JSON.

## Active-region parameter distributions

392 active/8 inactive region instances. Physical bound-hit tolerance1e-6; original EV region-specific lower/upper constraints. Full active/inactive/all groups, p05/p95 and 1200 scalar records are retained in summary/bound_values.

| Channel | Region | Count | Mean | Median | Min | Max | Lower/upper hits |
|---|---|---:|---:|---:|---:|---:|---:|
| ev | 00 | 96 | 1.089116 | 1.431586 | -0.500000 | 1.997842 | 3/4 |
| ev | 01 | 100 | 1.225358 | 1.704919 | -0.500000 | 1.997320 | 1/0 |
| ev | 10 | 98 | 1.282135 | 1.728653 | -0.500000 | 1.999377 | 3/3 |
| ev | 11 | 98 | 1.090553 | 1.430155 | -0.500000 | 1.997154 | 2/4 |
| ev | all | 392 | 1.172485 | 1.526396 | -0.500000 | 1.999377 | 9/11 |
| gamma | 00 | 96 | 0.549049 | 0.509739 | 0.500211 | 0.927217 | 0/0 |
| gamma | 01 | 100 | 0.547477 | 0.505451 | 0.500425 | 0.891905 | 0/0 |
| gamma | 10 | 98 | 0.559727 | 0.513413 | 0.500494 | 1.032915 | 0/0 |
| gamma | 11 | 98 | 0.564178 | 0.511345 | 0.500253 | 1.016834 | 0/0 |
| gamma | all | 392 | 0.555100 | 0.509528 | 0.500211 | 1.032915 | 0/0 |
| common_gain | 00 | 96 | 1.509689 | 1.476338 | 0.738405 | 1.998390 | 0/0 |
| common_gain | 01 | 100 | 1.411459 | 1.353926 | 0.677581 | 1.996859 | 0/0 |
| common_gain | 10 | 98 | 1.396899 | 1.310348 | 0.688510 | 1.998427 | 0/0 |
| common_gain | 11 | 98 | 1.487812 | 1.467625 | 0.782737 | 1.997570 | 0/0 |
| common_gain | all | 392 | 1.450964 | 1.409148 | 0.677581 | 1.998427 | 0/0 |

Active EV hits9/11; gamma0/0; common gain0/0. Gain maxima near1.998 and tanh bounds mean zero exact hits do not imply no near-bound behavior. The8 inactive regions remain EV0/gamma1/gain1 throughout histories; coincident inactive EV/gamma lower/upper constraints are separated from active hit counts. Tied RGB gains are exactly equal.

## Completion and verification

Sole run exited0 after100images×2starts×500updates=**100000 updates**, all501states/start retained. Outputs froze **2026-09-14T18:57:09.794647+00:00**, SHA `c74e911b6784dd9d201cbe10c68124e36502ab41bcdd210d21fdf881e173c89e`, before PSNR/SSIM scoring. Evaluation completed **2026-09-14T18:57:57.074032+00:00**. All400 new file hashes,200 finite/bounded histories, earliest minima, winning raws, output equality/range and validation-only decode paths passed. Reference MSE is intentionally permitted in this isolated diagnostic. Official-test access=false.

Independent dot-product PSNR and independently implemented RGB-SSIM agree on all both-start/winner outputs to maximum absolute error **1.7763568394002505e-14**. Local stdlib replay recomputed every metric/runtime/distribution aggregate, both paired comparisons, histograms and classification; max difference **3.552713678800501e-15**, and all100 retained state JSON hashes match the freeze. Full histories and image tensors were checked onserver; they are in the full F archive, not the local compact package. Ddisk capacity constrained local retrieval only, not experiments or evidence retention.

Runtime per image includes hashing/decoding, both optimizations and saving. Mean/median/p95 **12.129409277415 /12.239538745984 /12.409419671664 s**, total **1212.940927741467 s**. Postfreeze evaluation about47s. No scientific failure, restart, setting change or scope deviation. One SSH log-read connection closed during the run; immediate retry confirmed continued execution of the same job (52/100), with no process restart. Existing NVML warning was nonblocking.

## Archives and next action

Full archive `/media/wenchang/F/wjq/TTIE/shared/t035a/T035A_execution.tar`, **872243200 bytes**, SHA `38bec6ab81e38a39244c460243182436eec9b7e98870768fc3658de4573a42e9`. Compact **149461 bytes**, SHA `d638941cfe15251ce29b30fcb1460aec490dcb52de030e26ddce9d2ef205847c`, verified on both roots and locally. Exact commands/config/run/preflight logs are in `T035A_result/`. Recovery-critical report and delivery receipts are mirrored to outer project research_log.

Return the matched common-mode result, modest channel-specific residual and persistent SSIM decline to research lead. Stop awaiting review/new OPEN; no deployable action-family promotion, extra run, reference-derived controller or official-test access. PR #60 is for research-lead review, no self-merge.

common-mode explains most WB gain
