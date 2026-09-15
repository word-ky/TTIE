# T046-A: common-gain oracle convergence extension

**T035 common-gain oracle material underconvergence not supported under fixed extension**. Paired T046-minus-T035 PSNR mean **+0.0202049524 dB**, median **+0.0123440643 dB**. The only gate is mean>=1.00 AND median>=0.75. SSIM, counts, endpoint frequency and baseline gaps are descriptive only.

**REFERENCE_ORACLE_ONLY.** Normals are permitted exclusively inside this isolated reachability/convergence diagnostic. No reference, oracle state or result enters deployable TTT, checkpoint selection or learned-energy training. Official LOL-v2 Real test remains untouched.

T046-A-EXEC update `1853cb45` accepts the exact executed source `4ecababa94fb6779537cdebad1ee13bc0f734124` with unchanged settings/gate. It was observed during delivery after the one run and replay completed; this report also closes that execution request. No second run was started.

## Exact input/source bindings and low-only preflight

Same frozen T022/T033100-image split SHA `b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b`. Accepted T035 output freeze SHA `c74e911b6784dd9d201cbe10c68124e36502ab41bcdd210d21fdf881e173c89e`. The T035 renderer, hard Region2 masks/gate, EV/gamma bounds, RGB-shared gain bounds/order, native float32 decoding and optimizer function are reused unchanged. Complete original T035 source-binding checks pass, plus T046 source-binding SHA `542dc0f7e281e43646f8b86e8888ec25389e11d5ed7453798f52b35a88aa1977`. The immutable accepted source tree and new orchestration are enumerated in T046A_source_binding.json.

Before any normal decode, all100 accepted T035 winning states were verified against their original two-start histories, earliest winning step, gate/box and output files. All100 winners reconstructed from low image+fixed gate/raw state with **max absolute error 0.0**, **all bit-exact=True**. Frozen starts SHA `159c6b8621ce3680c1a8869a3ef6342397e1e7d49b885ba47b4450c36d9af048`. Preflight complete **2026-09-15T09:33:02.613923+00:00**, SHA `c7a75601cc2949fb1ef3d73d19a456cc34ae448a6965f299ffaa63fe5451c69a`; exactly100 allowed low decodes, normal decodes0. This reads already-accepted reference-derived T035 winner metadata for identity verification; it does not recompute a winner from new reference information.

## Fixed optimization and selection

Exactly100 images x one frozen T035 winning start x1000 Adam updates =100000 updates. Fresh Adam lr0.05 is constructed independently per image by the unchanged accepted optimize_start function; no prior moments existed to resume. This is a fixed restart-at-winner convergence probe, not a literal optimizer-moment continuation. Objective is full-frame RGB-MSE with float64 accumulation through the unchanged float32 renderer. Retain step0 and1000 updated states/MSEs; select earliest strict minimum. No alternate start, budget, optimizer, operator, bounds, geometry, full-WB run or scientific retry.

All100 selected outputs/states frozen **2026-09-15T09:54:24.508299+00:00**, SHA `2cdf5b8f104dc11445a44e6a68ebc8209192288e08e10bb60859cb95d5693057`, before the PSNR/SSIM evaluator. Every row binds the raw-state hash, selected-output tensor hash and output/history file hashes. All100100 retained states are finite and bounded; inactive gain coordinates remain identity. Normals opened only after the common preflight, validated by exact image pairing and hash.

## Results

| State | Mean PSNR | Median PSNR | Mean RGB-SSIM | Median RGB-SSIM |
|---|---:|---:|---:|---:|
| Accepted T035 | 19.5530228894 | 18.8982131190 | 0.4084669424 | 0.4318957724 |
| T046 fixed extension | 19.5732278418 | 18.9220694331 | 0.4093670846 | 0.4331425817 |
| Paired difference | +0.0202049524 | +0.0123440643 | +0.0009001422 | +0.0003533930 |

PSNR win/equal/loss: **{'win': 99, 'equal': 1, 'loss': 0}**. SSIM: **{'win': 81, 'equal': 1, 'loss': 18}**. Best extension step1000 on **49/100** images. Full earliest-best-step histogram: `{"0": 1, "1000": 49, "467": 1, "623": 1, "847": 1, "868": 1, "875": 1, "896": 1, "913": 1, "926": 1, "936": 1, "943": 1, "951": 1, "963": 1, "969": 1, "977": 1, "987": 1, "988": 1, "990": 3, "991": 1, "992": 1, "993": 2, "996": 3, "997": 4, "998": 4, "999": 16}`.

![Paired extension changes](T046A_result/paired_extension.png)

Remaining mean gaps (anchor minus T046): T033 Retinexformer **+1.9055586 dB / +0.3806941 SSIM**; T045 SNR-Aware **+3.8231021 dB / +0.4143973 SSIM**. Only prior scalar anchors are used; no anchor output was loaded by optimization or selection. These supervised baselines are training-exposed development anchors, not independent held-out SOTA comparisons. The present oracle is non-deployable.

## Metric and independent replay

Exact accepted T035/T026 native RGB pixel handling: float32 pixels promoted tofloat64, no crop/resize/Y conversion/brightness matching. PSNR is full RGB-MSE; RGB-SSIM range1,11x11 Gaussian sigma1.5, K1=.01/K2=.03, population covariance, reflect half-sample symmetric borders retained. T035 paired CSV is immutable, SHA `d531c45ce670f5b84c2b74059e931350e21e49c7d1c70d15229f4fbb26a24d72`.

Independent replay imports no main T046 score/statistics/verdict helper. It reads all100 histories, independently selects the earliest minimum, checks exact raw/start/output hashes, validates100100 finite bounded states using NumPy physical maps, recomputes200 selected-image PSNR/SSIM values with torch float64 MSE and explicit separable convolve1d SSIM, then reconstructs paired deltas/means/medians/counts/full best-step histogram/verdict. **712 scalar checks**, max discrepancy **7.105427357601002e-15 <=1e-10**, PASS; elapsed **7.923366s**. Preflight-before-normal and freeze-before-metric ordering are independently checked.

## Execution, failures and recovery

Tested source **4ecababa94fb6779537cdebad1ee13bc0f734124**, branch `codex/T046A-oracle-convergence`, PR https://github.com/word-ky/TTIE/pull/71. Release `20260915-173225-ttie-t046a-extension`; GPU1 preflight `20260915-173248-ttie-t046a-preflight`; GPU1 oracle `20260915-173335-ttie-t046a-oracle`; evaluation `20260915-175524-ttie-t046a-eval`. Exact commands in saved run.sh files. CUDA_VISIBLE_DEVICES1, CUBLAS_WORKSPACE_CONFIG=:4096:8, seed7, TF32off, OMP/MKL/OPENBLASthreads1. Preflight **3.847411s**; oracle100-image wall-time sum **1182.400011s**. Allactual runs exit0.

Validation: accepted baseline2tests24.13s; focused2tests21.67s; server4tests2.55s; allpass. Focused checks verify original optimizer identity, fresh-moment repeated-start behavior, retained states/earliest minima and both verdict boundaries.

One infrastructure failure: SSH255 connection closed during initial oracle launch before tmux started. Confirmed no session and no log, then launched the exact already-generated run.sh once. No partial trajectory, repeated image, altered settings or scientific restart. Existing NVML/protobuf deprecation warnings were nonblocking. Report-generator quoting errors were caught by compilation and corrected before result generation. No scientific deviation, deployable modification, model training, official-test access or extra probe.

All selected-output/history and source hashes reverified during archival. Full F archive **294942720 bytes**, SHA `08c343bf405f43e85121f0412ea64b5bf24e51d61e73cee42790245828bc661e`. Compact evidence including all retained raw/MSE histories **4325727 bytes**, SHA `db9b5f07bf9139f043b42b5532f8235d1055869668c09e22cc794b653c5da676`, verified home/F/local. Raw output images remain server/F; compact histories, starts, receipts, metrics, reports and Git-exact recovery are retained in project research_log. Main Codex mailbox receives one append-only completion report.

Stop after this one fixed convergence probe. Interpret the verdict as evidence about this specified budget and fresh-Adam probe; do not infer a global optimum or a hard capacity ceiling, and do not launch another operator/retraining experiment without the next research task.

T035 common-gain oracle material underconvergence not supported under fixed extension
