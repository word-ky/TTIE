# T069-BN: GRADIENT_NUMERICS_CHARACTERIZED

All 100 reconstructed development endpoint direct gradients match the original stored T062 optimizer gradients exactly (zero residual in all 1,200 coordinates). Two float32 component-sum rows fail the unchanged T069-B coordinatewise tolerance, while repeated direct/component gradients are hash-identical for all 100 rows. The unchanged float64 control reduces decomposition residuals to approximately machine precision. This supports a float32 backward decomposition/reduction explanation for the previous blocker; no mismatch with the stored direct optimization gradient was observed.

Source: `afc6fbeb60ef6debb777b2181480c323fe81313d`; branch `codex/T069BN-gradient-numerics`. Authorization: `65ae63ffd39c80045f27ddbfd9c79c8a5eea5768`. Review task-owned `research_log/T069BN/**`; the stacked historical diff is not a merge recommendation.

Files: `PLAN.md`, frozen `authorization.md`, `core.py`, `test_core.py`, `run.py`, `verify.py`, development-only `binding.json`, `REPORT.md`, and `evidence/{config.json,development_freeze.json,result.json,verification.json,archives.json,run.sh,train.log,meta.json}`. Existing renderer/loss/selection code was reused unchanged. No new dependencies.

## Numerical distributions

Absolute-coordinate distributions use all 1,200 coordinates; row-max and relative distributions use 100 endpoints. p95 uses the predeclared linear percentile convention.

| Quantity | Maximum | Median | p95 |
| --- | ---: | ---: | ---: |
| direct_vs_trace absolute_coordinates | 0 | 0 | 0 |
| direct_vs_trace absolute_row_max | 0 | 0 | 0 |
| direct_vs_trace l2_relative | 0 | 0 | 0 |
| sum_vs_direct absolute_coordinates | 1.12177804112e-06 | 4.98257577419e-08 | 3.02272383124e-07 |
| sum_vs_direct absolute_row_max | 1.12177804112e-06 | 2.14437022805e-07 | 6.88759610057e-07 |
| sum_vs_direct l2_relative | 2.06871792051e-06 | 4.9877330251e-07 | 1.71884033957e-06 |
| float32 score sensitivity | 4.90836523648e-07 | 8.24873190308e-08 | 3.09188842262e-07 |
| float64 sum_vs_direct_abs | 2.59514632006e-15 | 8.32667268469e-17 | 5.41927613895e-16 |
| float64 sum_vs_direct_l2_relative | 5.91129931884e-15 | 1.0356453549e-15 | 3.39176529735e-15 |
| float64 score_sensitivity | 8.881784197e-16 | 1.11022302463e-16 | 5.55111512313e-16 |

The original `abs(diff) <= 2e-7 + 2e-5*abs(anchor)` criterion is unchanged: direct-vs-trace failures 0/100 rows and 0/1,200 coordinates; sum-vs-direct failures 2/100 rows and 2/1,200 coordinates. Those rows are development index 8 (endpoint 21) and index 57 (endpoint 17). These identities were not selected using quality or used to tune anything.

All 100 direct repeats and all 300 component repeats are exact hash matches; direct repeat tolerance failures 0. Float64 control is supported for all 100 endpoints, with no renderer/loss semantic changes. The sensitivity-only `R_directnorm` was recorded but did not replace `R_cancel` or form a threshold.

## Validation and information boundary

Core tests: 3 passed in 0.56 s. Affected tests: 9 passed in 10.33 s. Final local suite: 9 passed, 1 CUDA test skipped in 10.50 s. A6000 suite: 10 passed in 2.88 s, including shared-versus-separate forward and repeat tests for float32/float64. Command: `python -B -m pytest research_log/T069BN research_log/T069B research_log/T067B -q`.

Independent verifier PASS: independently derives frozen endpoint indices, reconstructs every component and direct gradient using fresh forwards, repeats float32 computations, recomputes the float64 control and independently calculates numerical bookkeeping/distributions. Total 1,200 GPU forwards; exact gradient and output hashes; same audit status. The original acceptance tolerance was not changed.

Only the original 100 development degraded inputs, frozen traces/gradients, model/features and target-free endpoint binding were read. No development reference quality, exposed-transfer inputs/labels, clean images or fresh/final sets were accessed. `reference_reads=0`, `optimizer_runs=0`, `model_fits=0`. No T99, safety labels, PSNR/SSIM, safety ranking or cancellation-signal verdict was produced. Development table freeze SHA256: `6bd82a7115297b38c5da1d420ed1d9f50d5a3f6beb77a6be21ca68999f8e73b7`.

## Run and recovery

Run `20260921-135734-ttie-t069bn-gradient-numerics` on GPU 1, NVIDIA RTX A6000, float32 CUDA path with TF32 off and the existing environment. Primary source-pinned process repeated gradients without seed/settings changes. Start 2026-09-21 13:57:40 +08; primary finished 13:58:14 +08 (29.28 s); independent verifier and wrapper finished 13:59:01 +08; exit 0.

Release: `/media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t069bn-gradient-numerics`; output: `/media/wenchang/F/wjq/TTIE/runs/T069BN-gradient-numerics`. After tests, exact entry points: `python -B -m research_log.T069BN.run --out <output>` then `python -B -m research_log.T069BN.verify --out <output>`. Full environment/commands are in `evidence/run.sh` and source-bound config. Wrapper `meta.json` contains a stale historical releaseId; the explicit working directory and source SHA identify the actual release. Its repository copy is LF-normalized; original bytes remain in the recovery archive.

One read-only SSH log request timed out and succeeded on retry. No experiment failure, rerun, tolerance change or scientific deviation. An initial local inbox-print command encountered Windows GBK encoding and was retried with UTF-8 before implementation.

Raw archive: 1474560 bytes, SHA256 `c67c1bb87ee1413fe387465aec8616c07597a02777f0b330e8380d55fba63769`. Recovery: 1488389 bytes, SHA256 `ff5d3ee5a5b90af972460aad39c0c8d2bd43f1f0e41f4c4eda8a56868b41ae81`; verified locally and preserved on both server filesystems. See archive receipts for exact paths.

Next step: research lead should decide whether a principled consistency criterion/path can be frozen from this development-only audit. This result does not authorize a tolerance change or resuming T069-B transfer diagnosis. Stop after this report.
