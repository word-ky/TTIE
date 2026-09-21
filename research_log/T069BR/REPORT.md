# T069-BR: GRADIENT_CANCELLATION_SIGNAL_ABSENT

The frozen float64 endpoint gradient-cancellation statistic does not identify the residual unsafe endpoint. Development-only nearest-rank `T99_cancel=0.5881467784474902`; unsafe strict-above-threshold 0/1, safe strict-above-threshold 4/99. The predeclared classification is SIGNAL_ABSENT. Close this exact statistic; no guard or second gradient statistic was implemented.

Source: `a6cd7e4074521338dc3a5a9507cca89a24cf5804`; branch `codex/T069BR-float64-cancellation`; authorization `9458d5f233204077d84a1d636eb749d474938cab`. Review task-owned `research_log/T069BR/**`; the stacked historical diff is not a merge recommendation. Files: frozen authorization/plan, core/tests, run/verifier, source/evaluation bindings, report and evidence. Reuses exact T062 losses/CommonRegion2, T069BN float64 computation and T069B freeze/label flow; no new dependencies.

## Results

| Cohort | Minimum R | Median R | p95 R | Maximum R |
| --- | ---: | ---: | ---: | ---: |
| development | 0.0266466047286 | 0.131443088045 | 0.521550733165 | 0.676873753822 |
| transfer | 0.0308280326701 | 0.150968656506 | 0.537227601155 | 0.724526231589 |

Development above threshold: 1/100. The sole unsafe transfer endpoint is index 86, `k_FS=9`, `k_lambda=21`, existing margin vs T026 `-6.995482992779127 dB`. `R_cancel=0.1564035401013536`, descending rank46, empirical percentile0.55, strict flag false. The threshold and score were fixed before this existing label was read.

| Unsafe endpoint weighted gradient | L2 norm | SHA256 |
| --- | ---: | --- |
| spa | 0.02079376651042835 | `7098e748c0713dd912c53f44b351d99d1f4d5aceb1940a7c7551651a3503134b` |
| 10*exp | 0.5320584608040427 | `b518b85417123ddba1484f01e2edba9022abaf4e69068b27b073c8cf1eba4a18` |
| 5*col | 0.02890641287301272 | `b0a60f5bd313a3aa4c3945823245b08337799eb1836407cea31037cb8db7226c` |

Summed-gradient norm `0.49076952937761176`; sum of component norms `0.5817586401874838`. All12-dimensional component/direct/stored gradient vectors, hashes, residuals and endpoint bindings are in the frozen tables and result evidence.

Safe flags (index, R, descending rank):
- 3, 0.5981004770154987, 3
- 55, 0.5946282760712551, 4
- 68, 0.6590836060702259, 2
- 81, 0.7245262315894083, 1

## Numerical and independent checks

All 200 float32 direct total gradients match stored trajectory gradients exactly: max absolute residual0. The original `2e-7 + 2e-5*abs(trace)` path criterion is unchanged. All 200 float64 component-sum checks pass the research-lead-frozen `1e-12 + 1e-10*abs(direct64)` criterion. Primary max sum/direct residual `2.5951463200613034e-15`; independent max `2.6020852139652106e-15` (different float64 accumulation order). No tolerance was swept or relaxed.

Independent verifier PASS: 1,000 GPU forwards across100development+100transfer endpoints, independent endpoint selection reconstruction, fresh forwards for each float64 component and direct gradient, plus float32 direct/trace check. Component gradients reproduce exactly; max score error `3.3306690738754696e-16`; T99 error0; exact strict flags and same classification. State/input/float32 output/model/rule hashes and freeze boundary verified.

Tests: core3 passed4.56s; final local9 passed,2CUDA-skipped17.76s; remote11 passed2.72s including CUDA pinned endpoint adapter and float32/float64 shared-versus-independent/repeat tests. Command: `python -B -m pytest research_log/T069BR research_log/T069BN research_log/T067B -q`. Full diagnosis and independent verifier completed once, exit0.

## Information boundary and execution

Development100 table frozen 2026-09-21T06:54:32.182725+00:00 with `reference_reads=0`; SHA256 `b32f16f03f255517a96b9e2deba1222d510b069bcb4f86872c13913afa94a588`. Transfer100 table frozen 2026-09-21T06:54:42.031673+00:00 with `reference_reads=0`; SHA256 `e514318cea054dcfd1c5a9026d4db113eff9bd662871742fc6bd9f18762d21b4`. First transfer reference-quality-read marker 2026-09-21T06:54:42.058948+00:00. Evaluation-bound artifacts were hashed/read only after that transfer freeze. Existing T067C endpoint margins and independent T066B labels were joined afterwards. No new PSNR/SSIM evaluation, no clean/reference images, no development quality, no fresh/final cohort access.

`optimizer_runs=0`, `model_fits=0`; original float32 adaptation trajectory/states/selection unchanged. Float64 is only the authorized diagnostic computation. No component subset/reweight, threshold change, selector or guard.

Run `20260921-145411-ttie-t069br-float64-cancellation`, GPU1 NVIDIA RTX A6000, TF32off. Start2026-09-21 14:54:16+08; primary complete14:54:42+08 (20.06s); independent verify complete14:55:12+08; exit0. Release `/media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t069br-float64-cancellation`; output `/media/wenchang/F/wjq/TTIE/runs/T069BR-float64-cancellation`. Commands after tests: `python -B -m research_log.T069BR.run --out <output>` then `python -B -m research_log.T069BR.verify --out <output>`. Full pinned environment/source command in `evidence/run.sh`. Wrapper metadata has a stale historical releaseId; explicit working directory and source-bound config identify the actual release. Repository metadata is LF-normalized; original bytes remain in archive.

No failures, retries or deviations in this task. Raw archive 2252800bytes SHA256 `f1bc0c0039641fa54fab8097061befdd33482ea3e79a6909105fa1e3170ad2e7`; recovery 3470605bytes SHA256 `e5b8ade9f1c68d6f0708c2e74018d66f886eee5ff1a073615441b19363ada809`. Verified locally and preserved on both server filesystems; exact paths in archive receipt.

Next step: research lead review of this negative diagnosis. This result closes the exact frozen endpoint symmetric gradient-cancellation statistic; it does not rule out all possible gradient mechanisms. Stop with no second statistic, guard or qualification run.
