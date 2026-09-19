# T060-C-R1 fixed gain-slice substitution

**T059-E gain-direction advantage does not translate into a sufficiently safe/material fixed T036 trajectory improvement**. This is one fixed probe on the already-open T036 development cohort. It does not establish official-test performance or authorize tuning.

Authorization023402b13f36ed77d64e24f48d30adc67416691b; tested source 92c8bdd38e51b7d94c9a16cbd2e5c729dd382e52; accepted T036 source f80cea4c9d8186e0c4a0404b28ccd58c5e1b5678. Branch codex/T060CR1-gain-slice. Sole run 20260919-140505-ttie-t060cr1-gainslice; physical GPU1 (NVIDIA RTX A6000). Five focused/baseline/firewall tests passed4.15s. Full inference/evaluation/verifier exits0.

## Exact fixed intervention

Reuse raw-low/identity1x3x2x2 CommonRegion2, frozen gate, fresh12-coordinate Adam lr.03,40active updates, CommonBox and minimum T014 scalar energy/earliest tie. An AST substitution changes only the original autograd gradient call. T014 EV/gamma entries stay exact; the4gain entries are replaced by fresh current-image28-D feature/J_gain and frozen T059-E J-transpose-q. The T059-E scalar never enters selection, stopping, weighting or calibration. Each inactive episode would retain original identity/no-update behavior.

T014 checkpoint c3d1eef9f20af163e1268823cf16d3fdaed315e7723fc94db3b1138ad1336521; T059-E checkpoint e15e91c4e9be401bcac6d3de039ec2d1ae7141fb6cec40c388d22549652d43d0. Own normalizations/loaders remain unchanged. T014 uses its accepted CUDA path; E q retains the accepted CPU path; current image/J and joint optimizer use GPU. All261 source bindings match published Git bytes. Model/assets/input hashes and both head state hashes remain unchanged.

Synthetic prerequisite: exact identity,12coordinates,40updates; EV/gamma equals literal T014, gain matches direct E autograd and independent J-transpose-q; all states replay through Adam/CommonBox; T014 alone selects the checkpoint. No new gate, within-coordinate blending, optimizer alteration or alternate head.

## Same-cohort comparison

Exact100 rows/order and low bindings from accepted T036 manifest SHA279c74b335999d6631d436c79ea80e9e2cccfc4b97cb7c0b992829d04cfaf40b. The T026 baseline here is the exact persisted baseline evaluated on this T036 cohort, not the different original T026 development cohort.

| Quantity | Result | Fixed requirement |
|---|---:|---:|
| Mean paired PSNR vs T026 | 0.9206769979376332 dB | >=0.80dB |
| Median paired PSNR vs T026 | 0.699178881681461 dB | descriptive |
| PSNR improve/regress/tie vs T026 | 80/20/0 | regress<=20 |
| Worst paired PSNR vs T026 | -3.5784443917392466 dB at index33 | >=-3.0dB |
| Mean RGB-SSIM vs T026 | 0.006591908266098638 | >=0.006 |
| Mean paired PSNR vs T036 | -0.018580069748978866 dB | descriptive |
| Mean RGB-SSIM vs T036 | -0.0017641362083923476 | descriptive |

Exact gate verdict: {"mean_psnr": true, "regressions": true, "worst_regression": false, "mean_ssim": true}. Absolute means: T026 10.290783069326087dB/0.33794627217795875SSIM; T036 11.230040137012702dB/0.34630231665244976SSIM; new 11.21146006726372dB/0.34453818044405743SSIM. Complete100 paired rows and mean/median/p10/p90/min/max/counts are in evidence/metrics.json and result.json.

## Freeze and independent verification

Inference start 2026-09-19T06:05:15.197995+00:00; all100 selected outputs/actions/states/decisions frozen 2026-09-19T06:17:18.665065+00:00; separate reference/baseline evaluator entered 2026-09-19T06:17:21.512115+00:00. All100 low opens are bound/in order. Before freeze: normal/reference, reference gradient, per-image baseline outcome, official-test and inference-leakage reads all0. No baseline output/decision or normal is an inference input. Only afterward are exact persisted T026/T036 outputs and development normals opened. Total joint optimizer steps 4000; no retraining, rerun, tuning or official-test access.

Independent verifier checks every coordinate substitution and chain, explicit normalized SiLU E derivative, all100 NumPy Adam/CommonBox trajectories, T014 energy-only selected step, selected raw/grid and exact rendered image hashes, cohort/input/source bindings, freeze order and independent PSNR/RGB-SSIM/scalar tail summaries. At fixed indices0/25/50/75/99 and steps0/20/39 it freshly recomputes literal T014 and direct E autograd gradients. Maximum Adam state error 3.904619007011334e-07, representative direct-gradient error 4.76837158203125e-07, independent metric error 5.258016244624741e-13. Tolerances were frozen in source before the real run.

No execution failure/repair or inference rerun in the completed run. Preparation twice exceeded Windows SSH argument length before dispatch; transfer was reduced to raw-text compressed JSON and reused the already-remote manifest. The prior T060-C contract report is preserved separately; its server home/F copy was repaired before this real run. Existing library deprecation warnings do not change the scientific result.

Full selected output images are retained in the home/F raw archive; Git contains all compact trajectories/gradient fields/decisions, complete paired metrics, provenance, manifests, timestamps, tests, verification and run command/log. Result SHA 241f78d64af9e5ca858fcd834e3e69a03ec5e41c78334efe22ded0368ae24651. Stop for research-lead review; apply the fixed verdict without tuning this opened cohort or accessing the official test.
