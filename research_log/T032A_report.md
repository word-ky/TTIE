# T032-A — DONE

The fixed source-only support trust-region rule fails the predeclared fresh qualification. Mean PSNR falls from **10.203176519928638 to 8.444270860563833 dB**, and mean RGB-SSIM from **0.32282472579872723 to 0.20675635807073392**. Paired mean changes are **-1.7589056593648045 dB / -0.11606836772799327 SSIM**; paired medians are -1.5888744534194759 dB / -0.09724309641095819. Both joint acceptance conditions fail. No threshold was tuned and no second cohort was opened.

## Fixed source support and provenance

Source commit `b033eaa101dd6aa1fb07acf0cfa282bb67f72845`, branch `codex/T032A-support-trust`, PR #57. The radius was separately committed as `b96d513ad91d0a259a6727604d76c5da100d31c9` before fresh inference. Implementation calls the unchanged accepted T026-A `gamma_range_ttt.trajectory`; source file hashes are bound to accepted source `b2359721c89db732d17e03be273e0bdb71bb377a`. No model, feature, gate, bound, learning rate or update-budget change.

Exactly 400 accepted T014 training-bank files supply 7,346 feature rows from 80 source image IDs. All per-bank hashes, IDs and row counts are in `T032A_radius/freeze.json`, bound to the accepted T014 training manifest SHA256 `92fd60d01ca0780880d724464c4d0a76ba1721ab5b8a2d054d4035a141673125`. Energy checkpoint SHA256 `c3d1eef9f20af163e1268823cf16d3fdaed315e7723fc94db3b1138ad1336521`. Source population mean/std exactly reconstruct frozen float32 normalization buffers. Computation promotes the saved features/buffers to float64, as in T031.

For every source row, the GPU calculation masks every row with the same source image ID before minimizing Euclidean standardized 28-D distance / sqrt(28). The single linear 95th percentile is **r_support = 0.646940052237215**. All 7,346 cross-image distances are persisted in `cross_distances.json`, SHA256 `14b72d800b2bc3703ce18725f782b13c90fa4bd1d469fbacae4f7fd210c9546b`; source tensors SHA256 `691eed7c3cc6d75bc215d9d87540c93b5a2ca81740093270af3d6852227ae677`. Radius freeze SHA256 `9b2b96be12194453dc49405f2201a489b39a2fbe9be5b12e74efa5bcbd1ee17d`, completed **2026-09-14T15:27:01.614712Z**. Independent SciPy source replay maximum error 2.220446049250313e-16, radius difference 1.1102230246251565e-16; local independent export/check agrees. No image decode, reference label, T031 AUROC/cosine, real metric, calibration row or source target enters this construction.

## Fresh cohort and isolation

The committed ledger starts from the 689 official training records' existing encoded metadata, excludes the frozen validation100, T023-A's exact16 reference-decoded source pairs, and T030/T031's 100 reference-used qualification pairs. Prior T030 exclusion/decode receipts and accepted task reports account for earlier reference use; low-only baseline smokes do not exclude a pair. This leaves **473 eligible pairs**. The first100 by ascending SHA256(UTF-8(relative_low_path + "|" + low_file_sha256)) are frozen in `T032A_cohort/manifest.json`, SHA256 `f0c3f024fadbfc2aa3ad6616a4ba0e89a5c56536b300877031982f3b5acc53ff`. No official-test member was deployed, decoded or evaluated. These100 pairs are now development/reference-used and must be excluded from future fresh cohorts.

The sole fresh GPU run performs 100 × 40 unchanged Adam lr0.03 updates with identity reset, Region2, frozen gate/head, dark EV[0,2], bright EV[-0.5,0], active gamma[0.5,1.25], seed7 and TF32 off. Every image is active; all4,100 states are retained. The baseline selects minimum predicted energy over0..40 with earliest ties. The support rule selects the minimum over the prefix before the first strict distance>radius exit; step0 exit selects identity. No re-entry, soft weighting or fallback was introduced. Source and spec hashes plus radius equality are checked before low decoding. The scorer and selector have no reference/metric input. No self-reversal or reference gradient was computed.

Both outputs/decisions for all100 images were frozen at **2026-09-14T15:33:03.601370Z**, SHA256 `c3fab11ae98058cf474675df2bf9a6ff131e3dd907a05f8e9ff93845737bdd04`, with100 low opens and0 normal decodes. Independent full replay verifies all4,100 fresh distances to <=4.440892098500626e-16 and all200 decisions exactly before normal deployment. The independently persisted reference-deployment receipt binds that exact freeze before reading any normal archive member; deployment started **2026-09-14T15:34:26.207629Z**. The unchanged accepted evaluator verifies this binding before opening normals. Evaluation completed **2026-09-14T15:35:48.034455Z**. All400 frozen artifact hashes were verified; selected raw/grid tensors match their saved trajectory states.

## Qualification results

| Metric | Original T026-A | Source-support rule | Paired mean delta | Win / equal / loss |
|---|---:|---:|---:|---:|
| PSNR dB | 10.203176519928638 | 8.444270860563833 | -1.7589056593648045 | 13 / 0 / 87 |
| RGB-SSIM | 0.32282472579872723 | 0.20675635807073392 | -0.11606836772799327 | 28 / 0 / 72 |

All100 selections changed. All100 trajectories exit the source radius: no-exit fraction0; step0-exit fraction0.33. Support selects identity for34 images (33 step0 exits and one step1 exit). Original selection is step40 for89 images. Baseline selected-state distance median/max: **1.331540065085921 / 1.835097579571435**. Support selected-state distance median/max: **0.6355688317037358 / 1.0331292952823126**; distances above the radius among selected outputs are expected for the explicit step0-exit identity case.

Exact selected-step and first-exit histograms follow below; complete summaries and per-image paired metrics are in `T032A_result/audit/summary.json` and `metrics.csv`.

## Runtime, tests and failures

A6000 physical GPU1, PyTorch2.4.0+cu121 / CUDA12.1 / Python3.12.12. Release `20260914-232618-ttie-t032a-support`; source-radius job `20260914-232650-ttie-t032a-radius` exit0; sole fresh GPU job `20260914-232831-ttie-t032a-trust` exit0; completed CPU evaluation `20260914-233459-ttie-t032a-eval` exit0. Exact commands and logs are retained in compact evidence.

Radius GPU time0.532524234s, total source computation plus independent replay1.882816587s (excluding process imports/tests). Mean original trajectory time **2.405760545s/image**; incremental support scoring/selection and finite checks **0.146161478s/image**, **+6.075479%**; combined2.551922023s/image. This measures a full shared40-update comparison; it does not claim compute saved by early termination. No truncated trajectory was rerun.

Local original baseline regression:3 passed28.55s. Local support unit/integration tests:4 passed16.30s; remote4 passed2.57s. Tests cover cross-image exclusion/normalization, strict boundary, first exit/no re-entry, step0 identity, ties, radius hash/value mismatches, target-free selector API and reuse of the unchanged baseline trajectory. Independent accepted PSNR and RGB-SSIM evaluation maximum error5.329070518200751e-15. Local separate Torch export and NumPy/SciPy saved-evidence replay are recorded in `T032A_local_verification.json`.

Observed execution failure: the first normal-deployment SSH connection closed(exit255). The orchestrator mistakenly launched CPU evaluator `20260914-233346-ttie-t032a-eval` before checking that failure; it stopped at the missing deployment receipt before installing the normal opener or computing any metric. The successful deployment then preceded the sole completed evaluation. The failed log is preserved, and the premature progress note is explicitly corrected in the project HANDOFF. No GPU rerun, duplicate reference evaluation, scientific-setting change or result-dependent repair occurred. Separately, the first local source-hash comparison encountered Windows CRLF bytes; deployment uses exact accepted Git blobs and all deployed hashes verify. Existing NVML warnings did not prevent actual CUDA execution. No unresolved blockers.

## Evidence and decision

Full F backup `/media/wenchang/F/wjq/TTIE/shared/t032a/T032A_execution.tar`,592,875,520bytes, SHA256 `a69a46c45e5d31bfcbe6408da1f2056128c11b1970718d09b18a4af705408cbe`. Compact evidence2,446,996bytes, SHA256 `9402b8bfc32ebfe9dd7016468d2880ac9fa756753c8687aec9cfc020a749b809`, on both server roots and extracted locally. It contains all states/features/distances/decisions, scalar results and logs; full image tensors remain in the full backup. `guarded.pt` / `guarded_*` are retained T030 evaluator-compatible filenames for the **T032 support rule**, not the rejected self-reversal method.

The fixed source-derived radius is too restrictive to fresh-qualify this exact controller: every trajectory exits by step18 and one third already start outside. This result does not erase T031's diagnostic association or prove all source-support controllers impossible. Do not sweep thresholds on this consumed cohort. Stop and return the negative result to the research lead; T026-A remains unchanged. No benchmark, retraining, second cohort, official test or self-merge.


## Exact histograms (step: count)

Baseline selected step: 20: 1, 32: 1, 35: 1, 36: 4, 37: 1, 38: 1, 39: 2, 40: 89.

Support selected step: 0: 34, 1: 2, 2: 6, 3: 11, 4: 6, 5: 4, 6: 3, 7: 4, 8: 1, 9: 1, 10: 1, 11: 1, 12: 3, 13: 2, 14: 9, 15: 9, 16: 2, 17: 1.

First exit step: 0: 33, 1: 1, 2: 2, 3: 6, 4: 11, 5: 6, 6: 4, 7: 3, 8: 4, 9: 1, 10: 1, 11: 1, 12: 1, 13: 3, 14: 2, 15: 9, 16: 9, 17: 2, 18: 1.

Portable compact replay (separate processes):

```text
python scripts/export_t032a_saved.py --result research_log/T032A_result --out research_log/T032A_download/fresh_replay/arrays.npz
python scripts/check_t032a_saved.py --result research_log/T032A_result --arrays research_log/T032A_download/fresh_replay/arrays.npz --out research_log/T032A_local_verification.json
```

Local replay PASS4100 states /200 decisions; distance maximum error4.440892098500626e-16, metric aggregation error0.

negative/insufficient
