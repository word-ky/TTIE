# T030-A fresh qualification — negative/insufficient

The fixed learned-gradient self-reversal guard fails both predeclared quality conditions on the fresh100-pair cohort. Mean paired RGB-PSNR changes by **-0.206283973 dB** (required >=+0.30) and RGB-SSIM by **-0.007808096** (required >=0). No parameter, threshold, anchor, cohort or fallback was tuned after evaluation. Stop; no promotion.

| Same fresh100 pairs | Mean PSNR | Median PSNR | Mean SSIM | Median SSIM |
|---|---:|---:|---:|---:|
| Original T026-A selector | 10.328656774672057 | 9.661904941173189 | 0.3228187505356958 | 0.3035797104797868 |
| Fixed reversal guard | 10.12237280161248 | 9.501372117983188 | 0.31501065415150425 | 0.29655677277464054 |

| Guard minus original | Mean | Median | p10 | p90 | Win / equal / loss |
|---|---:|---:|---:|---:|---:|
| PSNR | -0.20628397305957497 | -0.03206592662040553 | -0.6939534324731174 | 0 | 5 / 41 / 54 |
| SSIM | -0.007808096384191523 | 0 | -0.02881366512671941 | 0.00022027210204751587 | 11 / 41 / 48 |

These are paired fresh-cohort comparisons, not comparisons against the old validation mean. The conclusion concerns this one fixed proxy/rule on this cohort.

## Frozen cohort and reference history

- The original100 validation pairs are excluded (split SHA256 `b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b`). Among the other589 canonical training pairs, the exact16 T023-A source pairs had normal pixels used for development/training and are excluded. Their committed source receipts contain actual `normal_open_started_utc` records; the16 identities match the source manifest exactly.
- Historical T022-A full-dataset work read encoded hashes and PNG IHDR headers only. Its source writes689 training records before test records; cohort construction accesses only that training prefix. Other real-reference diagnostics used the old validation100. Retinexformer/SNR-Aware smoke tests used lows only and did not require exclusion. The exclusion dossier binds the relevant committed reports, source receipts and project-state provenance. No other nonvalidation normal-use cohort was found in that committed provenance.
- From573 eligible pairs, select the first100 ascending SHA256 of the UTF-8 canonical relative path `Train/Low/lowNNNNN.png`, tie by path, with no seed prefix. The committed exclusion list and complete ordered low/normal byte hashes precede any fresh normal decode. Cohort SHA256 **`ec67f0a6af5682c8e1e929db56e1d771dfd3183f75cb4b365052cde024f55f2d`**.
- Only the100 explicitly named training-low ZIP members were deployed initially. No task normal member was read at that stage; no test-member listing, decode, inference or scoring. After this completed evaluation, these100 pairs are development-used and should be excluded from a later fresh qualification cohort.

## Fixed implementation and input boundary

Scientific source **`d2afd440a11032a9676f92d562804c1831e5d80f`**, branch `codex/T030A-self-reversal`, PR https://github.com/word-ky/TTIE/pull/55. Seventeen accepted transitive source files are byte-bound to T026-A source `b2359721c89db732d17e03be273e0bdb71bb377a`. The trajectory/renderer/gate/feature/energy code is unchanged. Asset receipt binds CLIP/prototypes/gate and T014 energy SHA256 **`c3d1eef9f20af163e1268823cf16d3fdaed315e7723fc94db3b1138ad1336521`**.

`ttie/self_reversal.py:select(energies, gradients, active)` has no image, normal/reference, metric, filename or decode argument. It takes active EV+gamma raw gradients, anchors at10, and visits11..40 until the first cosine<=0, setting cutoff to the preceding state. It then takes the earliest minimum-energy state over0..cutoff. No crossing uses40; a zero-norm anchor or visited comparison (`<=1e-12`) uses the original selector. The inherited no-active T026-A bypass remains supported but was not encountered. The rule file SHA256 is **`f8e9702dbe2f67ec5129344c83f8a8745bfcbf07f9d17f576162d0a47e1f290a`**.

Each image has one original40-update Adam0.03 trajectory from identity. Its41 saved states supply both decisions;41 learned gradients are recomputed without further updates. Frozen Region2, dark EV[0,2], bright EV[-0.5,0], active gamma[0.5,1.25], T014 weights/features, float32, seed7 and TF32off are unchanged. No T029 reference-gradient/cosine artifact, oracle state/metric or per-image target-derived quantity was read as a selector input or staged with this executable.

All200 selected outputs/raw states/grids and100 decisions/trajectories were persisted and hash-frozen at **10:41:25.328834 UTC**, freeze SHA256 **`e94811518388868b229b0e731db207459c27e86c360c739f5979a3eb1234516b`**. The inference decoder logged exactly100 allowed low opens and zero normal opens. Independent NumPy replay matched all100 cutoffs/crossings/fallbacks/selected steps exactly before references. Task reference deployment began **10:42:23.388604 UTC**; evaluation ran10:42:30–10:43:00 UTC using accepted native400x600 full-frame RGB PSNR and Gaussian11/sigma1.5 RGB SSIM conventions. Both decisions were already immutable.

## Selection and cost

There are59 crossings and59 changed selections;41 have no crossing and unchanged outputs. Zero degenerate/fallback cases. All4000 updates,4100 states and4100 recomputed gradients are finite. Cutoffs range15..40, with41 at40. Original selected-step40 count87; guarded count39. Full cutoff/crossing and both selected-step histograms, anchor/comparison cosines and gradient norms are saved per image and in `T030A_result/audit/summary.json`.

Mean trajectory compute2.403834967s/image; added gradient/guard compute2.391305892s/image; combined4.795140859s/image (**+99.48%**). This qualification executes all40 updates before selection and claims no online stopping speedup. A6000 physicalGPU1, Torch2.4.0+cu121/CUDA12.1, accepted seed7/TF32off settings; CPU used only for small selector reductions, metrics and independent aggregation.

## Verification, receipts and failures

- Sole low-only GPU job `20260914-183308-ttie-t030a-guard`, release `20260914-183209-ttie-t030a-guard`:100 pairs,4000 updates, exit0. CPU evaluation `20260914-184224-ttie-t030a-eval`:exit0. No scientific rerun or experimental deviation.
- Focused tests: local3 passed11.98s; server3 passed1.32s. Tests cover first/zero crossing, no crossing, earliest ties, degeneracy fallback, inactive-coordinate exclusion, API argument boundary and denied reference decode.
- All100 independent NumPy selector replays are exact on server and Windows. Server verifies selected raw/grid correspondence and all400 artifact hashes; local replay additionally verifies200 fetched decision/trajectory hashes and28 staged source/metadata files. Generated remote pytest cache is runtime output, not a deployed source file.
- Independent PSNR/SSIM computations on all200 outputs agree within `7.105427357601002e-15`. Independent local paired aggregation, win/equal/loss counts and every histogram agree; maximum statistic difference `3.552713678800501e-15`.
- One backup SSH connection timed out and succeeded on retry. A local source-hash check initially included generated remote pytest cache; excluding that runtime cache allowed comparison of actual staged files. Neither incident changed data, code used for inference, metrics or experiment count. Existing NVML warning did not prevent CUDA; no driver changes.

Evidence is in `T030A_cohort/`, `T030A_source_binding.json`, `T030A_preparation/`, `T030A_result/audit/`, `T030A_result/runs/`, `T030A_local_verification.json` and `T030A_backup.json`. Full200 images, source and execution backup on F: `shared/t030a/T030A_execution.tar`,590807040bytes, SHA256 **`9c0ffd9bde0dddf81c659b0560f31fe57ee979a2d1b9f74de21a632ca83a8779`**. Compact archive2301652bytes SHA256 **`7bfd88591dc2a7d2bd01eee45c60ec7e1d5680b1ebb1e54a0027a82821b9c117`**.

Stop for research-lead review. No second cohort, anchor/threshold sweep, alternate guard, retraining, baseline benchmark or official-test execution; original T026-A remains unchanged and scientific state remains research-lead owned.

negative/insufficient

Delivery note: a local disk-full error interrupted git add; sparsifying completed T022A/T023A duplicate worktrees freed space while preserving task artifacts and Git history. No scientific rerun.
