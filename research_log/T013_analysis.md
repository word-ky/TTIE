# T013 — controlled negative source-stage result

2026-09-12T09:10:00Z. The prescribed learned restoration energy passes **3/7** source criteria. Stage A exited normally; no fresh T013 manifest was created/read, no Stage B was run, and no calibration tuning or experimental rerun followed. This is a development-only result for this frozen model, feature/state-bank recipe and calibration split.

## Run and method

Frozen source+manifest `b6642ad6358045cb60296a71d1be72e2525833d2`; implementation `02933ed665692036ea17bbe4ed9c67e48ca314d6`; branch `codex/T013-learned-restoration-energy`. A6000 release `20260912-163818-ttie-t013-stage-a`; run `20260912-163826-ttie-t013-stage-a`; command `CUDA_VISIBLE_DEVICES=1 bash scripts/run_t013_a6000.sh A b6642ad6358045cb60296a71d1be72e2525833d2`. Physical GPU1, logical cuda:0; unrelated GPU0 job preserved. Exit0 at **2026-09-12T09:05:19Z**.

Official COCO image files only: first80 new eligible IDs23666..33114 train, next20 IDs33638..36539 calibration, excluding all408 previously used IDs. Exact manifest is `T013_source_manifest.json`, SHA256 `101ba5ed775c0a15002fdd3be3a01874b9dbbfb98242784e865de4e2ff9060e7`. All100 now remain permanent development data; future decisive exclusions total508. No annotations or external donor/dependency added.

The original T006/T007 signal and gate, T011 geometry/renderers/direct/discrete/semantic controls, Adam lr.03, action box and degradation definitions remain unchanged. New 28-feature smooth energy receives original active/winner/evidence, differentiable current dark/bright evidence, physical EV/gamma; no reference, label, condition, mask/gain, image ID or annotation. Global state repeats into four slots. Per active input each learned-energy method performs exactly40 updates and selects its minimum predicted-energy saved checkpoint, ties earliest. No-active inputs return exact identity with zero updates. The original semantic fixed16 baseline is never recalibrated.

Training bank: identity, direct, discrete, semantic1/4/8/16/40 and first16 unscrambled Sobol8 points mapped channel-major to legal physical states. Duplicate entries retained. 24 rows per active source input, one per all-inactive input: **400 training episodes / 7,530 rows**, hence310 active and90 all-inactive. CPU one-thread training; MLP28→64→64→1, SiLU, train-only population normalization, constant scales1, Huber delta1, AdamW lr1e-3/wd1e-4, batch256, seed7, exactly100epochs, finalepoch only. Huber loss epoch1 `.19657343623251872`, epoch100 `.026784924789273724`. No validation weight or architecture selection.

Frozen energy SHA256: `43181ee022bfa845d7b3433546a3d8d9899f2b253b4e1296dc826d784119fd47`. It was saved and frozen before any calibration inference. Current CLIP evidence keeps its gradient to the ISP. Calibration uses the head on the ISP device; full trajectories and selections are saved before reference-only evaluation. Full generation precedes selection; no early-exit compute saving is claimed.

## Predeclared source gates

Calibration has20images per condition and40 pooled heterogeneous episodes. Alignment includes every active non-clean episode, in all eight Region2 raw coordinates at identity. Zero-norm vectors count as cosine0/nonpositive. The energy vector is the actual stored first-update gradient; reference log-MSE gradient is computed only offline after label-free persistence.

| Clause | Observed | Required | Result |
|---|---:|---:|---|
| Positive reference-gradient cosine fraction | 54/78 = .6923076923076923 | >=.80 | Fail |
| Median reference-gradient cosine | .2476488006325565 | >=.50 | Fail |
| Clean p95 MSE | .003968007455114277 | <=.005 | Pass |
| Dark MSE / identity | .572854301476039 | <=.65 | Pass |
| Bright MSE / identity | .5844024470315694 | <=.65 | Pass |
| Heterogeneous MSE / projected discrete | 1.2131349179068975 | <=.95 | Fail |
| Heterogeneous MSE / fixed16 | 1.2406140448885508 | <=.95 | Fail |

Clean mean MSE is `.0005608959938399494`. Homogeneous dark/bright gains over identity are42.71%/41.56%. Heterogeneous gain over identity is41.16%; nevertheless primary MSE is **21.31% higher than discrete**, **24.06% higher than fixed16**, **25.70% higher than the old semantic final checkpoint**, and5.30% higher than direct, on this same calibration split. These strong controls prevent interpreting a gain over identity or the weak global-energy control as a qualified TTT gain.

| Method | Clean mean MSE | Clean p95 | Dark MSE | Bright MSE | Heterogeneous MSE |
|---|---:|---:|---:|---:|---:|
| Identity | 0 | 0 | .070901487 | .035500405 | .052393962 |
| Region2 direct | .000358014 | .002195431 | .046191182 | .014952537 | .029280140 |
| Region2 projected discrete | .000225017 | .001926385 | .038503298 | .017098331 | .025414366 |
| Original projected semantic final | .000243326 | .001614168 | .037718985 | .016955567 | .024528346 |
| Frozen semantic fixed16 | .000225839 | .001609583 | .037955826 | .016938808 | .024851448 |
| Global energy | .001887542 | .011553455 | .038109522 | .022812450 | .056602867 |
| Bilinear2 energy | .000426701 | .002108950 | .041558870 | .018073571 | .036969740 |
| Region2 energy (primary) | .000560896 | .003968007 | .040616222 | .020746524 | .030831055 |
| Reference-only energy checkpoint oracle | ~0 | ~0 | .039776644 | .017563133 | .028961224 |

Exact machine-readable per-case and aggregate MSE/PSNR, regional metrics, energies, selected steps and raw diagnostics are in the run's `metrics.json`, `summary.json/.md`, `alignments.json` and episode directories. Active zero-state rendering can have tiny floating-point drift; exact all-inactive bypasses are separate.

## The available trajectory is the main limitation in this experiment

The reference-only oracle takes the minimum actual MSE across the same primary energy checkpoints for each image, after label-free decisions are persisted. Its heterogeneous MSE is `.028961223550140858`; learned/oracle regret is **1.0645632855672291** (6.46% higher MSE). Even that oracle is:

- **13.96% worse than discrete**, ratio `1.1395611086291644`;
- **16.54% worse than fixed16**, ratio `1.1653736905152772`;
- **18.07% worse than the old semantic final**, ratio `1.1807246775470865`.

Both required oracle-versus-baseline5% diagnostics are false. Since every selector over these saved states has MSE at least the per-image oracle average, fixing checkpoint selection alone cannot clear the required restoration margins here. Combined with54/78 positive gradients and median cosine.248, the result does not support the hypothesis that this trained energy creates a better restoration trajectory. Fitting source scalar values with this bank/feature/SiLU-MLP recipe did not establish useful enough gradients on the new calibration images.

This conclusion is limited to the fixed T013 recipe and this source calibration split. It is not a universal impossibility result for learned energies, nor a held-out benchmark result. Do not compare absolute MSE across different T011/T012/T013 splits as a method improvement. No criterion was relaxed and oracle performance is not an alternate pass route.

Recommendation to the research lead: accept the controlled negative result, then decide explicitly whether to investigate derivative supervision, feature sufficiency or source-state coverage. These remain hypotheses, not diagnoses proved by this run. Do not tune the inspected20images or silently continue a selector-only repair, detector, meta-initialization, ViT3, prompt or learned-basis module.

## Software validation and disclosed implementation details

Baseline105tests pass49.209s. Core4pass10.289s, existing training regression2pass10.454s, evidence3pass11.804s, tiny source/calibration pipeline3pass35.566s. Full **115 local tests pass86.125s** and **115 A6000 tests pass14.908s**, including energy CPU/CUDA score/decision/gradient fixture. Original calibration scores match bitwise. Tests cover live evidence/state derivatives, fixed state bank, geometry/inactive identity, exact updates and earliest ties, frozen training/serialization, metadata/reference replacement, offline gradient diagnostics, seven/eleven gates, and source/fresh receipt barrier. The tiny pipeline is software validation, not research evidence.

Two pre-run failures were corrected and retained: an incorrect inactive-region slice in a16x18fixture, and an old evaluator default discrete-method name in the new Stage-B adapter. Neither repair changed a scientific threshold or inspected real outcomes. No experiment runtime failure or rerun occurred. No scientific code changed after b6642ad.

Post-run audit initially assumed every training-bank image had exact inactive pixels. The prescribed `region2_direct` entry retains the original T011 direct-control renderer, which uses identity parameters at inactive nodes but has no exact pixel mask. Its floating-point identity roundoff therefore differs from projected Region2 and the new energy methods. The audit replays that exact historical direct renderer, records its inactive differences separately, and retains strict bitwise inactive checks for every other bank entry and all primary energy checkpoints. It finds422 nonzero source-direct inactive-region differences, maximum **5.960464477539063e-08**; every direct image matches the historical renderer exactly and every inactive direct state is EV0/gamma1. It does not alter any experiment image, target, state or model. Full exception entries are in `inherited_direct_roundoff.json`; the initial assertion/diagnosis is retained in `T013_verification_initial_failure.txt`. This qualifies the earlier shorthand that all bank states used the masked renderer.

Known NVML mismatch warning and CUDA bicubic backward nondeterminism remain unchanged. Actual CUDA worked; no strict bitwise repeated-CUDA-trajectory claim is made. StageB is implemented and fixture-tested but has no real-data result because StageA failed.

Five fixed first-calibration-ID **33638** panels were inspected; all10tiles/labels are intact. No favorable-image selection or additional images were used. Recovery records remain project-local and mirrored under the remote project root, separate from the tested release.

## Completed remote evidence audit

All **3,000 file hashes** verified. All stored source-bank and calibration checkpoint/output MSE values recompute exactly from saved pixels; all28-feature vectors, train-only normalization, serialized-energy GPU scores/selected checkpoints, alignment cosines and final source summary reproduce exactly. The frozen energy/recipe/bank/source/model/gate identities are unchanged.

Counts:400 source training episodes,7,530 bank states;100 calibration episodes,10,020 energy checkpoints and9,720 energy updates across the three learned-energy geometries;2,789 old semantic checkpoints. Each energy method has81 active episodes at40updates and19 all-inactive bypasses; primary calibration has3 active clean and78 active non-clean episodes. **14,160 strict inactive-region checkpoint checks pass**, excluding only the separately replayed/reported legacy direct bank entry.

Full large float32 image packs, **80,026,086,548 bytes**, remain at `/home/wenchang/asdasdsad/wjq/TTIE/runs/20260912-163826-ttie-t013-stage-a/artifacts/audit/`. Exact relative paths/byte sizes/SHA256 hashes are in each bank/episode receipt and the artifact manifests. The receipt archive excludes only `bank_images.pt`, `checkpoint_images.pt` and `outputs.pt`; all raw states, features, scores, full gradients, projection traces, energy weights/training history, decisions, targets/metrics, figures, environment/test/run logs and verification records are retained in the small evidence tree. No output packs were deleted.

Local evidence destination: `research_log/remote_runs/20260912-163826-ttie-t013-stage-a/`. Final receipt archive SHA256: `f1b553880a43f5f74cc5a1d500241059577a632e869a5fb79e1f8ca0e204a147`, exact match remote/local before extraction. **1,700 local small-file hashes** and the energy hash match, all900 calibration rows and78 alignment entries match their episode files, and the complete source summary recomputes with maximum absolute difference **5.551115123125783e-17**, preserving the same verdict. Large packs remain verified remotely. Receipts: `output_verification.json`, `local_verification.json`.

## Additional reporting requested by the research lead

Interim review main `9a847735c2a53e1be50ff6f094fcc6ae8fdea36e` accepted the frozen implementation and requested gradient-cosine and selected-step distributions. `summarize_t013.py` computes them solely from the saved run; full per-condition and all-three-method results are in `distributions.json/.md`. No model, trajectory or criterion changed.

Across78 active non-clean episodes there are54 positive,24 negative and0 zero cosines. Mean `.21138412638500492`; min/max `-.9569045485274166` / `.9874440244351873`; P10 `-.6131000503721937`, P25 `-.14838803743895732`, median `.2476488006325565`, P75 `.6735466077677272`, P90 `.902599307897006`. Counts in bins `[-1,-.5)`, `[-.5,0)`, `[0,.5)`, `[.5,1]` are **10,14,28,26**.

Primary selected-step histograms (`step:count`,20inputs per condition):

- Clean: `{0:17,13:1,36:1,40:1}`;17 no-active bypasses.
- Dark: `{0:1,12:2,13:1,14:1,20:1,23:1,24:1,26:1,27:2,28:1,32:1,33:1,36:1,40:5}`;1 no-active bypass.
- Bright: `{0:1,12:1,22:1,27:1,29:2,30:1,31:1,33:1,34:2,35:1,37:2,38:1,39:1,40:4}`;1 no-active bypass.
- Left/right: `{12:5,13:1,20:1,23:1,24:1,28:1,30:1,32:1,39:2,40:6}`;0 bypasses.
- Quadrants: `{14:3,18:1,23:1,24:1,25:1,26:2,27:1,29:2,31:1,32:1,33:1,34:1,36:1,38:1,40:2}`;0 bypasses.

All primary step0 selections here are original no-active bypasses, not learned rejections of active inputs. The three active clean cases select13/36/40. This is descriptive and does not alter the source gates.
