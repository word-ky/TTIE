# T041-A — DONE: real selected-state field deficit beyond source high-gain supported

**REFERENCE_GRADIENT_DIAGNOSTIC_ONLY.** At fixed gain1.75, all100 real selected states have total-group positive-dot **0.37** versus frozen source **0.796633554084**, a **42.663355408 percentage-point deficit**; real median cosine **-0.185847992683** versus source **0.763459378857**, a **0.949307371541 deficit**. Both exceed the predeclared20pp/0.25 thresholds. The single primary verdict is **real selected-state field deficit beyond source high-gain supported**. The exact task-decimal constants define the gate; matched-source descriptive deltas use the full-precision accepted summaries. No thresholds were fitted.

This supports a real target/selected-state distribution effect beyond common gain value alone. It does not isolate image-domain content shift from selected legacy EV/gamma/feature-state extrapolation. The populations also differ:100 real selected states versus7346 canonical source states (7248 nondegenerate); source states are correlated within80 source images, so this is a prescribed descriptive diagnostic, not an independent-sample significance claim.

## Reuse and provenance

Executed/tested source **63e0cb7770c5f2f7c4d85244cf7a83a5a065e87a**, branch `codex/T041A-fixed-gain-real-state`, PR [#66](https://github.com/word-ky/TTIE/pull/66), baseef4969ef after accepted T040 mergeeabe742e333f97a0706117ce92abfc3b238964d1. Evidence/main-mailbox SHAs are retained in final delivery. Reuse T038 low-only original-selected-output preflight and FixedObjective gate verification, T036 model loading, T039 common-gain raw conversion, and T029/T038 active-coordinate alignment/aggregate conventions. No deployable module is changed.

Exact T036100 cohort SHA **279c74b335999d6631d436c79ea80e9e2cccfc4b97cb7c0b992829d04cfaf40b**, accepted T036 freeze SHA **46e667ded785e4f3f8ba341d40d00a7e02ca652e2778fcc7ead1750665161be4**. Source accepted audit is `/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-035341-ttie-t036a-common/artifacts/audit`. Its config hash and assets are checked against the accepted freeze; each common decision/output/trajectory/field file is checked against the accepted per-image hashes. Each original selected step must match both frozen metadata and the earliest target-free minimum-energy score, without making any new choice. Source bindings are retained for every deployed code/config file used by Stage A; exact Git bytes were deployed.

- clip: `1bd3c7172de5b207ceac554f5ab5266166f3b9baccc9af5989bc801016d080ad`
- energy: `c3d1eef9f20af163e1268823cf16d3fdaed315e7723fc94db3b1138ad1336521`
- prototypes: `b4b32dbd96c65dcf606ee38d7450ebf348f5731823503b9c71ba15ec78217ac7`
- gate: `b7458c51466fd89c1bfa8e7c4bf1dd820ddf56e33c12c406f6625ac64361a4ce`

Only the selected legacy EV/gamma raw channels and exact gate are reused. Each active cell's gain is replaced with the spatially uniform prescribed1.25 or1.75 through exp(log2*tanh(raw)); inactive gain stays1. No image selection, optimizer, interpolation, sweep, new cohort or source-state resampling. Exactly100x2=200 new probes. The original selected output is reconstructed for parity only and receives no new reference/gradient audit. Every gate score/active/winner/evidence tensor matches accepted T036 exactly.

Accepted source comparisons are hash-bound, never recomputed:
- Gain 1.25: `research_log/T039A_result/stage_b/summary.json`, SHA `6607b88f26b17cda18e6c34866d2f56b555e5164bdb8a6753dc8c3e607c43fdf`
- Gain 1.75: `research_log/T040A_result/stage_b/summary.json`, SHA `50abeaeb1a6d9bfc0c7c5a6cd5752f1bdf39001873b88c0397beaae6c0693799`

Stage A binds the comparison metadata but does not open those source summary files; Stage B checks both full source-summary hashes/values after the complete low-only freeze. Historical real PSNR/SSIM, metrics.csv and loss-ID files are not opened in either stage. The optional29/71 grouping is intentionally omitted; all100 states determine the sole verdict.

## Freeze and reference boundary

All100 original selected outputs are **bit-exact**, max absolute error **0.0**, before the first probe-gradient call. Stage A run **20260915-122826-ttie-t041a-stage-a**, release20260915-122752-ttie-t041a-fixed-gain, used A6000 physicalGPU1 and took **21.558654s**, exit0. It saved all200 native outputs, raw states, features, energies, g_E, active masks/gates and output hashes. Entire evidence froze **2026-09-15T04:28:57.516308+00:00**, SHA **b2612273d37eb982c40736d5d3434cfa717666a8cad70e18574f41968b08d64b**. Before freeze: normal decodes0, prior real metric/loss-ID file opens0, optimizer updates0, selection changes0. Frozen scorer/energy parameters and normalization remain unchanged, all tensors/scalars finite, raw states unchanged during gradient calls.

Only after freeze, Stage B run **20260915-123014-ttie-t041a-stage-b** started **2026-09-15T04:30:22.624879+00:00**; first normal **2026-09-15T04:30:23.358999+00:00**; completed **2026-09-15T04:30:29.276536+00:00**, duration **6.651865s**, exit0 on GPU1. It accesses only the same100 T036 normals already reference-used by T037/T038 under shared/t036a/normal and their lows. Exact accepted native float32 RGB decoding is reused, with float64 RGB-MSE accumulation for isolated reference gradients. **All200 output hashes and tensors match Stage A bit-exactly**; all100 Stage-A tensor hashes remain unchanged afterward and at archive. No reference quantity changes any state/output. No official-test access, fresh cohort, retraining, controller or deployable decision change.

## Matched-gain alignment

Use active raw-coordinate legacy EV+gamma, common gain and disjoint total masks. Float64 dot/norm/cosine, norm<=1e-12 degeneracy, null undefined cosine and nondegenerate denominators exactly follow T029/T038. Total-dot additivity is independently verified. All200 real probe rows are nondegenerate in every group. Positive dot indicates locally restorative negative-energy direction for RGB-MSE, not a finite-update performance guarantee.

| Population | Gain | Group | Count / nondegenerate | Median cosine | Positive dot | Median energy norm | Median reference norm | Median dot | Degenerate fraction |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| source frozen | 1.25 | legacy | 7346 / 7248 | 0.954278283516 | 0.965921633554 | 2.59919427848 | 0.074690669055 | 0.167884547461 | 0.013340593520 |
| source frozen | 1.25 | gain | 7346 / 7248 | 0.862191500210 | 0.907836644592 | 0.493454178511 | 0.0321444272161 | 0.00938045858759 | 0.013340593520 |
| source frozen | 1.25 | total | 7346 / 7248 | 0.916273820714 | 0.966887417219 | 2.66521904581 | 0.0815803350878 | 0.178854153489 | 0.013340593520 |
| real selected | 1.25 | legacy | 100 / 100 | -0.116235892004 | 0.400000000000 | 0.453220567446 | 0.0525924206697 | -0.00241681359744 | 0.000000000000 |
| real selected | 1.25 | gain | 100 / 100 | 0.371146417032 | 0.700000000000 | 0.242943370734 | 0.0278079387452 | 0.00278808788666 | 0.000000000000 |
| real selected | 1.25 | total | 100 / 100 | 0.076330882344 | 0.530000000000 | 0.542068759143 | 0.05942715301 | 0.00212050568124 | 0.000000000000 |
| source frozen | 1.75 | legacy | 7346 / 7248 | 0.770273713456 | 0.796495584989 | 2.53679635582 | 0.0666604063449 | 0.0915295428161 | 0.013340593520 |
| source frozen | 1.75 | gain | 7346 / 7248 | 0.612039101902 | 0.730546357616 | 0.210052226082 | 0.0107693483939 | 0.000695938704835 | 0.013340593520 |
| source frozen | 1.75 | total | 7346 / 7248 | 0.763459378857 | 0.796633554084 | 2.54891968883 | 0.0675701664371 | 0.0919711045495 | 0.013340593520 |
| real selected | 1.75 | legacy | 100 / 100 | -0.220698296054 | 0.370000000000 | 0.54647087393 | 0.0559062666342 | -0.00415364162441 | 0.000000000000 |
| real selected | 1.75 | gain | 100 / 100 | 0.309086393051 | 0.710000000000 | 0.117068921558 | 0.0122590193774 | 0.00027922770186 | 0.000000000000 |
| real selected | 1.75 | total | 100 / 100 | -0.185847992683 | 0.370000000000 | 0.553232202823 | 0.0570827620577 | -0.0033492553553 | 0.000000000000 |

| Gain | Group | Real-minus-source positive dot (pp) | Real-minus-source median cosine |
|---|---|---:|---:|
| 1.25 | legacy | -56.592163355 | -1.070514175519 |
| 1.25 | gain | -20.783664459 | -0.491045083177 |
| 1.25 | total | -43.688741722 | -0.839942938371 |
| 1.75 | legacy | -42.649558499 | -0.990972009509 |
| 1.75 | gain | -2.054635762 | -0.302952708851 |
| 1.75 | total | -42.663355408 | -0.949307371541 |

![Matched fixed-gain total alignment](T041A_result/matched_gain_alignment.png)

At gain1.25, real total positive-dot53%/median cosine0.076331 are already much worse than source96.6887%/0.916274. At1.75, real legacy is only37% positive with median cosine-0.220698, whereas real gain remains71% positive with median0.309086. Thus the all100 fixed-gain total deficit is not an all100 gain-specific failure; the legacy field is substantially misaligned in these selected states. This does not contradict T038's different question about the original29-loss subset and original spatial gain states.

Full means/p10/p90, norm/dot medians and zero/degenerate fractions are retained in summary.json; every raw state/gradient/mask/group scalar and output hash is retained in states.json. Full features/outputs and g_E remain in Stage-A tensor archives. Independent stdlib replay imports no main alignment/summarize/classify path, reconstructs masks from separately frozen gates, checks all200 identities/selected steps, recomputes600 group vectors, all aggregates, source-baseline hashes and deltas, and the single primary verdict. **4492 scalar checks**, max error **1.9506618542664e-13 <=1e-6** (includes comparison of full-precision baseline values to task-rounded constants). Baseline3tests passed9.75s; T0412local tests7.08s and2server tests1.38s passed. No further scientific modification after reference access.

## Failures, artifacts and next step

No GPU stage failures or restarts; one Stage A and one Stage B. Initial local test command was issued before the new sparse worktree was checked out, so it found no files and ran no tests; checkout fixed that mechanical preparation issue. No disk-full issue in this task. The existing NVML warning is nonblocking. No scientific deviations or extra probes. Optional29/71 analysis is not implemented, as permitted.

Full F archive **578078720 bytes**, SHA **910a733e2fe60036ddaf8a8c238c551177996698c437a985c23255f7590b1f68**; compact evidence **198036 bytes**, SHA **4db7447e98b5bcfa693905c95cddbd625d8470245aa89262b60cbd8fb77ca93e**, both server roots and locally. All exact commands, environment/run logs, original-parity/freeze/reference receipts, vectors and plots are retained; Git-exact source/evidence/mailbox recovery is mirrored locally and in home/F shared/t041a. Stage-A command is in stage_a_run/run.sh; Stage-B command in stage_b_run/run.sh; independent check: `python research_log/T041A_audit/replay.py research_log/T041A_result`.

Stop after this audit. Recommend research-lead review of the real selected legacy/feature-state distribution before choosing any repair; do not infer a pure content-domain cause or automatically retrain/design a controller from this diagnostic. No new experiment is launched in this cycle.

real selected-state field deficit beyond source high-gain supported
