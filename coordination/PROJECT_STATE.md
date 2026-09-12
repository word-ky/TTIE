# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Processing for Task-Oriented Vision**

## Core scientific question

Can a vision system adapt a compact spatial correction field per test image, without test labels or clean targets, so that spatially heterogeneous nuisance degradation is corrected before a frozen downstream task model processes the image?

## Current hypothesis

A useful spatial TTT system requires four aligned pieces: (i) a content-safe nuisance signal, (ii) a label-free inner objective whose gradient field produces restoration-useful states, (iii) safe projected action/stopping geometry, and (iv) a spatial representation whose geometry can be chosen safely per image. T014 establishes the first three for canonical hard Region2. T015 rules out simple routing among global/bilinear2/Region2 because that three-output set has only 1.94% oracle headroom. T016-A shows strong development-only headroom from image-dependent hard boundary placement while keeping the four Region2 actions fixed. T016-B shows the frozen T014 energy cannot rank those boundaries. T016-C now shows that two small **pointwise absolute-value** OOF probes—28-D and 28-D plus explicit boundary coordinates—also fail all five safe-selection clauses. The immediate unresolved question is whether this is a representation failure or a **training-objective mismatch**, because deployment needs within-image ranking rather than absolute cross-image restoration value.

## Best fresh-validated method

The best deployable result remains **T014 Sobolev Region2 TTT**:

1. frozen T006 CLIP exposure readout and T007 clean-abstention gate;
2. frozen T014 28-D Sobolev restoration energy;
3. canonical hard Region2 projected EV+gamma state, identity initialization;
4. exactly 40 label-free projected updates when active;
5. minimum predicted-energy checkpoint, earliest tie;
6. all label-free trajectories/decisions persisted before clean-reference evaluation.

Test-time adaptation never consumes test labels, clean targets, condition IDs, degradation masks/gains, annotations, image IDs as shortcuts, source-only reference gradients/Jacobians, or evaluation metrics.

## Established controlled findings

- **T001–T003:** spatial ISP adaptation can outperform global correction on favorable heterogeneous shifts, but capacity alone and simple absolute-statistic priors are insufficient.
- **T004–T007:** zero-shot exposure signals are content-confounded; a source-trained frozen-CLIP exposure readout plus clean-abstention calibration yields a usable nuisance signal.
- **T008–T011:** semantic EV+gamma TTT has useful directions but suffers clean drift, over-correction, stopping and spatial-coupling issues; projected Region2 becomes strong but does not freshly qualify.
- **T012:** learned checkpoint selection cannot rescue the T011 trajectory; even the reference-only checkpoint oracle lacks required headroom.
- **T013:** scalar source-trained restoration energy is a controlled negative; fitting restoration value does not sufficiently constrain the test-time derivative field.
- **T014:** source-supervised Sobolev restoration energy is the first fully qualified learned-inner-objective result. Stage A passes **8/8** development clauses and frozen Stage B passes **12/12** fresh clauses on 40 unseen images. Fresh heterogeneous MSE is `0.03385803`, 19.17% below matched value-only, 11.72% below projected discrete, and 8.31% below semantic fixed16. Unseen-calibration gradient alignment is `73/74` positive with median cosine `0.93606`, versus `59/74` and `0.42464` for value-only.
- **T015:** frozen routing among global/bilinear2/Region2 is a controlled fresh negative (**4/10**). Routed spatial MSE `0.03780638`; best fixed Region2 `0.03504357`; oracle among the same three outputs `0.03436452`, only **1.94%** better than Region2. A smarter selector over the same three outputs is not justified.
- **T016-A:** fixed-action renderer-transfer is a positive development-only capacity diagnostic. Twenty-seven predeclared shiftable/soft renderers reuse the same selected four Region2 EV/gamma corners. Per-image reference oracle spatial MSE `0.03250770 = 0.92764×` Region2 and `0.94597×` the T015 three-basis oracle. `113/120` oracle selections use hard (`tau=0`) boundaries, so the main capacity is adaptive boundary placement, not smoothing.
- **T016-B:** frozen T014 energy is a controlled development negative for nine-hard cross-boundary selection (**0/5**). Spatial MSE `0.03785726 = 1.08029×` Region2 and `1.16257×` the nine-hard oracle; selector/oracle disagreement 72.5%, mean Spearman `0.2398`. The nine-hard oracle is only `1.00172×` the full T016-A oracle, so candidate capacity is sufficient; ranking is the bottleneck.
- **T016-C:** two deterministic image-grouped five-fold OOF **pointwise value** probes are controlled development negatives (**0/5 each**). `probe28` spatial MSE `0.03580669 = 1.02178×` Region2 and `1.09959×` hard oracle. Adding only boundary coordinates gives `probe30 = 0.03545103 = 1.01163×` Region2 and `1.08867×` oracle. `probe30` improves pooled MSE only 0.99% over `probe28`; offset nearly reaches the 5% target (`0.95132×` Region2) but left/right (`1.02438×`) and quadrants (`1.07720×`) remain unsafe. Coordinates alone therefore do not restore safe rankability under absolute-value Huber training.

## Interpretation of the strongest evidence

T014 supports the narrow causal claim that **derivative supervision matters more than scalar value fit for gradient-based TTT**. The supported method is still Sobolev objective plus projected action geometry, not unconstrained learned energy.

T015–T016 isolate the remaining spatial problem. There is real capacity in moving the hard Region2 boundary, especially under boundary-offset degradation, but neither the frozen T014 energy nor a small absolute-value OOF ranker can safely exploit it. T016-C does **not** establish universal feature insufficiency: its heads regress absolute `log(MSE)` across images even though selection only requires within-episode candidate ordering. Moderate OOF rank correlation and the near-threshold offset result make a direct rank-aligned supervision test the minimal next diagnostic before any feature/model expansion.

The best fresh-validated spatial basis remains canonical Region2. T016-A/B/C are development diagnostics only and do not replace T014.

## Non-negotiable design principles

- Test-time adaptation/selection must never consume test labels, clean targets, degradation masks/gain maps, condition IDs, annotations, image IDs as shortcuts, source-only Jacobians/reference gradients, or evaluation metrics.
- Source/development clean references may be used only in explicitly declared training/diagnostic stages; outputs must be frozen before held-out reference evaluation.
- Label-free outputs/trajectories/decisions must be finalized and persisted before clean-reference metrics/oracles are attached.
- Fresh/development IDs become permanently unavailable for corrective tuning after inspection.
- A selector claim must be compared against a reference-only oracle; weak oracle headroom does not justify learning the selector.
- Do not infer rankability from train loss or average correlation alone; deployment-facing selected MSE and safety clauses remain decisive.
- Fresh-run launchers must fail closed by binding declared source SHAs to actual runtime scientific files.
- Development-only oracle/OOF diagnostics guide research direction but are not deployable or fresh qualification.

## Milestones

- **M0 / T001:** completed — mechanism scaffold.
- **M1 / T002:** completed — spatiality/capacity controls.
- **M2 / T003:** completed — simple-prior rescue negative.
- **M3 / T004–T008:** completed — nuisance readout qualified; original semantic TTT unqualified.
- **M4 / T009:** completed — objective/action geometry diagnosis.
- **M5 / T010:** completed — residual-target repair negative.
- **M6 / T011:** completed — projected Region2 strong but fresh qualification negative.
- **M7 / T012:** completed — learned stopping negative; oracle-limited.
- **M8 / T013:** completed — scalar learned inner objective negative.
- **M9 / T014:** **COMPLETED — SOBOLEV INNER OBJECTIVE PASSED 8/8 DEVELOPMENT AND 12/12 FRESH QUALIFICATION.**
- **M10 / T015:** **COMPLETED — CROSS-BASIS ROUTING NEGATIVE (4/10); ORACLE HEADROOM 1.94%.**
- **M11 / T016-A:** **COMPLETED — SHIFTABLE-BOUNDARY CAPACITY POSITIVE; 7.24% BELOW REGION2 ORACLE MSE.**
- **M11 / T016-B:** **COMPLETED — FROZEN ENERGY BOUNDARY SELECTOR NEGATIVE (0/5).**
- **M11 / T016-C:** **COMPLETED — 28-D AND 30-D POINTWISE OOF VALUE PROBES BOTH NEGATIVE (0/5).**
- **M11 / T016-D:** **ACTIVE — GROUPED OOF PAIRWISE RANK-SUPERVISION DIAGNOSTIC.**

## Current open task

`T016-D — grouped OOF pairwise boundary-ranking probe` in `coordination/CHATGPT_TO_CODEX.md`.

T016-D is CPU-only, uses the same already-inspected 40 development IDs and fixed folds/candidates, changes only the training objective from absolute value regression to within-episode pairwise logistic ranking, and compares the same 28-D and 30-D feature sets. No new data, renderer, CLIP, TTT, larger model, confidence gate, fresh run, detector, meta-learning, prompt retraining, or ViT3 work is authorized in this cycle.
