# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Processing for Task-Oriented Vision**

## Core scientific question

Can a vision system adapt a compact spatial correction field per test image, without test labels or clean targets, so that spatially heterogeneous nuisance degradation is corrected before a frozen downstream task model processes the image?

## Current hypothesis

A useful spatial TTT system requires four aligned pieces: (i) a content-safe nuisance signal, (ii) a label-free inner objective whose gradient field produces restoration-useful states, (iii) safe projected action/stopping geometry, and (iv) a spatial representation whose geometry can adapt safely per image. T014 establishes the first three for canonical hard Region2. T015 rules out simple routing among global/bilinear2/Region2. T016-A establishes real development-only headroom in moving the Region2 boundary with fixed actions, while T016-B–F show that treating boundary geometry as a discrete scalar-ranking plus abstention problem is not robust enough. The immediate question is now whether that boundary headroom is **locally accessible through a smooth geometry landscape**, which would justify extending the T014 derivative-supervision idea to spatial geometry itself.

## Best fresh-validated method

The best deployable result remains **T014 Sobolev Region2 TTT**:

1. frozen T006 CLIP exposure readout and T007 clean-abstention gate;
2. frozen T014 28-D Sobolev restoration energy;
3. canonical hard Region2 projected EV+gamma state from identity;
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
- **T016-A:** fixed-action renderer transfer is a positive development-only capacity diagnostic. Twenty-seven predeclared shiftable/soft renderers reuse the same selected four Region2 EV/gamma corners. Per-image reference oracle spatial MSE `0.03250770 = 0.92764×` Region2. `113/120` oracle selections use hard (`tau=0`) boundaries, so the main capacity is adaptive boundary placement, not smoothing.
- **T016-B:** frozen T014 energy is a controlled development negative for nine-hard cross-boundary selection (**0/5**). Spatial MSE `0.03785726 = 1.08029×` Region2 and `1.16257×` the nine-hard oracle. The nine-hard oracle captures essentially all T016-A headroom, so candidate capacity is sufficient while ranking is the bottleneck.
- **T016-C:** image-grouped pointwise-value probes are controlled development negatives (**0/5**). Adding explicit `(bx,by)` coordinates is not sufficient under absolute-value regression.
- **T016-D:** pairwise OOF ranking improves signal but remains unsafe (**1/5**). `rank30` reaches `0.97645×` Region2 with strong offset gain, but forced selection harms left/right and quadrants.
- **T016-E:** a canonical confidence fallback gives a literal **5/5** on reused OOF scores, but the score reuse creates inherited outer-fold dependence, so this result is methodologically non-decisive.
- **T016-F:** proper fully nested `rank30 + single confidence fallback` is a controlled development negative (**4/5**). Spatial pooled MSE is `0.03396188 = 0.96913×` Region2 and `1.04294×` the nine-hard oracle; offset is `0.90521×` Region2 and quadrants `1.00000×`, but left/right is `1.01850×`, violating the predeclared `1.01` safety limit. All six harmful adaptive episodes are left/right. The fixed scalar-head/single-confidence cycle is closed without retuning.

## Interpretation of the strongest evidence

T014 supports the narrow causal claim that **derivative supervision matters more than scalar value fit for gradient-based TTT**. The supported method is Sobolev objective plus projected action geometry, not unconstrained learned energy.

T015–T016 isolate the remaining spatial problem. Adaptive boundary placement has real oracle headroom, and nine hard candidates capture essentially all of it. Frozen-energy ranking, pointwise value fitting, pairwise scalar ranking, and a single confidence fallback do not provide robust development-safe boundary selection. The next justified question is not another selector tweak; it is whether the reference boundary landscape has enough local smooth structure that geometry could itself become a fast variable trained with derivative supervision.

The best fresh-validated spatial basis remains canonical Region2. T016/T017 diagnostics are development-only and do not replace T014 until a later independently frozen fresh qualification exists.

## Non-negotiable design principles

- Test-time adaptation/selection must never consume test labels, clean targets, degradation masks/gain maps, condition IDs, annotations, image IDs as shortcuts, source-only Jacobians/reference gradients, or evaluation metrics.
- Source/development clean references may be used only in explicitly declared training/calibration/diagnostic stages; a held-out image's deployment decision path must exclude that image's reference.
- Label-free outputs/trajectories/decisions must be finalized and persisted before their held-out clean-reference metrics/oracles are attached.
- Fresh/development IDs become permanently unavailable for corrective fresh tuning after inspection.
- A selector claim must be compared against a reference-only oracle; weak oracle headroom does not justify learning the selector.
- Do not infer deployable rankability from training loss/correlation alone; selected MSE and family safety remain decisive.
- Cross-validation claims must be image-grouped and properly nested when second-stage calibration is fit.
- Fresh-run launchers must fail closed by binding declared source SHAs to runtime scientific files.
- Development-only oracle/OOF/reference-gradient diagnostics guide research direction but are not deployable or fresh qualification.

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
- **M11 / T016-A:** **COMPLETED — SHIFTABLE-BOUNDARY CAPACITY POSITIVE.**
- **M11 / T016-B:** **COMPLETED — FROZEN ENERGY BOUNDARY SELECTOR NEGATIVE (0/5).**
- **M11 / T016-C:** **COMPLETED — POINTWISE OOF VALUE PROBES NEGATIVE (0/5).**
- **M11 / T016-D:** **COMPLETED — PAIRWISE OOF RANK PROBES NEGATIVE (1/5).**
- **M11 / T016-E:** **COMPLETED — LITERAL CONFIDENCE 5/5, NOT FULLY NESTED; NON-DECISIVE.**
- **M11 / T016-F:** **COMPLETED — FULLY NESTED CONFIDENCE NEGATIVE (4/5); LEFT/RIGHT SAFETY FAILS.**
- **M12 / T017-A:** **ACTIVE — REFERENCE-ONLY LOCAL GEOMETRY-LANDSCAPE VIABILITY AUDIT.**

## Current open task

`T017-A — reference-only local geometry-landscape viability audit` in `coordination/CHATGPT_TO_CODEX.md`.

T017-A uses only the accepted T016-A 120 × 27 precomputed candidate-MSE table. It performs no model training, rendering, CLIP, TTT, new data or fresh evaluation. Its sole purpose is to test whether a fixed `tau=0.05` local cross around canonical contains enough reference-direction information to recover the nine-hard boundary headroom safely. A positive result would justify considering derivative-supervised fast geometry in a later cycle; a negative result would stop that inference without ruling out all richer spatial representations.
