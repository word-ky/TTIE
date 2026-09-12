# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Processing for Task-Oriented Vision**

## Core scientific question

Can a vision system adapt a compact spatial correction field per test image, without test labels or clean targets, so that spatially heterogeneous nuisance degradation is corrected before a frozen downstream task model processes the image?

## Current hypothesis

A useful spatial TTT system requires four aligned pieces: (i) a content-safe nuisance signal, (ii) a label-free inner objective whose gradient field produces restoration-useful states, (iii) safe projected action/stopping geometry, and (iv) a spatial representation/decision rule that chooses geometry safely per image. T014 establishes the first three for canonical hard Region2. T015 rules out routing among global/bilinear2/Region2 because that three-output set has only 1.94% oracle headroom. T016-A establishes real development-only capacity in moving the hard Region2 boundary with fixed actions. T016-B/C/D show that frozen-energy ranking, pointwise value supervision, and forced pairwise rank30 selection are unsafe. T016-E then shows that a simple canonical confidence fallback can numerically pass all five development clauses, but the prescribed reuse of old OOF scores creates an inherited outer-fold dependency. The immediate question is therefore whether the confidence mechanism survives **proper fully nested image-level cross-fitting**.

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
- **T016-A:** fixed-action renderer-transfer is a positive development-only capacity diagnostic. Twenty-seven predeclared shiftable/soft renderers reuse the same selected four Region2 EV/gamma corners. Per-image reference oracle spatial MSE `0.03250770 = 0.92764×` Region2. `113/120` oracle selections use hard (`tau=0`) boundaries, so the main capacity is adaptive boundary placement, not smoothing.
- **T016-B:** frozen T014 energy is a controlled development negative for nine-hard cross-boundary selection (**0/5**). Spatial MSE `0.03785726 = 1.08029×` Region2 and `1.16257×` the nine-hard oracle. The nine-hard oracle is only `1.00172×` the full T016-A oracle, so candidate capacity is sufficient; ranking is the bottleneck.
- **T016-C:** two image-grouped five-fold OOF pointwise-value probes are controlled development negatives (**0/5 each**). `probe28 = 1.02178×` Region2; adding explicit boundary coordinates gives `probe30 = 1.01163×`. Coordinates alone do not restore safe rankability under absolute-value regression.
- **T016-D:** pairwise OOF ranking probes are controlled development negatives (**1/5 each**). `rank30` improves to `0.03421816 = 0.97645×` Region2 and median pooled Spearman `0.75`, with strong offset gain (`0.88846×` Region2), but forced argmin harms left/right (`1.04687×`) and quadrants (`1.01626×`). Pairwise supervision exposes useful ranking signal while forced selection remains unsafe.
- **T016-E:** the prescribed confidence fallback on frozen T016-D OOF scores gives a literal **5/5 numerical pass**: spatial `0.96750×` Region2, hard-oracle ratio `1.04118×`, offset `0.90806×`, left/right `1.00988×`, quadrants `1.00000×`; 32/120 episodes adapt and 27/32 adapted episodes are beneficial. However this is **not leakage-free nested CV**: calibration-image OOF scores can come from rankers trained on the current outer-held-out IDs. The result therefore supports confidence/abstention as a promising mechanism only; it does not establish development-safe generalization, a deployable `t=0.75`, or fresh qualification.

## Interpretation of the strongest evidence

T014 supports the narrow causal claim that **derivative supervision matters more than scalar value fit for gradient-based TTT**. The supported method is still Sobolev objective plus projected action geometry, not unconstrained learned energy.

T015–T016 isolate the remaining spatial issue. Adaptive hard-boundary placement has real oracle headroom and nine hard candidates capture essentially all of it. Direct frozen-energy selection fails. Pairwise rank30 learns substantial ordering signal, especially with explicit boundary coordinates, but unconditional selection is unsafe. T016-E indicates that canonical abstention may be the missing decision rule, yet its 5/5 pass is methodologically non-decisive because the threshold calibration reuses OOF scores with inherited outer-held-out training dependence. The next required evidence is a fully nested ranker-plus-threshold audit, not a larger model or fresh run.

The best fresh-validated spatial basis remains canonical Region2. T016-A through T016-E are development diagnostics only and do not replace T014.

## Non-negotiable design principles

- Test-time adaptation/selection must never consume test labels, clean targets, degradation masks/gain maps, condition IDs, annotations, image IDs as shortcuts, source-only Jacobians/reference gradients, or evaluation metrics.
- Source/development clean references may be used only in explicitly declared training/calibration/diagnostic stages; a held-out image's decision path must exclude that image's reference.
- Label-free outputs/trajectories/decisions must be finalized and persisted before their held-out clean-reference metrics/oracles are attached.
- Fresh/development IDs become permanently unavailable for corrective fresh tuning after inspection.
- A selector claim must be compared against a reference-only oracle; weak oracle headroom does not justify learning the selector.
- Do not infer deployable rankability from training loss or correlation alone; selected MSE and safety clauses remain decisive.
- Cross-validation claims must be genuinely image-grouped and, when a second-stage calibrator is fit, properly nested so outer-held-out examples cannot influence first-stage models used for calibration.
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
- **M11 / T016-A:** **COMPLETED — SHIFTABLE-BOUNDARY CAPACITY POSITIVE.**
- **M11 / T016-B:** **COMPLETED — FROZEN ENERGY BOUNDARY SELECTOR NEGATIVE (0/5).**
- **M11 / T016-C:** **COMPLETED — POINTWISE OOF VALUE PROBES NEGATIVE (0/5).**
- **M11 / T016-D:** **COMPLETED — PAIRWISE OOF RANK PROBES NEGATIVE (1/5), WITH STRONGER RANK SIGNAL.**
- **M11 / T016-E:** **COMPLETED — LITERAL CONFIDENCE AUDIT 5/5, BUT NOT FULLY NESTED; SCIENTIFICALLY NON-DECISIVE.**
- **M11 / T016-F:** **ACTIVE — FULLY NESTED RANK30 + CONFIDENCE AUDIT.**

## Current open task

`T016-F — fully nested rank30 + confidence fallback audit` in `coordination/CHATGPT_TO_CODEX.md`.

T016-F trains exactly 25 small CPU rank30 heads under fixed outer/inner image folds, uses the unchanged pairwise recipe and unchanged T016-E confidence/grid, and tests whether the 5/5 result survives proper nesting. No new data, feature, renderer, CLIP, TTT, threshold definition, larger model, fresh split, detector, meta-learning, prompt retraining, or ViT3 work is authorized in this cycle.
