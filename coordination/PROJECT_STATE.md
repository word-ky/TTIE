# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Processing for Task-Oriented Vision**

## Core scientific question

Can a vision system adapt a compact spatial correction field per test image, without test labels or clean targets, so that spatially heterogeneous nuisance degradation is corrected before a frozen downstream task model processes the image?

## Current hypothesis

A useful spatial TTT system requires four aligned pieces: (i) a content-safe nuisance signal, (ii) a label-free inner objective whose derivative field produces restoration-useful states, (iii) safe projected action/stopping geometry, and (iv) a spatial representation whose geometry can adapt safely per image. T014 establishes the first three for canonical hard Region2. T015 rules out simple routing among global/bilinear2/Region2. T016-A establishes real development-only headroom in moving the Region2 boundary with fixed actions, while T016-B–F show that discrete scalar ranking plus abstention is not robust enough.

T017-A showed that a fixed reference-only `tau=0.05` local cross recovers substantial pooled and offset hard-boundary headroom but fails quadrants safety after the chosen boundary is hardened. T017-B now resolves that failure mechanism: **soft→hard transfer dominates**, accounting for 19/20 harmful hard moves overall and 10/11 harmful quadrant moves. Independent x/y choice is nearly soft-oracle on the frozen table, so non-separability is not the primary explanation. The immediate question is therefore whether keeping the chosen geometry soft (`tau=0.05`) removes the safety failure and whether any resulting gain comes from adaptive boundary movement rather than fixed smoothing alone.

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
- **T016-B:** frozen T014 energy is a controlled development negative for nine-hard cross-boundary selection (**0/5**). Spatial MSE `0.03785726 = 1.08029×` Region2 and `1.16257×` the nine-hard oracle. Candidate capacity is sufficient while ranking is the bottleneck.
- **T016-C:** image-grouped pointwise-value probes are controlled development negatives (**0/5**). Adding explicit `(bx,by)` coordinates is not sufficient under absolute-value regression.
- **T016-D:** pairwise OOF ranking improves signal but remains unsafe (**1/5**). `rank30` reaches `0.97645×` Region2 with strong offset gain, but forced selection harms left/right and quadrants.
- **T016-E:** a canonical confidence fallback gives a literal **5/5** on reused OOF scores, but inherited outer-fold dependence makes the result methodologically non-decisive.
- **T016-F:** proper fully nested `rank30 + single confidence fallback` is a controlled development negative (**4/5**). Spatial pooled MSE is `0.03396188 = 0.96913×` Region2 and `1.04294×` the nine-hard oracle; offset is `0.90521×`, quadrants `1.00000×`, but left/right is `1.01850×`, violating the predeclared `1.01` safety limit. The fixed scalar-head/single-confidence cycle is closed.
- **T017-A:** reference-only local geometry viability is a controlled development negative (**4/5**). The fixed `tau=0.05` local-cross rule selects an already-rendered hard boundary and reaches spatial MSE `0.03331791 = 0.95076×` Region2 and `1.02317×` the nine-hard oracle, capturing `69.58%` of pooled oracle headroom. Left/right improves to `0.98782×` and offset to `0.85782×`, but quadrants degrade to `1.03303×`, failing the `1.01` safety limit.
- **T017-B:** reference-only failure attribution is **soft→hard transfer-dominant**. Among 20 harmful T017-A hard moves, 19 (**95%**) improve the same selected candidate in the `tau=0.05` soft renderer before becoming harmful after hardening; quadrants are 10/11 (**90.91%**) transfer flips and left/right 9/9. Only one quadrant case is a genuine soft interaction failure. The independent soft choice lies in the exact nine-soft oracle tie set on `104/120` episodes; mean soft separability regret is `3.224e-05`. This rules against training a hard-deployment geometry objective directly from the current `tau=0.05` surrogate.

## Interpretation of the strongest evidence

T014 supports the narrow causal claim that **derivative supervision matters more than scalar value fit for gradient-based TTT**. The supported method is Sobolev objective plus projected action geometry, not unconstrained learned energy.

T015–T017 isolate the remaining spatial problem. Adaptive boundary placement has real oracle headroom, and nine hard candidates capture essentially all of it. Frozen-energy ranking, pointwise value fitting, pairwise scalar ranking, and a single confidence fallback do not provide robust development-safe hard-boundary selection. T017-A shows that a local reference neighborhood contains useful geometry signal, and T017-B shows that the dominant failure is not x/y separability but **renderer mismatch between the soft signal and hard deployment**.

The next justified question is therefore a matched-renderer capacity question: **does the same frozen local choice become family-safe when kept at `tau=0.05`, and does adaptive movement add value beyond a fixed `tau=0.05` renderer?** No derivative-supervised soft-geometry model should be trained until that distinction is resolved.

The best fresh-validated spatial basis remains canonical hard Region2. T016/T017 diagnostics are development-only and do not replace T014 until a later independently frozen fresh qualification exists.

## Non-negotiable design principles

- Test-time adaptation/selection must never consume test labels, clean targets, degradation masks/gain maps, condition IDs, annotations, image IDs as shortcuts, source-only reference gradients/Jacobians, or evaluation metrics.
- Source/development clean references may be used only in explicitly declared training/calibration/diagnostic stages; a held-out image's deployment decision path must exclude that image's reference.
- Label-free outputs/trajectories/decisions must be finalized and persisted before held-out clean-reference metrics/oracles are attached.
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
- **M12 / T017-A:** **COMPLETED — REFERENCE LOCAL-GEOMETRY VIABILITY NEGATIVE (4/5); QUADRANT SAFETY FAILS.**
- **M12 / T017-B:** **COMPLETED — SOFT→HARD TRANSFER-DOMINANT ATTRIBUTION.**
- **M12 / T017-C:** **ACTIVE — MATCHED-SOFT DEPLOYMENT VIABILITY / SMOOTHING-vs-ADAPTATION AUDIT.**

## Current open task

`T017-C — matched-soft deployment viability and smoothing-vs-adaptation attribution` in `coordination/CHATGPT_TO_CODEX.md`.

T017-C uses only already-merged T016-A/T017-A tables and frozen T017-A decisions. It performs no training, rendering, CLIP, TTT, new data or fresh evaluation. It evaluates the same frozen geometry choice at `tau=0.05`, compares it against canonical hard Region2, canonical fixed-soft rendering and the nine-soft oracle, and separates fixed-smoothing benefit from adaptive-boundary benefit. The task remains reference-only development diagnosis and cannot establish a deployable selector by itself.