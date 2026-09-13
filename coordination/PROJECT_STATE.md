# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Processing for Task-Oriented Vision**

## Core scientific question

Can a vision system adapt a compact spatial correction field per test image, without test labels or clean targets, so that spatially heterogeneous nuisance degradation is corrected before a frozen downstream task model processes the image?

## Current hypothesis

A useful spatial TTT system requires four aligned pieces: (i) a content-safe nuisance signal, (ii) a label-free inner objective whose derivative field produces restoration-useful states, (iii) safe projected action/stopping geometry, and (iv) spatial geometry that can adapt per image without reference information at test time.

T014 establishes the first three for canonical hard Region2. T015 shows that simple routing among global/bilinear2/Region2 has too little oracle headroom. T016-A establishes substantial development-only headroom in moving the hard Region2 boundary with fixed actions; T016-B–F show that global scalar ranking/value/ranking-plus-confidence over nine boundaries is not robust enough. T017 shows that a `tau=0.05` soft neighborhood carries useful diagnostic signal but is neither a safe hard-transfer surrogate nor a viable deployed renderer.

**T018-A now changes the geometry conclusion materially:** on the hard renderer itself, a center-local factorized reference target is viable. Independent x/y choices from the five hard-cross reference MSE values pass all 5/5 family/pooled clauses, recover 98.26% of pooled nine-hard oracle headroom, and produce zero harmful moves. Therefore the remaining bottleneck is no longer hard-target capacity or x/y factorization; it is whether a label-free signal can recover that local hard direction.

The immediate active question is T018-B: before training any new geometry predictor, test whether the already-frozen T014 Sobolev energy has usable **local hard-direction** information when applied factorwise to the same hard cross. This is distinct from T016-B's failed global nine-way argmin.

## Best fresh-validated method

The best deployable result remains **T014 Sobolev Region2 TTT**:

1. frozen T006 CLIP exposure readout and T007 clean-abstention gate;
2. frozen T014 28-D Sobolev restoration energy;
3. canonical hard Region2 projected EV+gamma state from identity;
4. exactly 40 label-free projected updates when active;
5. minimum predicted-energy checkpoint, earliest tie;
6. all label-free trajectories/decisions persisted before clean-reference evaluation.

T016–T018 are development diagnostics and do not replace T014 until an independently frozen fresh qualification succeeds.

## Established controlled findings

- **T001–T003:** spatial ISP adaptation can outperform global correction on favorable heterogeneous shifts, but capacity alone and simple absolute-statistic priors are insufficient.
- **T004–T007:** zero-shot exposure signals are content-confounded; a source-trained frozen-CLIP exposure readout plus clean-abstention calibration yields a usable nuisance signal.
- **T008–T011:** semantic EV+gamma TTT has useful directions but suffers clean drift, over-correction, stopping and spatial-coupling issues; projected Region2 becomes strong but does not freshly qualify.
- **T012:** learned checkpoint selection cannot rescue the T011 trajectory; even the reference-only checkpoint oracle lacks required headroom.
- **T013:** scalar source-trained restoration energy is a controlled negative; fitting restoration value does not sufficiently constrain the test-time derivative field.
- **T014:** source-supervised Sobolev restoration energy is the first fully qualified learned-inner-objective result. Stage A passes **8/8** development clauses and frozen Stage B passes **12/12** fresh clauses on 40 unseen images. Fresh heterogeneous MSE is `0.03385803`; unseen-calibration gradient alignment is `73/74` positive with median cosine `0.93606`.
- **T015:** frozen routing among global/bilinear2/Region2 is a controlled fresh negative (**4/10**). The oracle among those three outputs improves only **1.94%** over Region2, so a smarter selector over that fixed trio is not justified.
- **T016-A:** shiftable hard-boundary capacity is a positive development diagnostic. The 27-candidate oracle reaches `0.03250770 = 0.92764×` Region2; `113/120` oracle selections use `tau=0`, so adaptive boundary placement—not smoothing—is the main capacity.
- **T016-B:** frozen T014 energy as a global nine-hard selector is negative (**0/5**): `1.08029×` Region2 and `1.16257×` the nine-hard oracle.
- **T016-C:** grouped pointwise 28-D/30-D value probes are negative (**0/5**).
- **T016-D:** pairwise OOF ranking improves signal but remains unsafe (**1/5**); `rank30` reaches `0.97645×` Region2 but harms left/right/quadrants.
- **T016-E:** canonical confidence fallback gives a literal **5/5** on reused OOF scores, but inherited outer-fold dependence makes it non-decisive.
- **T016-F:** proper fully nested `rank30 + confidence` is negative (**4/5**): pooled `0.96913×` Region2, but left/right `1.01850×` violates safety. The fixed scalar-head/single-confidence cycle is closed.
- **T017-A:** soft-cross→hard local geometry is negative (**4/5**): pooled `0.95076×` Region2 and near the hard oracle, but quadrants `1.03303×` fail safety.
- **T017-B:** failure attribution is **soft→hard transfer-dominant**: 19/20 harmful hard moves are transfer flips; x/y interaction is not the main issue.
- **T017-C:** matched-soft deployment is negative (**2/5**): adaptive soft choices are near the soft oracle, but `tau=0.05` itself is poor relative to hard Region2, especially quadrants. The matched-soft route is closed.
- **T018-A:** **hard local-direction reference target is positive (5/5)**. Pooled `H1/H0 = 0.930464`, `H1/H* = 1.001327`; offset `0.850731`; left/right `0.963289`; quadrants `0.999946`. It recovers **98.26%** of pooled nine-hard oracle headroom. Movement counts are 51 no-move / 6 x-only / 33 y-only / 30 both; outcomes are **69 beneficial / 51 equal / 0 harmful**; `115/120` choices lie in the exact oracle tie set. No pure x/y interaction failures are observed. This is reference-only target viability, not a deployable selector.

## Interpretation of strongest evidence

T014 supports the narrow causal claim that **derivative supervision matters more than scalar value fit for gradient-based TTT**. The supported deployed method remains Sobolev objective plus projected canonical Region2 geometry.

T015–T018 isolate the remaining spatial-geometry problem. Hard boundary placement has real oracle headroom. Global boundary ranking is difficult, but T018-A shows that the useful target is much simpler than nine-way global ranking: around canonical Region2, the hard landscape factorizes almost perfectly into two local axis decisions. This suggests the scientifically relevant object may be a **local geometry direction** rather than a globally calibrated boundary value.

The next justified test is therefore conservative: reuse the already-frozen T014 label-free energy and ask whether its local hard-cross direction matches enough of T018-A to pass the same deployment-facing family gates. A positive T018-B would show that T016-B's failure was partly a decision-rule mismatch (global argmin versus local direction). A negative T018-B would show that target simplicity alone is insufficient and would justify, only in a later cycle, considering a dedicated source-supervised geometry-direction learner.

## Non-negotiable design principles

- Test-time adaptation/selection must never consume test labels, clean targets, degradation masks/gain maps, condition IDs, annotations, image IDs as semantic shortcuts, source-only reference gradients/Jacobians, or evaluation metrics.
- Source/development clean references may be used only in explicitly declared training/calibration/diagnostic stages; a held-out image's deployment decision path must exclude that image's reference.
- Label-free outputs/trajectories/decisions must be finalized and persisted before clean-reference metrics/oracles/family labels are attached.
- Fresh/development IDs become permanently unavailable for corrective fresh tuning after inspection.
- A selector claim must be compared against a reference-only oracle; weak oracle headroom does not justify learning a selector.
- Do not infer deployable rankability from training loss/correlation alone; selected MSE and family safety remain decisive.
- Cross-validation claims must be image-grouped and properly nested when second-stage calibration is fit.
- Fresh-run launchers must fail closed by binding declared source SHAs to runtime scientific files.
- Development-only oracle/OOF/reference diagnostics guide research direction but are not fresh qualification.

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
- **M11 / T016-B:** **COMPLETED — FROZEN ENERGY GLOBAL BOUNDARY SELECTOR NEGATIVE (0/5).**
- **M11 / T016-C:** **COMPLETED — POINTWISE OOF VALUE PROBES NEGATIVE (0/5).**
- **M11 / T016-D:** **COMPLETED — PAIRWISE OOF RANK PROBES NEGATIVE (1/5).**
- **M11 / T016-E:** **COMPLETED — LITERAL CONFIDENCE 5/5, NOT FULLY NESTED; NON-DECISIVE.**
- **M11 / T016-F:** **COMPLETED — FULLY NESTED CONFIDENCE NEGATIVE (4/5); LEFT/RIGHT SAFETY FAILS.**
- **M12 / T017-A:** **COMPLETED — SOFT-CROSS→HARD LOCAL VIABILITY NEGATIVE (4/5).**
- **M12 / T017-B:** **COMPLETED — SOFT→HARD TRANSFER-DOMINANT ATTRIBUTION.**
- **M12 / T017-C:** **COMPLETED — MATCHED-SOFT VIABILITY NEGATIVE (2/5); SOFT RENDERER FAMILY CLOSED.**
- **M13 / T018-A:** **COMPLETED — HARD LOCAL-DIRECTION REFERENCE TARGET POSITIVE (5/5).**
- **M13 / T018-B:** **ACTIVE — FROZEN-T014 HARD-CROSS LOCAL-DIRECTION AUDIT.**

## Current open task

`T018-B — frozen-T014 hard-cross local-direction audit` in `coordination/CHATGPT_TO_CODEX.md`.

T018-B must make all 120 boundary decisions using only the already-saved frozen T014 hard-cross energy scores from T016-B, with literal center→lower→upper axis tie handling. Decisions are frozen before any reference MSE, family label, image ID, T018-A target, or oracle information is attached. No training, rerendering, model rerun, threshold tuning, new data, or GPU work is authorized.