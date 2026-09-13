# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Processing for Task-Oriented Vision**

## Core scientific question

Can a vision system adapt a compact spatial correction field per test image, without test labels or clean targets, so that spatially heterogeneous nuisance degradation is corrected before a frozen downstream task model processes the image?

## Current hypothesis

A useful spatial TTT system requires four aligned pieces: (i) a content-safe nuisance signal, (ii) a label-free inner objective whose derivative field produces restoration-useful states, (iii) safe projected action/stopping geometry, and (iv) spatial geometry that can adapt per image without reference information at test time.

T014 establishes the first three for canonical hard Region2 and remains the best fresh-qualified method. T016-A establishes substantial hard-boundary headroom. T016-B–F and T017 show that scalar ranking/value/confidence and soft-renderer routes are not robust enough. T018 isolates the geometry bottleneck: the hard local x/y target is nearly factorized and viable, the frozen 28-D representation contains direction information, and a direct direction classifier succeeds under grouped OOF development. However, the frozen T018-D selector fails one-shot fresh qualification at 4/5 because three fresh quadrants make unnecessary x-boundary moves.

T019-A shows that a fixed, family-agnostic **1% per-axis utility deadband** is a viable development reference target: it preserves almost all hard-boundary headroom while eliminating harmful target moves. T019-B then holds representation, folds, architecture and training recipe fixed and changes only those labels. The grouped-OOF selector passes **7/7**, reduces harmful development episodes from `11` to `5`, preserves zero quadrant harm, and makes substantially fewer moves. This is controlled evidence that the remaining geometry issue is **movement-necessity semantics / utility-aware supervision**, not missing representation capacity.

The immediate active task is T019-C: freeze exactly one all-development x head and one y head using the accepted T019-A labels and unchanged T019-B recipe. This is an engineering barrier only; no new fresh cohort or qualification is authorized in the same cycle.

## Best fresh-validated method

The best deployable result remains **T014 Sobolev Region2 TTT**:

1. frozen T006 CLIP exposure readout and T007 clean-abstention gate;
2. frozen T014 28-D Sobolev restoration energy;
3. canonical hard Region2 projected EV+gamma state from identity;
4. exactly 40 label-free projected updates when active;
5. minimum predicted-energy checkpoint, earliest tie;
6. all label-free trajectories/decisions persisted before clean-reference evaluation.

T016–T019 are geometry diagnostics/candidate-method development and do not replace T014. T018-E failed fresh family safety, and T019-B is development-only, so no geometry-adaptive selector is yet accepted as the deployable replacement.

## Established controlled findings

- **T001–T003:** spatial ISP adaptation can outperform global correction on favorable heterogeneous shifts, but capacity alone and simple absolute-statistic priors are insufficient.
- **T004–T007:** zero-shot exposure signals are content-confounded; a source-trained frozen-CLIP exposure readout plus clean-abstention calibration yields a usable nuisance signal.
- **T008–T011:** semantic EV+gamma TTT has useful directions but suffers clean drift, over-correction, stopping and spatial-coupling issues; projected Region2 becomes strong but does not freshly qualify.
- **T012:** learned checkpoint selection cannot rescue the T011 trajectory; even the reference-only checkpoint oracle lacks required headroom.
- **T013:** scalar source-trained restoration energy is a controlled negative; fitting restoration value does not sufficiently constrain the test-time derivative field.
- **T014:** source-supervised Sobolev restoration energy is the first fully qualified learned-inner-objective result. Stage A passes **8/8** development clauses and frozen Stage B passes **12/12** fresh clauses on 40 unseen images. Fresh heterogeneous MSE is `0.03385803`; unseen-calibration gradient alignment is `73/74` positive with median cosine `0.93606`.
- **T015:** frozen routing among global/bilinear2/Region2 is a controlled fresh negative (**4/10**). Oracle improvement over Region2 is only **1.94%**.
- **T016-A:** shiftable hard-boundary capacity is positive: 27-candidate oracle `0.92764×` Region2; `113/120` oracle choices use `tau=0`.
- **T016-B:** frozen T014 energy as global nine-hard selector is negative (**0/5**), `1.08029×` Region2.
- **T016-C:** grouped pointwise 28-D/30-D value probes are negative (**0/5**).
- **T016-D:** pairwise OOF ranking improves signal but remains unsafe (**1/5**); `rank30` reaches `0.97645×` Region2 but harms families.
- **T016-E:** confidence fallback is literal **5/5** on reused OOF scores but is not fully nested and is non-decisive.
- **T016-F:** properly nested `rank30 + confidence` is negative (**4/5**); left/right safety fails at `1.01850×` Region2. Scalar-head/confidence cycle closed.
- **T017-A:** soft-cross→hard local geometry is negative (**4/5**); pooled strong, quadrants unsafe.
- **T017-B:** failure attribution is **soft→hard transfer-dominant**: `19/20` harmful hard moves are transfer flips.
- **T017-C:** matched-soft deployment is negative (**2/5**); `tau=0.05` renderer itself is poor relative to hard Region2. Soft-renderer route closed.
- **T018-A:** **hard local-direction reference target positive (5/5)**. Pooled `H1/H0 = 0.930464`, `H1/H* = 1.001327`; offset `0.850731`; left/right `0.963289`; quadrants `0.999946`. It recovers **98.26%** of pooled nine-hard oracle headroom with **69 beneficial / 51 equal / 0 harmful** moves.
- **T018-B:** **frozen-T014 hard-cross local direction negative (0/5)**. Pooled selected/H0 `1.086198`, selected/H* `1.168922`; offset `0.964061`; left/right `1.075253`; quadrants `1.258630`. Outcomes **24 beneficial / 38 equal / 58 harmful**.
- **T018-C:** **grouped-OOF direct hard-axis direction probe positive (5/5), development only**. Using only `concat(f0, f- - f0, f+ - f0)` from the frozen 28-D representation, pooled selected/H0 is `0.946340`, selected/H* `1.018412`, offset/H0 `0.874913`, left/right/H0 `0.983709`, quadrants/H0 `1.000000`. Outcomes are **57 beneficial / 52 equal / 11 harmful**; all 40 development quadrants remain canonical center.
- **T018-D:** completed engineering freeze of exactly one all-development x head and one y head using the unchanged T018-C recipe. This adds no performance evidence.
- **T018-E:** **one-shot fresh qualification negative (4/5)**. Pooled `H1/H0 = 0.932290`, pooled `H1/H* = 1.019900`, offset `0.846630`, left/right `0.964971`, but quadrants `1.011785 > 1.01`. Outcomes are **61 beneficial / 46 equal / 13 harmful**; quadrants are `0 / 37 / 3`, with all three harmful cases unnecessary x moves from a canonical nine-hard oracle. The cohort is burned for future qualification and may not be used for corrective tuning.
- **T019-A:** **fixed 1% utility-deadband reference target positive, development only**. Pooled `H_delta/H0 = 0.930928`, `H_delta/H* = 1.001826`; offset `0.850867`; left/right `0.964534`; quadrants `1.000000`. Outcomes are **56 beneficial / 64 equal / 0 harmful**. The deadband suppresses **26** original non-center axis labels and retains **97.6018%** of pooled nine-hard oracle headroom. All 40 development quadrants stay canonical center. No threshold search and no T018-E fresh references are used.
- **T019-B:** **grouped-OOF 1% deadband-direction selector positive (7/7), development only**. With the exact T018-C features/folds/model/optimizer/seed/epochs and only the labels changed, pooled selected/H0 is `0.949628`, selected/H* `1.021951`, offset/H0 `0.884558`, left/right/H0 `0.982292`, quadrants/H0 `1.000000`. Outcomes are **46 beneficial / 69 equal / 5 harmful**, versus T018-C's `57 / 52 / 11`; moving episodes fall `68→51`, both-axis moves `27→13`, and quadrants remain `0 / 40 / 0`. The small pooled/offset regression versus T018-C is the cost of the safer movement policy, while all seven predeclared clauses pass. PR #33 is merged as `e04a96da31a1a2f359e45ba5f251e04989ab895d`.

## Interpretation of strongest evidence

T014 supports the narrow causal claim that **derivative supervision matters more than scalar value fit for gradient-based TTT** and remains the deployed baseline.

The geometry evidence now separates four issues. First, hard boundary placement has real headroom and the local x/y target is nearly factorized. Second, representation-level direction information exists, because T018-C succeeds without changing the frozen candidate features. Third, exact-direction supervision is not sufficiently conservative under fresh transfer: T018-E fails only because a few near-zero-headroom quadrants move unnecessarily. Fourth, T019-A/B show that utility-aware supervision is a credible remedy: a 1% deadband removes low-value move labels, and under a label-only OOF intervention the learned selector materially reduces harmful and unnecessary moves without losing the established development gates.

T019-B is still development evidence. It does **not** erase the T018-E fresh negative or justify reuse of that cohort. The next defensible sequence is: freeze one literal all-development T019 selector without further tuning, then qualify that frozen artifact once on a different unseen cohort with strict pre-reference hashing/provenance.

## T018-E provenance note

A later review found that the original T018-E frozen config did not contemporaneously hash-bind `prepared.json`. PR #31 repairs this for future runs and is merged as `417141bcd697eb088e35e91686e338cb36885eef`. A retrospective audit confirms all 120 saved row assignments and input bindings are consistent with the frozen manifest/decisions, but this cannot retroactively prove the complete pre-inference mapping chain. Therefore preserve the reported T018-E 4/5 negative and raw artifacts, but do not claim that the original verifier established full contemporaneous mapping-chain integrity.

## Non-negotiable design principles

- Test-time adaptation/selection must never consume test labels, clean targets, degradation masks/gain maps, condition IDs, annotations, image IDs as semantic shortcuts, source-only reference gradients/Jacobians, or evaluation metrics.
- Source/development clean references may be used only in explicitly declared training/calibration/diagnostic stages; a held-out/fresh image's deployment decision path must exclude that image's reference.
- Label-free outputs/trajectories/decisions must be finalized and persisted before clean-reference metrics/oracles/family labels are attached.
- Fresh/development IDs become permanently unavailable for corrective fresh tuning after inspection.
- A selector claim must be compared against a reference-only oracle; weak oracle headroom does not justify learning a selector.
- Do not infer deployable rankability from training loss/correlation alone; selected MSE and family safety remain decisive.
- Cross-validation claims must be image-grouped and properly nested when second-stage calibration is fit.
- Fresh-run launchers must fail closed by binding declared source SHAs and all row-mapping/preparation artifacts to runtime scientific files.
- Development-only oracle/OOF/reference diagnostics guide research direction but are not fresh qualification.
- A failed fresh cohort may motivate a new development-only hypothesis, but its references/logits/outcomes must not be used to tune that new method; any revised method requires a different future unseen cohort.

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
- **M11 / T016-A–F:** completed — hard-boundary headroom established; scalar/ranking/confidence selectors closed.
- **M12 / T017-A–C:** completed — soft diagnostic signal identified; soft-transfer/deployment route closed.
- **M13 / T018-A:** **COMPLETED — HARD LOCAL-DIRECTION REFERENCE TARGET POSITIVE (5/5).**
- **M13 / T018-B:** **COMPLETED — FROZEN-T014 HARD-CROSS LOCAL-DIRECTION NEGATIVE (0/5).**
- **M13 / T018-C:** **COMPLETED — GROUPED-OOF DIRECT HARD-AXIS DIRECTION POSITIVE (5/5), DEVELOPMENT ONLY.**
- **M13 / T018-D:** **COMPLETED — FINAL ALL-DEVELOPMENT DIRECT-DIRECTION SELECTOR FROZEN.**
- **M13 / T018-E:** **COMPLETED — ONE-SHOT FRESH QUALIFICATION NEGATIVE (4/5), QUADRANTS SAFETY FAILS AT 1.011785×.**
- **M14 / T019-A:** **COMPLETED — FIXED 1% UTILITY-DEADBAND TARGET POSITIVE, 5/5 + ZERO HARMFUL, DEVELOPMENT ONLY.**
- **M14 / T019-B:** **COMPLETED — GROUPED-OOF 1% DEADBAND-DIRECTION POSITIVE, 7/7, DEVELOPMENT ONLY.**
- **M14 / T019-C:** **ACTIVE — FINAL ALL-DEVELOPMENT 1% DEADBAND-DIRECTION SELECTOR FREEZE.**

## Current open task

`T019-C — freeze the final all-development 1% deadband direction selector` in `coordination/CHATGPT_TO_CODEX.md`.

T019-C is an engineering freeze only. It must train exactly one x head and one y head from the fixed 120 development features and accepted T019-A labels using the unchanged T019-B recipe, package a reference-free inference interface, and independently replay the frozen heads without opening target/reference artifacts. It must not launch a new cohort, fresh qualification, or any additional development tuning in the same cycle.