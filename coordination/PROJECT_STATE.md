# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Processing for Task-Oriented Vision**

## Core scientific question

Can a vision system adapt a compact spatial correction field per test image, without test labels or clean targets, so that spatially heterogeneous nuisance degradation is corrected before a frozen downstream task model processes the image?

## Current hypothesis

A useful spatial TTT system requires four aligned pieces: (i) a content-safe nuisance signal, (ii) a label-free inner objective whose derivative field produces restoration-useful states, (iii) safe projected action/stopping geometry, and (iv) spatial geometry that can adapt per image without reference information at test time.

T014 establishes the first three for canonical hard Region2. T015 shows that simple routing among global/bilinear2/Region2 has too little oracle headroom. T016-A establishes substantial development-only headroom in moving the hard Region2 boundary, while T016-B–F show that global scalar ranking/value/ranking-plus-confidence over nine boundaries is not robust enough. T017 shows that a `tau=0.05` soft neighborhood carries diagnostic signal but is neither a safe hard-transfer surrogate nor a viable deployed renderer.

T018 isolates the geometry bottleneck. T018-A shows the **hard local x/y target is simple and viable**. T018-B shows the existing frozen T014 scalar Sobolev energy fails even under that correct local decision geometry. T018-C passes **5/5** under strict image-grouped OOF when the same frozen 28-D candidate representation is trained directly for the x/y three-way direction. T018-D freezes the literal all-development two-head selector. T018-E then gives the decisive fresh result: pooled, oracle-proximity, offset, and left/right clauses transfer, but **quadrants safety fails narrowly at `1.011785×` Region2 (4/5 overall)** because three fresh quadrant episodes make unnecessary x-boundary moves while canonical center is already the nine-hard oracle.

The supported interpretation is now more specific: **useful hard-geometry direction information survives in the representation, but a direct exact-direction classifier is not conservative enough about whether movement is materially useful on fresh near-zero-headroom cases.** T018-E is a one-shot fresh negative; its cohort is burned for future qualification and cannot be used for corrective tuning.

The immediate active task is T019-A: a development-only reference audit of a fixed, family-agnostic **1% utility deadband** on the hard local target. It asks whether requiring at least 1% per-axis reference improvement before assigning a non-center target preserves the established geometry headroom with zero harmful target moves. No new selector is trained in this cycle.

## Best fresh-validated method

The best deployable result remains **T014 Sobolev Region2 TTT**:

1. frozen T006 CLIP exposure readout and T007 clean-abstention gate;
2. frozen T014 28-D Sobolev restoration energy;
3. canonical hard Region2 projected EV+gamma state from identity;
4. exactly 40 label-free projected updates when active;
5. minimum predicted-energy checkpoint, earliest tie;
6. all label-free trajectories/decisions persisted before clean-reference evaluation.

T016–T019 are geometry diagnostics/candidate-method development and do not replace T014. T018-E explicitly fails the fresh family-safety qualification, so the frozen T018-D direct-direction selector is not accepted as the deployable replacement.

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
- **T018-A:** **hard local-direction reference target positive (5/5)**. Pooled `H1/H0 = 0.930464`, `H1/H* = 1.001327`; offset `0.850731`; left/right `0.963289`; quadrants `0.999946`. It recovers **98.26%** of pooled nine-hard oracle headroom with **69 beneficial / 51 equal / 0 harmful** moves; `115/120` choices lie in the exact oracle tie set.
- **T018-B:** **frozen-T014 hard-cross local direction negative (0/5)**. Pooled selected/H0 `1.086198`, selected/H* `1.168922`; offset `0.964061`; left/right `1.075253`; quadrants `1.258630`. Outcomes **24 beneficial / 38 equal / 58 harmful**; exact T018-A joint agreement `32/120`.
- **T018-C:** **grouped-OOF direct hard-axis direction probe positive (5/5), development only**. Using only `concat(f0, f- - f0, f+ - f0)` from the frozen 28-D representation, pooled selected/H0 is `0.946340`, selected/H* `1.018412`, offset/H0 `0.874913`, left/right/H0 `0.983709`, quadrants/H0 `1.000000`. Exact T018-A agreement is x `105/120`, y `98/120`, joint `87/120`; outcomes are **57 beneficial / 52 equal / 11 harmful**. All 40 development quadrants remain canonical center.
- **T018-D:** **completed engineering freeze** of exactly one all-development x head and one y head using the unchanged T018-C recipe. Reference-free inference accepts only the five required hard-cross 28-D feature vectors. This milestone adds no performance evidence.
- **T018-E:** **one-shot fresh qualification negative (4/5)** on a deterministic disjoint 40-image/120-episode cohort. Pooled `H1/H0 = 0.932290`, pooled `H1/H* = 1.019900`, offset `0.846630`, left/right `0.964971`, but quadrants `1.011785 > 1.01`. Outcomes are pooled **61 beneficial / 46 equal / 13 harmful**; quadrants are **0 / 37 / 3**, and all three harmful quadrant cases are unnecessary x moves from a canonical nine-hard oracle. The cohort and decisions were frozen before reference evaluation, so this is a valid fresh negative rather than a leakage artifact.

## Interpretation of strongest evidence

T014 supports the narrow causal claim that **derivative supervision matters more than scalar value fit for gradient-based TTT** and remains the deployed baseline.

T015–T018 isolate the spatial-geometry problem. Hard boundary placement has real oracle headroom. T018-A shows that around canonical Region2 the useful hard landscape almost factorizes into local x/y decisions. T018-B rejects reuse of the old scalar Sobolev energy for those directions. T018-C succeeds on grouped OOF without changing the representation, renderer, candidates, or feature computation—only the supervised task changes from scalar/ranking objectives to direct axis direction classification. This establishes representation-level direction sufficiency on development data.

T018-E adds an important counterweight: **direction sufficiency is not the same as safe movement transfer.** The frozen direct classifier retains strong pooled and offset gains on fresh images, but three unnecessary quadrant x moves are enough to violate the family-safety gate. Because fresh quadrants have essentially no useful movement headroom, the next method question is whether the target should encode a minimum utility margin before assigning a non-center direction. Any such design must be developed without tuning on T018-E references and then, if it survives development controls, be qualified on a different future unseen cohort.

## Non-negotiable design principles

- Test-time adaptation/selection must never consume test labels, clean targets, degradation masks/gain maps, condition IDs, annotations, image IDs as semantic shortcuts, source-only reference gradients/Jacobians, or evaluation metrics.
- Source/development clean references may be used only in explicitly declared training/calibration/diagnostic stages; a held-out/fresh image's deployment decision path must exclude that image's reference.
- Label-free outputs/trajectories/decisions must be finalized and persisted before clean-reference metrics/oracles/family labels are attached.
- Fresh/development IDs become permanently unavailable for corrective fresh tuning after inspection.
- A selector claim must be compared against a reference-only oracle; weak oracle headroom does not justify learning a selector.
- Do not infer deployable rankability from training loss/correlation alone; selected MSE and family safety remain decisive.
- Cross-validation claims must be image-grouped and properly nested when second-stage calibration is fit.
- Fresh-run launchers must fail closed by binding declared source SHAs to runtime scientific files.
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
- **M14 / T019-A:** **ACTIVE — FIXED 1% UTILITY-DEADBAND HARD-LOCAL TARGET VIABILITY AUDIT.**

## Current open task

`T019-A — fixed 1% utility-deadband hard-local target viability audit` in `coordination/CHATGPT_TO_CODEX.md`.

T019-A is development-only and reference-only. It must use the accepted T018-A/T016-A hard candidate table, a single predeclared `delta=0.01` per-axis improvement threshold, no threshold search, no T018-E fresh references for method design, no new model training, no fresh cohort, and no CLIP/TTT/rerender/GPU work. It must stop after deciding whether the conservative target itself passes the five established clauses with zero harmful development moves.