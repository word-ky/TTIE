# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Enhancement**

## Core scientific question

Can a vision system adapt a compact spatial image-processing state per test image, without test labels or clean targets, so that unknown and spatially heterogeneous degradation is corrected by a reference-free test-time optimization field?

## Current scientific state

**T014 Sobolev Region2 TTT remains the broad fresh-qualified Ours.** It combines the frozen T006/T007 nuisance readout + clean-abstention gate, the source-supervised T014 Sobolev restoration energy, canonical hard Region2 EV+gamma adaptation, 40 projected label-free updates when active, and minimum predicted-energy checkpoint selection. T014 passed 8/8 development and 12/12 fresh clauses; fresh heterogeneous MSE is `0.03385803`, and unseen-calibration gradient alignment is `73/74` positive with median cosine `0.93606`.

**T019 remains a heterogeneous-only fresh-qualified geometry extension, not the broad default.** T019-B changed only supervision from exact direction to a fixed 1% utility-deadband target while holding representation/folds/network/optimizer/seed/epochs fixed; it passed 7/7 grouped-OOF development clauses and reduced harmful episodes `11→5`. T019-C froze the final two-head selector. T019-D passed 5/5 on a new 40-image / 120-episode heterogeneous fresh cohort: pooled `H1/H0=0.952464`, oracle proximity `1.027465`, offset `0.874063`, left/right `1.004090`, quadrants `1.000000`, with `43 beneficial / 73 equal / 4 harmful` and zero quadrant movement.

**T020-A is a valid fresh non-spatial negative (3/4).** On a second new 40-image / 120-episode cohort, the frozen T019-C selector gives pooled `0.986616×`, clean `1.046125×` **fail**, homogeneous-dark `0.994103×`, homogeneous-bright `0.963036×`. Clean outcomes are `0 beneficial / 39 equal / 1 harmful`; that fresh cohort is burned for corrective tuning.

**T020-B is a development-only non-spatial target positive (5/5).** The unchanged fixed `delta=0.01` per-axis reference deadband target on accepted development images gives pooled `0.900891×`, clean `0.721586×`, dark `0.906699×`, bright `0.889559×`, with `61 beneficial / 59 equal / 0 harmful`. Thus the target principle itself is viable on these development cases.

**T020-C is a development-only non-spatial OOF negative (3/5).** It kept the frozen T019 28-D representation, historical five image-grouped folds, and exact T019-B `84→64→64→3` unweighted-CE learner fixed. Results are pooled `0.947906×` pass, clean `1.207265×` fail, dark `0.937079×` pass, bright `0.970188×` pass, and clean harmful count `1` fail. Pooled outcomes are `37 beneficial / 63 equal / 20 harmful`.

**T020-D is a development-only mechanistic diagnosis: the T020-C failure is direction-dominant under the predeclared oracle test.** Necessity-oracle / frozen predicted-sign passes `3/5`, while frozen-necessity / direction-oracle passes `5/5`. Among the 20 harmful T020-C episodes, `19` contain at least one wrong-direction axis. This identifies sign prediction as a first-order bottleneck but does not provide a deployable repair.

**T020-E is now a controlled development-only negative (3/5): simple binary sign factorization does not repair the non-spatial safety failure.** With every T020-C move/no-move decision frozen, exact cached features/folds/normalizers reused, and only fixed `84→64→64→2` lower-vs-upper heads learned from fold-training non-center targets, the result is pooled `0.952217×` pass, clean `1.207265×` fail, dark `0.940200×` pass, bright `0.977458×` pass, and clean harmful count `1` fail. Pooled harmful episodes remain `20→20`; wrong-direction moved axes improve only `31→29`; homogeneous-bright harmful episodes worsen `15→16`. Therefore the hypothesis that unsafe non-spatial behavior is mainly a consequence of using a single symmetric three-way CE head is rejected. PR #40 was accepted and merged as `e6874f7f8d0b05a507af0d12eecc1e08f39200ff`.

The universal adaptive-geometry repair branch is **paused**. T014 remains the broad default and T019 remains a heterogeneous-only extension. No confidence threshold, class weighting, larger MLP, extra seed, or combined-domain selector training is currently justified by the controlled evidence.

## Best current methods

### Broad fresh-qualified Ours

**T014 Sobolev Region2 TTT** — broad deployable image-enhancement method under the accepted protocol.

### Heterogeneous-only fresh-qualified extension

**T019 = T014 + frozen 1%-deadband hard-boundary selector** — fresh-qualified only on the prescribed heterogeneous spatial protocol. T020-A/T020-C/T020-E prevent promotion to a universal default.

## Strongest controlled findings

- T001–T003: spatial ISP capacity helps heterogeneous shifts; capacity/simple priors alone are insufficient.
- T004–T007: zero-shot exposure signals are content-confounded; frozen source-trained nuisance readout + clean abstention is usable.
- T008–T012: semantic TTT has useful directions but drift/stopping/coupling problems; learned checkpointing cannot rescue oracle-limited trajectories.
- T013: scalar restoration-value fitting does not sufficiently constrain the TTT derivative field.
- **T014:** Sobolev derivative supervision is the first fully qualified learned inner objective; the learned optimization field matters more than scalar value fit.
- T015–T017: routing/soft-geometry alternatives expose limited or unsafe transfer paths.
- T018–T019: utility-aware hard geometry is viable and fresh-qualified for heterogeneous shifts.
- T020-A: the heterogeneous geometry selector is not broadly safe on a fresh non-spatial cohort.
- T020-B: the ideal 1% non-spatial deadband target itself is development-safe with zero harmful moves.
- T020-C: unchanged in-domain three-way direction learning remains unsafe.
- T020-D: oracle attribution is direction-dominant.
- **T020-E: binary lower-vs-upper factorization does not materially close the direction gap; the simple classifier-repair route is closed for now.**

## Information-boundary rules

- Test-time adaptation/selection must never consume test labels, clean targets, degradation masks/gain maps, condition IDs, annotations, semantic image IDs, reference gradients/Jacobians, oracle values, or evaluation metrics.
- Source/development references may be used only in explicitly declared training/calibration/diagnostic stages.
- Held-out/fresh decisions and enhanced outputs must be finalized and persisted before their reference metrics, labels, families, or oracles are attached.
- Inspected fresh IDs are permanently excluded from corrective fresh cohorts.
- Failed fresh cohorts may motivate a development-only hypothesis but their per-row references/logits/features/outcomes may not tune that hypothesis.
- OOF claims must be image-grouped; held-out decisions must not depend on held-out labels/reference values.
- Development-only target/oracle/OOF evidence is not fresh qualification.
- Fresh runs must fail closed on source/provenance/preparation binding mismatches.

## Interpretation

The main paper-level method story is now image-enhancement-first. T014 establishes the **optimization-field principle**: a reference-free test-time energy is useful when its derivatives are restoration-useful, not merely when its scalar values fit a reference loss. T018–T019 add a secondary **utility-aware geometry principle** for heterogeneous spatial degradation.

T020 defines the boundary of that extension. The ideal non-spatial target is safe, and a direction oracle shows headroom, but both the original three-way learner and the simplest binary sign decomposition fail the same clean-safety contract. Continuing with small classifier patches would risk development overfitting without addressing the paper's larger evidence gaps.

The active validation priority therefore shifts from geometry rescue to **image-enhancement evidence**. The first question is whether T014's already accepted fresh gain transfers beyond MSE to a standard structural metric on the exact frozen outputs. Real-world enhancement data and test-time efficiency remain later paper-level gaps, to be handled in separate scoped cycles. Downstream detection is not an active requirement for the current enhancement-focused validation plan.

## Milestones

- T001–T013: completed mechanism/diagnostic sequence.
- **T014: COMPLETED — Sobolev inner objective passed 8/8 development and 12/12 fresh qualification.**
- T015–T018: completed routing/geometry diagnostics; exact-direction selector fresh negative.
- T019-A/B/C: completed deadband target, OOF selector, and final freeze.
- **T019-D: COMPLETED — one-shot fresh heterogeneous geometry qualification positive 5/5.**
- **T020-A: COMPLETED — one-shot fresh non-spatial safety negative 3/4.**
- **T020-B: COMPLETED — development-only non-spatial fixed-1% target positive 5/5, zero harmful.**
- **T020-C: COMPLETED — development-only non-spatial grouped-OOF negative 3/5.**
- **T020-D: COMPLETED — frozen OOF attribution is direction-dominant.**
- **T020-E: COMPLETED — frozen-necessity binary-sign repair negative 3/5.**
- **T021-A: ACTIVE — frozen-fresh SSIM transfer audit for T014.**

## Current open task

`T021-A — frozen-fresh SSIM transfer audit for T014` in `coordination/CHATGPT_TO_CODEX.md`.

Use only exact accepted T014 fresh frozen outputs and provenance; do not rerun TTT. Compute fixed full-RGB Gaussian-window SSIM and an image-clustered 10,000-resample paired bootstrap. Positive iff the two-sided 95% cluster-bootstrap CI for pooled mean `ΔSSIM = SSIM(T014)-SSIM(baseline)` has lower bound above zero. If exact frozen artifact triples cannot be recovered without rerunning TTT, stop as structurally unsupported.
