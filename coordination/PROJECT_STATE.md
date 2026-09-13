# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Processing for Task-Oriented Vision**

## Core scientific question

Can a vision system adapt a compact spatial correction field per test image, without test labels or clean targets, so that spatially heterogeneous nuisance degradation is corrected before a frozen downstream task model processes the image?

## Current scientific state

**T014 Sobolev Region2 TTT remains the broad fresh-qualified Ours.** It combines the frozen T006/T007 nuisance readout + clean-abstention gate, the source-supervised T014 Sobolev restoration energy, canonical hard Region2 EV+gamma adaptation, 40 projected label-free updates when active, and minimum predicted-energy checkpoint selection. T014 passed 8/8 development and 12/12 fresh clauses; fresh heterogeneous MSE is `0.03385803`, and unseen-calibration gradient alignment is `73/74` positive with median cosine `0.93606`.

**T019 is a heterogeneous-only fresh-qualified geometry extension, not the broad default.** T019-B changed only supervision from exact direction to a fixed 1% utility-deadband target while holding representation/folds/network/optimizer/seed/epochs fixed; it passed 7/7 grouped-OOF development clauses and reduced harmful episodes `11→5`. T019-C froze the final two-head selector. T019-D then passed 5/5 on a new 40-image / 120-episode heterogeneous fresh cohort: pooled `H1/H0=0.952464`, oracle proximity `1.027465`, offset `0.874063`, left/right `1.004090`, quadrants `1.000000`, with `43 beneficial / 73 equal / 4 harmful` and zero quadrant movement.

**T020-A is a valid fresh non-spatial negative (3/4).** On a second new 40-image / 120-episode cohort, the frozen T019-C selector gives pooled `0.986616×`, clean `1.046125×` **fail**, homogeneous-dark `0.994103×`, homogeneous-bright `0.963036×`. Clean outcomes are `0 beneficial / 39 equal / 1 harmful`; that fresh cohort is burned for corrective tuning.

**T020-B is now a development-only non-spatial target positive (5/5).** The exact same fixed `delta=0.01` per-axis reference deadband target was audited on the accepted 40 development images under clean / homogeneous-dark / homogeneous-bright, with no selector training and no threshold search. Results:

- pooled `H_delta/H0 = 0.9008910034988411`;
- clean `0.7215855454024681`;
- homogeneous-dark `0.9066994112406046`;
- homogeneous-bright `0.8895594969478190`;
- outcomes `61 beneficial / 59 equal / 0 harmful`.

Clean alone is `2/38/0`. Thus the target principle is safe on these non-spatial development cases. This narrows the current bottleneck from target design toward **learned movement generalization / training-domain coverage**, while remaining development-only evidence.

The immediate active task is **T020-C**, a strict image-grouped OOF probe on only the T020-B non-spatial development episodes. It keeps the frozen 28-D representation and the exact T019-B `84→64→64→3` learner fixed, and asks whether the same learner can predict the conservative target in-domain before any heterogeneous+non-spatial combined training is attempted.

## Best current methods

### Broad fresh-qualified baseline

**T014 Sobolev Region2 TTT** — broad deployable Ours.

### Heterogeneous-only fresh-qualified extension

**T019 = T014 + frozen 1%-deadband hard-boundary selector** — fresh-qualified only on the prescribed heterogeneous spatial protocol. T020-A prevents promotion to a universal default.

## Strongest controlled findings

- T001–T003: spatial ISP capacity helps heterogeneous shifts; capacity/simple priors alone are insufficient.
- T004–T007: zero-shot exposure signals are content-confounded; frozen source-trained nuisance readout + clean abstention is usable.
- T008–T012: semantic TTT has useful directions but drift/stopping/coupling problems; learned checkpointing cannot rescue oracle-limited trajectories.
- T013: scalar restoration-value fitting does not sufficiently constrain the TTT derivative field.
- **T014:** Sobolev derivative supervision is the first fully qualified learned inner objective; the learned optimization field matters more than scalar value fit.
- T015: cross-basis routing negative; oracle headroom only 1.94%.
- T016: hard-boundary headroom exists; scalar energy/value/ranking/confidence selectors are not robust enough.
- T017: soft geometry is diagnostically useful but unsafe as transfer/deployment route.
- T018-A: hard local x/y reference target positive, near nine-hard oracle with zero harmful development moves.
- T018-B: frozen T014 scalar energy fails for local geometry direction.
- T018-C: unchanged frozen representation supports grouped-OOF direct direction prediction.
- T018-E: exact-direction frozen selector fresh negative 4/5 due unnecessary quadrant moves.
- T019-A: fixed 1% utility-deadband target positive on heterogeneous development with zero harm.
- T019-B: deadband-label OOF selector positive 7/7, fewer harmful moves.
- T019-D: one-shot heterogeneous fresh geometry positive 5/5.
- T020-A: one-shot non-spatial fresh safety negative 3/4; clean mean `1.046125×` fails.
- **T020-B: development-only non-spatial fixed-1% target positive 5/5 with zero harmful moves.**

## Information-boundary rules

- Test-time adaptation/selection must never consume test labels, clean targets, degradation masks/gain maps, condition IDs, annotations, semantic image IDs, reference gradients/Jacobians, oracle values, or evaluation metrics.
- Source/development references may be used only in explicitly declared training/calibration/diagnostic stages.
- Held-out/fresh decisions must be finalized and persisted before their reference metrics, labels, families, or oracles are attached.
- Inspected fresh IDs are permanently excluded from corrective fresh cohorts.
- Failed fresh cohorts may motivate a development-only hypothesis but their per-row references/logits/features/outcomes may not tune that hypothesis.
- OOF claims must be image-grouped; held-out decisions must not depend on held-out labels/reference values.
- Development-only target/oracle/OOF evidence is not fresh qualification.
- Fresh runs must fail closed on source/provenance/preparation binding mismatches.

## Interpretation

The method story has two levels. T014 establishes the **objective-field principle**: useful test-time energies need restoration-useful derivatives, not merely accurate scalar values. T018–T019 establish a **utility-aware geometry principle** for heterogeneous shifts: direction information exists in the frozen representation, but movement should be supervised only when expected utility is material.

T020-A/T020-B sharpen the limitation. The heterogeneous-trained predictor is not broad clean-safe, while the ideal fixed deadband target is safe on non-spatial development cases. The current question is therefore whether the unchanged representation/learner is sufficient **in-domain** on non-spatial cases. T020-C answers only that question. Larger paper-level gaps—downstream detector metrics, real adverse-image distributions, and test-time cost—remain deferred until this selector-safety branch is closed.

## Milestones

- T001–T013: completed mechanism/diagnostic sequence.
- **T014: COMPLETED — Sobolev inner objective passed 8/8 development and 12/12 fresh qualification.**
- T015–T018: completed routing/geometry diagnostics; exact-direction selector fresh negative.
- T019-A/B/C: completed deadband target, OOF selector, and final freeze.
- **T019-D: COMPLETED — one-shot fresh heterogeneous geometry qualification positive 5/5.**
- **T020-A: COMPLETED — one-shot fresh non-spatial safety negative 3/4; clean `1.046125×`.**
- **T020-B: COMPLETED — development-only non-spatial fixed-1% target positive 5/5, zero harmful.**
- **T020-C: ACTIVE — non-spatial image-grouped OOF deadband-direction sufficiency probe.**

## Current open task

`T020-C — non-spatial grouped-OOF deadband-direction sufficiency probe` in `coordination/CHATGPT_TO_CODEX.md`.

Use only the accepted T020-B 40-image / 120-episode non-spatial development set, the unchanged frozen T019 28-D representation, the exact historical five image-grouped folds, and the unchanged T019-B learner. Freeze all held-out predictions before held-out reference evaluation. No combined-domain final selector, fresh cohort, confidence gate, threshold search, architecture/feature change, detector experiment, or real low-light benchmark is authorized in this cycle.