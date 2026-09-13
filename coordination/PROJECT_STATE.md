# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Processing for Task-Oriented Vision**

## Core scientific question

Can a vision system adapt a compact spatial correction field per test image, without test labels or clean targets, so that spatially heterogeneous nuisance degradation is corrected before a frozen downstream task model processes the image?

## Current scientific state

**T014 Sobolev Region2 TTT remains the broad fresh-qualified Ours.** It combines the frozen T006/T007 nuisance readout + clean-abstention gate, the source-supervised T014 Sobolev restoration energy, canonical hard Region2 EV+gamma adaptation, 40 projected label-free updates when active, and minimum predicted-energy checkpoint selection. T014 passed 8/8 development and 12/12 fresh clauses; fresh heterogeneous MSE is `0.03385803`, and unseen-calibration gradient alignment is `73/74` positive with median cosine `0.93606`.

**T019 remains a heterogeneous-only fresh-qualified geometry extension, not the broad default.** T019-B changed only supervision from exact direction to a fixed 1% utility-deadband target while holding representation/folds/network/optimizer/seed/epochs fixed; it passed 7/7 grouped-OOF development clauses and reduced harmful episodes `11→5`. T019-C froze the final two-head selector. T019-D then passed 5/5 on a new 40-image / 120-episode heterogeneous fresh cohort: pooled `H1/H0=0.952464`, oracle proximity `1.027465`, offset `0.874063`, left/right `1.004090`, quadrants `1.000000`, with `43 beneficial / 73 equal / 4 harmful` and zero quadrant movement.

**T020-A is a valid fresh non-spatial negative (3/4).** On a second new 40-image / 120-episode cohort, the frozen T019-C selector gives pooled `0.986616×`, clean `1.046125×` **fail**, homogeneous-dark `0.994103×`, homogeneous-bright `0.963036×`. Clean outcomes are `0 beneficial / 39 equal / 1 harmful`; that fresh cohort is burned for corrective tuning.

**T020-B is a development-only non-spatial target positive (5/5).** The unchanged fixed `delta=0.01` per-axis reference deadband target, audited on the accepted 40 development images under clean / homogeneous-dark / homogeneous-bright, gives pooled `0.900891×`, clean `0.721586×`, dark `0.906699×`, bright `0.889559×`, with `61 beneficial / 59 equal / 0 harmful`. Thus the target principle itself is viable on these development cases.

**T020-C is now a development-only non-spatial OOF negative (3/5).** It kept the frozen T019 28-D representation, the historical five image-grouped folds, and the exact T019-B `84→64→64→3` unweighted-CE learner fixed. Results:

- pooled `H1/H0 = 0.947906080161726` — pass;
- clean `1.2072654157145626` — fail;
- homogeneous-dark `0.9370790223162285` — pass;
- homogeneous-bright `0.9701882237639385` — pass;
- clean harmful count = `1` — fail (required zero).

Pooled outcomes are `37 beneficial / 63 equal / 20 harmful`; clean `0/39/1`; dark `21/15/4`; bright `16/9/15`. Bright target agreement is only x `14/40`, y `18/40`, joint `6/40`. All 120 OOF decisions were frozen and independently replayed before held-out reference evaluation, so this is a valid scientific negative rather than a leakage artifact.

This materially changes the interpretation: **training-domain coverage alone is not an adequate explanation for T020-A.** The 1% reference target is viable on the non-spatial development set (T020-B), but the unchanged three-way direct-direction representation/readout does not recover broad safety even when trained in-domain (T020-C). Combined heterogeneous+non-spatial final training is therefore blocked pending mechanistic failure attribution.

The immediate active task is **T020-D**, a frozen-evidence diagnostic that decomposes T020-C errors into movement-necessity versus direction-sign failures and compares two fixed reference-only oracle counterfactuals. No model is trained in this cycle.

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
- T018-C: unchanged frozen representation supports grouped-OOF direct direction prediction on heterogeneous development data.
- T018-E: exact-direction frozen selector fresh negative 4/5 due unnecessary quadrant moves.
- T019-A: fixed 1% utility-deadband target positive on heterogeneous development with zero harm.
- T019-B: deadband-label OOF selector positive 7/7, fewer harmful moves.
- T019-D: one-shot heterogeneous fresh geometry positive 5/5.
- T020-A: one-shot non-spatial fresh safety negative 3/4; clean mean `1.046125×` fails.
- T020-B: development-only non-spatial fixed-1% target positive 5/5 with zero harmful moves.
- **T020-C: unchanged non-spatial in-domain OOF learner negative 3/5; pooled/dark/bright means pass, but clean safety and zero-harmful fail.**

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

The method story has two established levels. T014 establishes the **objective-field principle**: useful test-time energies need restoration-useful derivatives, not merely accurate scalar values. T018–T019 establish a **utility-aware geometry principle** for heterogeneous shifts: direction information exists in the frozen representation, but movement should be supervised only when expected utility is material.

T020 now sharpens the limit of that geometry extension. The ideal 1% target remains safe on non-spatial development cases (T020-B), but simply exposing the unchanged three-way classifier to those non-spatial labels does not make it safely learnable (T020-C). The next scientific question is not whether to add more data indiscriminately; it is whether the failures come chiefly from deciding **whether to move** or from deciding **which direction to move**. T020-D performs only that attribution. Larger paper-level gaps—downstream detector metrics, real adverse-image distributions, and test-time cost—remain deferred until this selector-safety branch is closed.

## Milestones

- T001–T013: completed mechanism/diagnostic sequence.
- **T014: COMPLETED — Sobolev inner objective passed 8/8 development and 12/12 fresh qualification.**
- T015–T018: completed routing/geometry diagnostics; exact-direction selector fresh negative.
- T019-A/B/C: completed deadband target, OOF selector, and final freeze.
- **T019-D: COMPLETED — one-shot fresh heterogeneous geometry qualification positive 5/5.**
- **T020-A: COMPLETED — one-shot fresh non-spatial safety negative 3/4; clean `1.046125×`.**
- **T020-B: COMPLETED — development-only non-spatial fixed-1% target positive 5/5, zero harmful.**
- **T020-C: COMPLETED — development-only non-spatial grouped-OOF negative 3/5.**
- **T020-D: ACTIVE — frozen OOF movement-vs-direction failure-attribution audit.**

## Current open task

`T020-D — frozen OOF movement-vs-direction failure attribution audit` in `coordination/CHATGPT_TO_CODEX.md`.

Use only immutable T020-C OOF logits/decisions and the accepted T020-B development target/reference table. No training, no feature recomputation, no thresholds, no fresh artifacts, and no T020-A per-row data. Stop after classifying the failure as necessity-dominant, direction-dominant, both individually sufficient/mixed, or neither sufficient/interaction-or-representation-limited under the predeclared oracle counterfactual rules.