# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Processing for Task-Oriented Vision**

## Core scientific question

Can a vision system adapt a compact spatial correction field per test image, without test labels or clean targets, so that spatially heterogeneous nuisance degradation is corrected before a frozen downstream task model processes the image?

## Current scientific state

**T014 Sobolev Region2 TTT remains the broad fresh-qualified Ours.** It combines the frozen T006/T007 nuisance readout + clean-abstention gate, the source-supervised T014 Sobolev restoration energy, canonical hard Region2 EV+gamma adaptation, 40 projected label-free updates when active, and minimum predicted-energy checkpoint selection. T014 passed 8/8 development and 12/12 fresh clauses; fresh heterogeneous MSE is `0.03385803`, and unseen-calibration gradient alignment is `73/74` positive with median cosine `0.93606`.

**T019 remains a heterogeneous-only fresh-qualified geometry extension, not the broad default.** T019-B changed only supervision from exact direction to a fixed 1% utility-deadband target while holding representation/folds/network/optimizer/seed/epochs fixed; it passed 7/7 grouped-OOF development clauses and reduced harmful episodes `11→5`. T019-C froze the final two-head selector. T019-D passed 5/5 on a new 40-image / 120-episode heterogeneous fresh cohort: pooled `H1/H0=0.952464`, oracle proximity `1.027465`, offset `0.874063`, left/right `1.004090`, quadrants `1.000000`, with `43 beneficial / 73 equal / 4 harmful` and zero quadrant movement.

**T020-A is a valid fresh non-spatial negative (3/4).** On a second new 40-image / 120-episode cohort, the frozen T019-C selector gives pooled `0.986616×`, clean `1.046125×` **fail**, homogeneous-dark `0.994103×`, homogeneous-bright `0.963036×`. Clean outcomes are `0 beneficial / 39 equal / 1 harmful`; that fresh cohort is burned for corrective tuning.

**T020-B is a development-only non-spatial target positive (5/5).** The unchanged fixed `delta=0.01` per-axis reference deadband target, audited on the accepted 40 development images under clean / homogeneous-dark / homogeneous-bright, gives pooled `0.900891×`, clean `0.721586×`, dark `0.906699×`, bright `0.889559×`, with `61 beneficial / 59 equal / 0 harmful`. Thus the target principle itself is viable on these development cases.

**T020-C is a development-only non-spatial OOF negative (3/5).** It kept the frozen T019 28-D representation, historical five image-grouped folds, and the exact T019-B `84→64→64→3` unweighted-CE learner fixed. Results are pooled `0.947906×` pass, clean `1.207265×` fail, dark `0.937079×` pass, bright `0.970188×` pass, and clean harmful count `1` fail. Pooled outcomes are `37 beneficial / 63 equal / 20 harmful`. This shows that simply adding non-spatial in-domain labels to the unchanged three-way direct-direction learner does not recover broad safety.

**T020-D is now a development-only mechanistic positive diagnosis: the T020-C failure is direction-dominant under the predeclared oracle test.** Using only frozen T020-C OOF predictions and accepted T020-B targets/nine-hard reference values, with no training or feature recomputation:

- necessity-oracle / frozen predicted-sign A passes `3/5`: pooled `0.934817×`, clean `1.229472×` fail, dark `0.927146×`, bright `0.948885×`, clean harmful `2`;
- frozen-necessity / direction-oracle B passes `5/5`: pooled `0.916320×`, clean `0.754146×`, dark `0.915794×`, bright `0.920111×`, clean harmful `0`.

Among the 20 harmful T020-C episodes, `19` contain a wrong-direction axis, `5` a false-move axis, and `6` a missed-move axis; exact overlaps are `9` wrong-direction only, `6` missed+wrong, `4` false+wrong, and `1` false-only. The original clean harmful row contains a missed x move plus a wrong y direction, not a false move. Therefore the earlier false-movement-dominance hypothesis is rejected: under this fixed reference diagnostic, correcting direction sign while preserving the original movement decisions is individually sufficient to satisfy all five non-spatial clauses, whereas perfect movement necessity with the frozen sign signal is not.

This does **not** establish a deployable repair. Counterfactual B is reference-only and still contains two harmful homogeneous-bright episodes; it only identifies direction sign as the next mechanism to test. Combined heterogeneous+non-spatial final training remains blocked until a learned sign probe succeeds without held-out reference information.

The immediate active task is **T020-E**, a development-only grouped-OOF probe that freezes T020-C move/no-move decisions and trains only a binary lower-vs-upper sign head on the unchanged 84-D representation and fold-training non-center targets.

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
- T020-C: unchanged non-spatial in-domain OOF learner negative 3/5; pooled/dark/bright means pass, but clean safety and zero-harmful fail.
- **T020-D: frozen oracle attribution is direction-dominant — necessity oracle 3/5, direction oracle 5/5; 19/20 harmful episodes contain wrong-direction error.**

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

T020 sharpens the limit of that geometry extension. The ideal 1% target is safe on non-spatial development cases (T020-B), yet the unchanged symmetric three-way classifier is unsafe even in-domain (T020-C). T020-D resolves the first-order failure attribution: under the fixed oracle counterfactuals, **direction sign is the dominant bottleneck**, not false movement. The next question is therefore whether the same 84-D representation supports a clean binary lower-vs-upper readout when movement necessity is held fixed. Larger paper-level gaps—downstream detector metrics, real adverse-image distributions, and test-time cost—remain deferred until this selector-safety branch is closed.

## Milestones

- T001–T013: completed mechanism/diagnostic sequence.
- **T014: COMPLETED — Sobolev inner objective passed 8/8 development and 12/12 fresh qualification.**
- T015–T018: completed routing/geometry diagnostics; exact-direction selector fresh negative.
- T019-A/B/C: completed deadband target, OOF selector, and final freeze.
- **T019-D: COMPLETED — one-shot fresh heterogeneous geometry qualification positive 5/5.**
- **T020-A: COMPLETED — one-shot fresh non-spatial safety negative 3/4; clean `1.046125×`.**
- **T020-B: COMPLETED — development-only non-spatial fixed-1% target positive 5/5, zero harmful.**
- **T020-C: COMPLETED — development-only non-spatial grouped-OOF negative 3/5.**
- **T020-D: COMPLETED — frozen OOF attribution is direction-dominant (A 3/5, B 5/5).**
- **T020-E: ACTIVE — frozen-necessity + binary-direction grouped-OOF sufficiency probe.**

## Current open task

`T020-E — frozen-necessity + binary-direction grouped-OOF probe` in `coordination/CHATGPT_TO_CODEX.md`.

Use the exact frozen T020-C non-spatial features, historical image-grouped folds and training-only normalization; train only `84→64→64→2` lower/upper heads on fold-training rows whose axis target is non-center. Held-out movement necessity must remain exactly the already frozen T020-C center-vs-move decision. Freeze all 120 combined decisions before opening held-out references. No new necessity head, no thresholds, no heterogeneous combined training, no fresh data, and no T020-A per-row artifacts.