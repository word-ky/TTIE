# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Processing for Task-Oriented Vision**

## Core scientific question

Can a vision system adapt a compact spatial correction field per test image, without test labels or clean targets, so that spatially heterogeneous nuisance degradation is corrected before a frozen downstream task model processes the image?

## Current scientific state

T014 establishes a fully fresh-qualified canonical hard-Region2 TTT-ISP: a frozen nuisance gate, a source-supervised Sobolev restoration energy whose derivative field drives label-free projected test-time updates, and a fixed spatial basis. It remains the broad fresh-qualified baseline because its original qualification covers clean, homogeneous, and heterogeneous conditions.

T016-T019 isolate and solve the adaptive-geometry subproblem on the prescribed heterogeneous spatial protocol. Hard boundary placement has real oracle headroom; the local x/y problem nearly factorizes; the frozen 28-D candidate representation contains direction information; direct exact-direction supervision is too eager to move on fresh near-zero-headroom quadrants; and a fixed family-agnostic 1% utility deadband yields a safer target.

T019-B provides the controlled development intervention: with representation, folds, network, optimizer, seed and epochs held fixed, changing only exact-direction labels to 1%-deadband labels reduces harmful OOF episodes `11→5`, moving episodes `68→51`, both-axis moves `27→13`, preserves zero quadrant harm, and passes 7/7 predeclared development clauses.

T019-C freezes exactly one all-development x head and one y head. T019-D then applies that immutable selector once to a new deterministic disjoint 40-image / 120-episode cohort. **T019-D passes all 5/5 predeclared fresh geometry clauses** with pooled `H1/H0=0.952464`, pooled `H1/H*=1.027465`, offset `0.874063`, left/right `1.004090`, and quadrants `1.000000`. Outcomes are `43 beneficial / 73 equal / 4 harmful`; all four harmful cases are left/right. All 40 fresh quadrants remain canonical center, with zero moves and zero harm. PR #35 is merged as `f50a3b6a027f647efebf46183e934f8bba423882`.

This changes the scientific status: **the utility-aware adaptive hard-boundary selector is now fresh-qualified on the specified heterogeneous spatial protocol.** However, T019-D does not test clean or spatially homogeneous exposure shifts. Therefore T019 is not yet promoted above T014 as the globally deployable Ours. The immediate open question is whether the added geometry selector is safely non-degrading outside heterogeneous shifts.

## Best current methods

### Broad fresh-qualified baseline

**T014 Sobolev Region2 TTT** remains the broad deployable baseline:

1. frozen T006 CLIP exposure readout and T007 clean-abstention gate;
2. frozen T014 28-D Sobolev restoration energy;
3. canonical hard Region2 projected EV+gamma state from identity;
4. exactly 40 label-free projected updates when active;
5. minimum predicted-energy checkpoint, earliest tie;
6. all label-free trajectories/decisions persisted before clean-reference evaluation.

T014 Stage A passes 8/8 development clauses and frozen Stage B passes 12/12 fresh clauses on 40 unseen images. Fresh heterogeneous MSE is `0.03385803`; unseen-calibration gradient alignment is `73/74` positive with median cosine `0.93606`.

### Fresh-qualified heterogeneous geometry extension

**T019 = T014 + frozen 1%-deadband hard-boundary selector** is now fresh-qualified on the three prescribed heterogeneous spatial conditions. It uses the same T014 trajectory and five hard-cross `tau=0` candidates, the frozen 28-D candidate features, and the immutable T019-C x/y heads. T019-D improves pooled MSE by about 4.75% versus canonical Region2 and preserves quadrants exactly.

The extension may replace T014 as the main Ours only after non-spatial safety is independently checked. Downstream detector/task validation remains a separate paper-level gap and must not be mixed into the current safety task.

## Established controlled findings

- **T001-T003:** spatial ISP capacity can outperform global correction on favorable heterogeneous shifts; capacity and simple absolute-statistic priors alone are insufficient.
- **T004-T007:** zero-shot exposure signals are content-confounded; a source-trained frozen-CLIP exposure readout plus clean-abstention calibration provides a usable nuisance signal.
- **T008-T012:** semantic EV+gamma TTT contains useful directions but suffers drift, over-correction, stopping and coupling issues; learned checkpoint selection cannot rescue an oracle-limited trajectory.
- **T013:** scalar restoration-value fitting is insufficient for a good TTT derivative field.
- **T014:** Sobolev/derivative supervision is the first fully qualified learned-inner-objective result; it supports the claim that for gradient-based TTT the learned optimization field matters more than scalar value fit.
- **T015:** routing among global/bilinear2/Region2 is negative; oracle headroom is only 1.94%.
- **T016-A:** shiftable hard-boundary capacity is positive; 27-candidate oracle is `0.92764×` canonical Region2.
- **T016-B-F:** scalar energy/value/ranking/confidence approaches are not robust enough for hard-boundary selection.
- **T017:** `tau=0.05` soft geometry carries diagnostic signal but soft→hard transfer and matched-soft deployment are unsafe; the soft-renderer route is closed.
- **T018-A:** hard local x/y reference target is positive (5/5), pooled `0.930464×` Region2 and `1.001327×` nine-hard oracle, with zero harmful development moves.
- **T018-B:** frozen T014 scalar Sobolev energy fails even under the correct local decision geometry (0/5).
- **T018-C:** direct axis-direction classification on the unchanged frozen representation passes 5/5 grouped OOF, establishing representation-level direction sufficiency.
- **T018-E:** frozen exact-direction selector is a one-shot fresh negative (4/5): pooled/offset/left-right transfer, but quadrants reach `1.011785×` Region2 because three unnecessary x moves occur. This cohort is burned for corrective tuning.
- **T019-A:** fixed 1% utility-deadband reference target passes 5/5 development plus zero harmful moves; it retains 97.6018% of pooled nine-hard oracle headroom.
- **T019-B:** label-only grouped-OOF intervention passes 7/7; harmful episodes fall `11→5` and quadrants remain zero-harm.
- **T019-C:** final all-development 1% deadband selector frozen; engineering barrier only.
- **T019-D:** **one-shot fresh heterogeneous geometry qualification positive (5/5)**. Pooled `H1/H0=0.952464`, pooled `H1/H*=1.027465`, offset `0.874063`, left/right `1.004090`, quadrants `1.000000`; `43/73/4` beneficial/equal/harmful; quadrants `0/40/0` with zero movement.

## Information-boundary status

T019-D repairs the provenance weakness discovered after T018-E. The accepted pipeline contemporaneously hash-binds exclusion, manifest, manifest-frozen, mapping, input index, preparation metadata, scientific source/assets, feature rows and selector decisions. All 120 decisions are independently replayed exactly before clean/reference evaluation opens.

Non-negotiable rules remain:

- test-time adaptation/selection must never consume test labels, clean targets, degradation masks/gain maps, condition IDs, annotations, image IDs as semantic shortcuts, source-only reference gradients/Jacobians, or evaluation metrics;
- source/development clean references may be used only in explicitly declared training/calibration/diagnostic stages;
- fresh image decisions must be finalized and persisted before reference metrics, family labels or oracles are attached;
- inspected fresh IDs are permanently excluded from future corrective fresh cohorts;
- development-only oracle/OOF diagnostics are not fresh qualification;
- fresh qualification must fail closed on source/provenance/preparation binding mismatches.

## Interpretation of strongest evidence

The method story now has two aligned levels.

First, T014 shows an **objective-field principle**: a test-time energy should learn restoration-useful derivatives, not merely scalar restoration values. Sobolev derivative supervision dramatically improves unseen gradient alignment and fresh adaptation.

Second, T018-T019 show a **utility-aware geometry principle**: adaptive hard-boundary direction information exists in the frozen representation, but movement should be supervised only when the expected utility is material. The fixed 1% deadband converts a direction classifier from a fresh quadrants-safety failure into a new-cohort 5/5 heterogeneous geometry qualification without test labels or clean targets in the decision path.

The remaining deployment question before promoting T019 over T014 is non-spatial safety. After that barrier, the project should pivot toward the larger paper-level gaps: real adverse-image distributions, frozen downstream detector/task metrics, and test-time computational cost.

## Milestones

- **M0 / T001:** completed — mechanism scaffold.
- **M1 / T002:** completed — spatiality/capacity controls.
- **M2 / T003:** completed — simple-prior rescue negative.
- **M3 / T004-T008:** completed — nuisance readout qualified; original semantic TTT unqualified.
- **M4 / T009:** completed — objective/action geometry diagnosis.
- **M5 / T010:** completed — residual-target repair negative.
- **M6 / T011:** completed — projected Region2 strong but fresh qualification negative.
- **M7 / T012:** completed — learned stopping negative; oracle-limited.
- **M8 / T013:** completed — scalar learned inner objective negative.
- **M9 / T014:** **COMPLETED — SOBOLEV INNER OBJECTIVE PASSED 8/8 DEVELOPMENT AND 12/12 FRESH QUALIFICATION.**
- **M10 / T015:** completed — cross-basis routing negative.
- **M11 / T016:** completed — hard-boundary headroom established; scalar/ranking/confidence selectors closed.
- **M12 / T017:** completed — soft diagnostic signal identified; soft-transfer/deployment route closed.
- **M13 / T018:** completed — direct hard-axis direction representation established; exact-direction frozen selector fresh negative 4/5.
- **M14 / T019-A:** completed — 1% utility-deadband target positive.
- **M14 / T019-B:** completed — grouped-OOF deadband selector positive 7/7.
- **M14 / T019-C:** completed — immutable all-development deadband selector frozen.
- **M14 / T019-D:** **COMPLETED — ONE-SHOT FRESH HETEROGENEOUS GEOMETRY QUALIFICATION POSITIVE 5/5.**
- **M15 / T020-A:** **ACTIVE — ONE-SHOT FRESH NON-SPATIAL SAFETY QUALIFICATION.**

## Current open task

`T020-A — one-shot fresh non-spatial safety qualification of the frozen T019 selector` in `coordination/CHATGPT_TO_CODEX.md`.

T020-A must use one new deterministic disjoint 40-image cohort, exactly the accepted T014 clean/homogeneous-dark/homogeneous-bright settings, and the unchanged T019-C selector. It is a pure safety test versus canonical T014 Region2: pooled and each condition must satisfy `mean(H1) <= 1.01 × mean(H0)`. No training, tuning, downstream detector experiment, or second cohort is authorized in this cycle.