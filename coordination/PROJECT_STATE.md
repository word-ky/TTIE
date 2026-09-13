# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Processing for Task-Oriented Vision**

## Core scientific question

Can a vision system adapt a compact spatial correction field per test image, without test labels or clean targets, so that spatially heterogeneous nuisance degradation is corrected before a frozen downstream task model processes the image?

## Current scientific state

T014 remains the broad fresh-qualified baseline. It combines a frozen nuisance gate, a source-supervised Sobolev restoration energy whose derivative field drives label-free projected test-time updates, and a canonical hard Region2 spatial basis. Its original qualification covers clean, homogeneous, and heterogeneous conditions.

T016–T019 solve a narrower adaptive-geometry problem on the prescribed heterogeneous spatial protocol. Hard boundary placement has meaningful oracle headroom; the local x/y problem nearly factorizes; the frozen 28-D candidate representation contains direction information; exact-direction supervision moves too eagerly on fresh near-zero-headroom cases; and a fixed family-agnostic 1% utility deadband produces a safer movement target.

T019-B is the clean development intervention: representation, folds, network, optimizer, seed and epochs are held fixed while only the target changes from exact direction to 1%-deadband direction. Harmful OOF episodes fall `11→5`, moving episodes `68→51`, both-axis moves `27→13`, quadrants retain zero harm, and all 7/7 predeclared development clauses pass.

T019-C freezes exactly one final x head and one final y head. T019-D applies that immutable selector once to a new deterministic disjoint 40-image / 120-episode heterogeneous cohort and passes **5/5**: pooled `H1/H0=0.952464`, pooled `H1/H*=1.027465`, offset `0.874063`, left/right `1.004090`, quadrants `1.000000`, with `43 beneficial / 73 equal / 4 harmful`. All 40 quadrants remain canonical center with zero movement and zero harm.

T020-A then tests the same frozen selector on a second new deterministic disjoint 40-image / 120-episode non-spatial cohort containing `clean`, `homogeneous_dark`, and `homogeneous_bright`. It is a valid one-shot fresh **negative (3/4)**. Pooled, dark, and bright safety pass, but clean fails the predeclared 1% mean criterion:

- pooled `H1/H0 = 0.9866159548610750`;
- clean `H1/H0 = 1.0461253551018836` — fail;
- homogeneous-dark `H1/H0 = 0.9941025363625269`;
- homogeneous-bright `H1/H0 = 0.9630361903920437`.

Clean outcomes are `0 beneficial / 39 equal / 1 harmful`. Six clean episodes move x; five are MSE-equal and one is harmful. The result blocks promotion of T019 as a broad default. The T020-A fresh cohort is burned for corrective tuning.

The immediate scientific question is now **target versus predictor**. T019-A proved the 1% deadband target on heterogeneous development episodes only. T020-B therefore audits the exact same fixed target, without training, on clean and homogeneous development cases. This will determine whether the target principle itself is broadly safe or whether the learned selector is the remaining failure point.

## Best current methods

### Broad fresh-qualified baseline

**T014 Sobolev Region2 TTT** remains the broad deployable Ours:

1. frozen T006 CLIP exposure readout and T007 clean-abstention gate;
2. frozen T014 28-D Sobolev restoration energy;
3. canonical hard Region2 projected EV+gamma state from identity;
4. exactly 40 label-free projected updates when active;
5. minimum predicted-energy checkpoint, earliest tie;
6. all label-free trajectories/decisions persisted before clean-reference evaluation.

T014 Stage A passes 8/8 development clauses and frozen Stage B passes 12/12 fresh clauses on 40 unseen images. Fresh heterogeneous MSE is `0.03385803`; unseen-calibration gradient alignment is `73/74` positive with median cosine `0.93606`.

### Heterogeneous-only fresh-qualified geometry extension

**T019 = T014 + frozen 1%-deadband hard-boundary selector** is fresh-qualified on the three prescribed heterogeneous spatial conditions via T019-D. It uses the unchanged T014 trajectory, five hard-cross `tau=0` candidates, frozen 28-D candidate features, and immutable T019-C x/y heads.

T020-A shows this extension is **not broadly clean-safe under the predeclared criterion**, so T019 must remain a heterogeneous-only extension rather than replacing T014 globally.

## Established controlled findings

- **T001–T003:** spatial ISP capacity can outperform global correction on favorable heterogeneous shifts; capacity and simple absolute-statistic priors alone are insufficient.
- **T004–T007:** zero-shot exposure signals are content-confounded; a source-trained frozen-CLIP exposure readout plus clean-abstention calibration provides a usable nuisance signal.
- **T008–T012:** semantic EV+gamma TTT contains useful directions but suffers drift, over-correction, stopping and coupling issues; learned checkpoint selection cannot rescue an oracle-limited trajectory.
- **T013:** scalar restoration-value fitting is insufficient for a good TTT derivative field.
- **T014:** Sobolev/derivative supervision is the first fully qualified learned-inner-objective result; it supports the claim that for gradient-based TTT the learned optimization field matters more than scalar value fit.
- **T015:** routing among global/bilinear2/Region2 is negative; oracle headroom is only 1.94%.
- **T016-A:** shiftable hard-boundary capacity is positive; 27-candidate oracle is `0.92764×` canonical Region2.
- **T016-B–F:** scalar energy/value/ranking/confidence approaches are not robust enough for hard-boundary selection.
- **T017:** `tau=0.05` soft geometry carries diagnostic signal but soft→hard transfer and matched-soft deployment are unsafe; the soft-renderer route is closed.
- **T018-A:** hard local x/y reference target is positive (5/5), pooled `0.930464×` Region2 and `1.001327×` nine-hard oracle, with zero harmful development moves.
- **T018-B:** frozen T014 scalar Sobolev energy fails even under the correct local decision geometry (0/5).
- **T018-C:** direct axis-direction classification on the unchanged frozen representation passes 5/5 grouped OOF, establishing representation-level direction sufficiency.
- **T018-E:** frozen exact-direction selector is a one-shot fresh negative (4/5): pooled/offset/left-right transfer, but quadrants reach `1.011785×` Region2 because three unnecessary x moves occur. That cohort is burned for corrective tuning.
- **T019-A:** fixed 1% utility-deadband reference target passes 5/5 heterogeneous development plus zero harmful moves; it retains 97.6018% of pooled nine-hard oracle headroom.
- **T019-B:** label-only grouped-OOF intervention passes 7/7; harmful episodes fall `11→5` and quadrants remain zero-harm.
- **T019-C:** final all-development 1% deadband selector frozen; engineering barrier only.
- **T019-D:** **one-shot fresh heterogeneous geometry qualification positive (5/5)**. Pooled `0.952464×`, oracle proximity `1.027465×`, offset `0.874063×`, left/right `1.004090×`, quadrants `1.000000×`; `43/73/4` beneficial/equal/harmful; quadrants have zero movement.
- **T020-A:** **one-shot fresh non-spatial safety negative (3/4)**. Pooled `0.986616×`, clean `1.046125×` fail, dark `0.994103×`, bright `0.963036×`. Clean has `0/39/1` beneficial/equal/harmful. PR #36 merged as `e59c89badfcc785678b1e88c9c46d30e481d9b66`.

## Information-boundary status

T019-D and T020-A use the repaired fail-closed provenance chain: exclusion, manifest, manifest-frozen, mapping, input index, preparation metadata, scientific source/assets, feature rows, and selector decisions are contemporaneously hash-bound. All decisions are independently replayed before clean/reference evaluation opens.

Non-negotiable rules remain:

- test-time adaptation/selection must never consume test labels, clean targets, degradation masks/gain maps, condition IDs, annotations, image IDs as semantic shortcuts, source-only reference gradients/Jacobians, or evaluation metrics;
- source/development clean references may be used only in explicitly declared training/calibration/diagnostic stages;
- fresh image decisions must be finalized and persisted before reference metrics, family labels, or oracles are attached;
- inspected fresh IDs are permanently excluded from future corrective fresh cohorts;
- a failed fresh cohort may motivate a development-only hypothesis but its references/logits/outcomes may not be used to tune that hypothesis;
- development-only oracle/OOF/reference diagnostics are not fresh qualification;
- fresh qualification must fail closed on source/provenance/preparation binding mismatches.

## Interpretation of strongest evidence

The method story has two distinct levels.

First, T014 establishes an **objective-field principle**: a useful test-time energy should learn restoration-useful derivatives, not merely scalar restoration values. Sobolev derivative supervision strongly improves unseen gradient alignment and fresh adaptation.

Second, T018–T019 establish a **utility-aware geometry principle on heterogeneous shifts**: adaptive hard-boundary direction information exists in the frozen representation, but movement should be supervised only when expected utility is material. A fixed 1% deadband removes the prior fresh quadrant failure and yields a new-cohort 5/5 heterogeneous geometry qualification.

T020-A adds the necessary limitation: **heterogeneous geometry qualification is not equivalent to broad clean safety**. The frozen selector can still make a rare unnecessary clean movement large enough to violate the mean safety clause. We must therefore determine whether this is a target-domain problem or a learned-selector generalization problem before adding another model component.

The larger paper-level gaps remain downstream detector/task metrics, real adverse-image distributions, and test-time computational cost. They should be addressed after the current target-vs-predictor diagnostic rather than mixed into it.

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
- **M10 / T015:** completed — cross-basis routing negative.
- **M11 / T016:** completed — hard-boundary headroom established; scalar/ranking/confidence selectors closed.
- **M12 / T017:** completed — soft diagnostic signal identified; soft-transfer/deployment route closed.
- **M13 / T018:** completed — direct hard-axis direction representation established; exact-direction frozen selector fresh negative 4/5.
- **M14 / T019-A:** completed — 1% utility-deadband target positive on heterogeneous development.
- **M14 / T019-B:** completed — grouped-OOF deadband selector positive 7/7.
- **M14 / T019-C:** completed — immutable all-development deadband selector frozen.
- **M14 / T019-D:** **COMPLETED — ONE-SHOT FRESH HETEROGENEOUS GEOMETRY QUALIFICATION POSITIVE 5/5.**
- **M15 / T020-A:** **COMPLETED — ONE-SHOT FRESH NON-SPATIAL SAFETY NEGATIVE 3/4; CLEAN MEAN FAILS AT 1.046125×.**
- **M15 / T020-B:** **ACTIVE — DEVELOPMENT-ONLY NON-SPATIAL 1% DEADBAND TARGET VIABILITY AUDIT.**

## Current open task

`T020-B — development-only non-spatial 1% utility-deadband target viability audit` in `coordination/CHATGPT_TO_CODEX.md`.

T020-B uses only the accepted T018/T019 development image IDs and the original T014 clean/homogeneous-dark/homogeneous-bright settings. It applies the literal fixed `delta=0.01` reference deadband rule with no training and no threshold search. The target is considered viable only if pooled plus all three non-spatial condition means remain within 1% of canonical T014 **and** the combined target produces zero harmful development episodes. No selector retraining or fresh cohort is authorized in this cycle.