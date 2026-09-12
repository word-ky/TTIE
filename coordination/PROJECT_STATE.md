# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Processing for Task-Oriented Vision**

## Core scientific question

Can a vision system adapt a **compact spatial correction field per test image**, without test labels or clean targets, so that spatially heterogeneous nuisance degradation is corrected before a frozen downstream task model processes the image?

## Current hypothesis

A useful spatial TTT system requires four pieces to align: (i) a content-safe nuisance signal, (ii) a label-free inner objective whose **gradient field** generates restoration-useful states, (iii) safe action/stopping geometry, and (iv) a spatial basis that does not unnecessarily couple regions. T012 ruled out selection-only repair of the old semantic trajectory; T013 showed scalar restoration-value regression alone does not produce a sufficiently useful learned derivative field. T014 now provides development-side evidence that explicit source derivative supervision can repair that gradient geometry while keeping the same 28-feature representation and test-time optimizer fixed.

## Current method abstraction

For a test image `x_t`:

1. extract four fixed local quadrant views;
2. obtain the frozen T006 CLIP exposure readout and frozen T007 clean-abstention gate;
3. freeze active/dark/bright gate decisions from the original test image;
4. optimize the compact T011 projected EV+gamma Region2 state, with inactive regions exact identity and gate-consistent EV direction;
5. evaluate the frozen T014 Sobolev restoration energy from the same 28 test-time features used by T013;
6. backpropagate only through current pixels / frozen CLIP evidence / ISP state; source Jacobians and clean-reference gradients are training-only and absent at deployment;
7. save the complete label-free trajectory and choose the minimum predicted-energy checkpoint;
8. compare against identity, direct, matched projected discrete, frozen semantic step16/final, global/bilinear Sobolev controls, and a same-source value-only energy control;
9. attach clean-reference metrics/oracles only after label-free outputs and decisions are finalized and persisted.

## Established controlled findings

- **T001–T003:** spatial ISP adaptation is viable on favorable heterogeneous shifts, but spatial capacity alone is not sufficient and simple absolute-statistic priors/regularization are not identity-safe.
- **T004–T007:** zero-shot exposure signals are content-confounded; a source-trained frozen-CLIP exposure readout plus joint clean-abstention calibration provides a usable but imperfect label-free degradation signal.
- **T008:** the signal drives real correction, but the original semantic EV+gamma TTT fails clean safety and loses to simple local direct correction.
- **T009:** development geometry audit shows semantic gradients are usually directionally useful at identity, but over-correction/stopping and bilinear spatial coupling are major failure modes. Gamma removal is not justified.
- **T010:** residual semantic targets improve clean safety but erase most advantage over direct correction; no predeclared residual target qualifies.
- **T011:** projected gate-consistent Region2 TTT is strong but still unqualified on a fresh split. Heterogeneous MSE is `0.02559107`, improving 55.03% over identity, 46.00% over projected global and 12.64% over direct, but only 4.46% over matched discrete and clean p95 is `0.005844413 > 0.005`. Projection is strongly active and hard Region2 does not win the boundary-offset stress.
- **T012:** learned checkpoint selection cannot rescue the frozen T011 trajectory; even the reference-only trajectory oracle misses the required 5% margins over fixed16/discrete.
- **T013:** scalar source-trained restoration energy is a controlled development negative. Positive gradient cosine is only `54/78 = 69.23%`, median `0.24765`; heterogeneous performance is worse than discrete/fixed16, and even the energy-trajectory oracle is worse than those controls. Scalar value fit therefore does not sufficiently constrain `∇_phi E`.
- **T014 Stage A:** explicit source derivative supervision passes **8/8 predeclared development clauses** on a new 20-image calibration split while holding features, architecture, bank, action geometry and optimizer fixed. Sobolev calibration gradient alignment is `73/74 = 98.65%` positive with median cosine `0.93606`, versus `59/74` and `0.42464` for the matched value-only control. Clean p95 is `0.00023943`; dark/identity `0.55913`; bright/identity `0.41087`; heterogeneous ratios are `0.92635` vs projected discrete, `0.91866` vs frozen semantic step16, and `0.83425` vs same-source value-only energy. The reference-only Sobolev trajectory oracle is also strong (`oracle/discrete=0.88668`, `oracle/fixed16=0.87932`). This is **development evidence only**, not fresh qualification.

## Interpretation of T014 Stage A

The matched-control result supports the narrow causal hypothesis that **derivative supervision matters**: the Sobolev head has worse scalar source value Huber than the value-only head (`0.05101` vs `0.03319`) yet much better unseen-calibration gradient alignment and lower heterogeneous restoration MSE. This separates scalar value fitting from useful test-time gradient geometry.

The result must not be overclaimed. The Sobolev trajectory remains heavily constrained by the T011 trust box: about `93.6%` of primary updates are projected and about `65.05%` of movable final coordinates lie on a boundary. A future positive fresh result would therefore support the combined **Sobolev objective + frozen projected action geometry**, not an unconstrained learned-energy claim.

The Stage-A freeze barrier was completed before any fresh scoring: both heads and the passing receipt were committed/verified at `6a9870f836b6763f843d78e5c549eba432b0a150`; only then was the deterministic 40-image `evaluation_t014` manifest generated and committed as `7416a1b91949985312cef002d47568091fb57761`, excluding all 608 prior/source IDs.

## Non-negotiable design principles

- Test-time adaptation must never consume test labels, clean targets, degradation masks/gain maps, condition IDs, annotations, image IDs as shortcuts, source-only Jacobians/reference gradients, or evaluation metrics.
- Source clean references are permitted only for explicitly source-side training/diagnosis whose learned outputs are frozen before later evaluation.
- Label-free outputs/trajectories/decisions must be finalized and persisted before clean-reference metrics or oracles are attached.
- Keep identity, global, direct and matched discrete controls in every major spatial/TTT claim.
- Do not infer useful gradients from low self-supervised loss or scalar value-regression accuracy; derivative quality must be checked explicitly.
- A gradient-based TTT claim must beat matched non-gradient direct/discrete controls and strong frozen-depth semantic controls.
- A selector/trajectory claim must be checked against a reference-only oracle; if the oracle is weak, repair trajectory generation rather than selection.
- Fresh/development splits become permanently unavailable for corrective tuning after inspection.
- Hard Region2 remains limited by the boundary-misaligned stress evidence.

## Milestones

- **M0 / T001:** completed — mechanism scaffold.
- **M1 / T002:** completed — spatiality/capacity controls.
- **M2 / T003:** completed — simple-prior rescue negative.
- **M3 / T004–T008:** completed — frozen nuisance readout qualified; original semantic TTT unqualified.
- **M4 / T009:** completed — objective/action geometry diagnosis.
- **M5 / T010:** completed — residual-target repair negative.
- **M6 / T011:** completed — projected action geometry strong but fresh qualification negative.
- **M7 / T012:** completed — learned stopping/selection negative; oracle-limited.
- **M8 / T013:** completed — scalar learned inner objective negative; gradient/trajectory limited.
- **M9 / T014:** **ACTIVE — STAGE A PASSED 8/8; FROZEN FRESH STAGE B RUNNING.**

T014 fresh Stage B uses the immutable Sobolev/value-only heads and a new 40-image split. Qualification requires all twelve predeclared restoration/safety/spatial/control clauses. `offset_left_right_40` remains report-only. No Stage-B result has yet been accepted by the research lead.

## Open task

`T014 — Source-Supervised Sobolev Restoration Energy`, with the current instruction in `coordination/CHATGPT_TO_CODEX.md`: continue only the frozen fresh Stage-B run and report the literal twelve-clause verdict after complete verification.

No feature expansion, learned spatial basis, detector work, meta-initialization, prompt retraining, or ViT3-style fast model is authorized until T014 fresh Stage B is resolved.
