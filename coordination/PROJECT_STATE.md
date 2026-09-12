# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Processing for Task-Oriented Vision**

## Core scientific question

Can a vision system adapt a **compact spatial correction field per test image**, without test labels or clean targets, so that spatially heterogeneous nuisance degradation is corrected before a frozen downstream task model processes the image?

## Current hypothesis

A useful spatial TTT system requires four pieces to align: (i) a content-safe nuisance signal, (ii) a label-free inner objective whose **gradient field** generates restoration-useful states, (iii) safe action/stopping geometry, and (iv) an appropriate spatial basis. T014 now provides fresh evidence that explicit source derivative supervision can produce a useful learned inner objective under the frozen projected geometry. The remaining immediate limitation is basis choice: global, bilinear and hard Region2 have complementary regimes, and hard Region2 still loses its advantage under a boundary-offset stress.

## Current method abstraction

For a test image `x_t`:

1. obtain the frozen T006 CLIP exposure readout and T007 clean-abstention gate;
2. freeze active/dark/bright decisions from the original image;
3. use the frozen T014 Sobolev restoration energy built from the same 28 test-time features as T013;
4. optimize a compact projected EV+gamma fast state from identity using only current pixels/frozen features, never test labels or clean targets;
5. save the full label-free trajectory and choose the minimum predicted-energy checkpoint;
6. in T015, run the unchanged global, bilinear2 and hard Region2 Sobolev trajectories and route among their selected outputs using only the three frozen Sobolev energy scores;
7. attach clean-reference metrics and reference-only oracle diagnostics only after all trajectories, scores and routing decisions are persisted.

## Established controlled findings

- **T001–T003:** spatial ISP adaptation is viable on favorable heterogeneous shifts, but spatial capacity alone is insufficient and simple absolute-statistic priors/regularization are not identity-safe.
- **T004–T007:** zero-shot exposure signals are content-confounded; a source-trained frozen-CLIP exposure readout plus joint clean-abstention calibration yields a usable but imperfect label-free degradation signal.
- **T008:** the signal drives real correction, but the original semantic EV+gamma TTT fails clean safety and loses to simple local direct correction.
- **T009:** development geometry audit shows semantic gradients are usually directionally useful at identity, but over-correction/stopping and bilinear spatial coupling are major failure modes. Gamma removal is not justified.
- **T010:** residual semantic targets improve clean safety but erase most advantage over direct correction; no predeclared residual target qualifies.
- **T011:** projected gate-consistent Region2 TTT is strong but still unqualified on a fresh split. It beats identity/global/direct materially, but misses clean p95 and the required margin over matched discrete. Projection is strongly active and hard Region2 does not win the boundary-offset stress.
- **T012:** learned checkpoint selection cannot rescue the frozen T011 trajectory; even the reference-only checkpoint oracle misses the required margins over fixed16/discrete.
- **T013:** scalar source-trained restoration energy is a controlled development negative. Scalar value fit does not sufficiently constrain `∇_phi E`; gradient alignment and reachable trajectory quality remain weak.
- **T014:** source-supervised Sobolev restoration energy is the first fully qualified learned-inner-objective result. Stage A passed **8/8** development clauses and the immutable frozen Stage B passed **12/12** fresh clauses on 40 unseen images. Fresh heterogeneous MSE is `0.03385803`, 19.17% below the same-source value-only energy, 11.72% below matched projected discrete and 8.31% below frozen semantic step16. Fresh ratios are `0.56250` vs global Sobolev, `0.84696` vs direct, `0.88275` vs discrete, `0.91688` vs fixed16 and `0.80830` vs value-only. Clean mean/p95 are `0.00025326 / 0.00012560`; dark/bright homogeneous ratios are `0.57095 / 0.42042`; dark/bright regional ratios are `0.53157 / 0.44764`; exact-quadrant/bilinear is `0.71437`.

## Interpretation of T014

T014 supports the narrow causal claim that **derivative supervision matters**. The Sobolev head fits scalar source restoration values worse than the matched value-only head (`Huber 0.05101` vs `0.03319`) yet transfers much better derivative geometry to unseen calibration episodes (`73/74` positive, median cosine `0.93606` vs `59/74`, median `0.42464`) and produces better fresh restoration.

The positive result is not an unconstrained-energy claim. The projected action box remains heavily active: about 94% of primary updates are projected and roughly 63% of movable final coordinates end on a boundary. The supported mechanism is therefore the **Sobolev inner objective inside the frozen projected action geometry**.

The spatial-basis limitation also remains unresolved. On `offset_left_right_40`, hard Region2 is slightly worse than bilinear (`1.00589×`), with only 2.32% / 1.81% gains over discrete / fixed16, while global Sobolev remains better on homogeneous dark/bright. Thus no universal Region2 superiority is supported.

PR #14 was accepted and squash-merged as `cebecffbd1335fade336df17d653eb4e5fb65ba3`.

## Non-negotiable design principles

- Test-time adaptation must never consume test labels, clean targets, degradation masks/gain maps, condition IDs, annotations, image IDs as shortcuts, source-only Jacobians/reference gradients, or evaluation metrics.
- Source clean references are permitted only for explicitly source-side training/diagnosis whose learned outputs are frozen before later evaluation.
- Label-free outputs/trajectories/decisions must be finalized and persisted before clean-reference metrics or oracles are attached.
- Keep identity, global, direct and matched discrete controls in every major spatial/TTT claim.
- Do not infer useful gradients from low self-supervised loss or scalar value-regression accuracy; derivative quality must be checked explicitly.
- A gradient-based TTT claim must beat matched non-gradient direct/discrete and strong frozen semantic controls.
- A selector/routing claim must be checked against a reference-only oracle; if the oracle is weak, do not spend effort learning the selector.
- Fresh/development splits become permanently unavailable for corrective tuning after inspection.
- Hard Region2 remains limited by boundary-misaligned evidence; any general spatial claim must address this directly.

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
- **M9 / T014:** **COMPLETED — SOBOLEV INNER OBJECTIVE PASSED 8/8 DEVELOPMENT AND 12/12 FRESH QUALIFICATION.**
- **M10 / T015:** **ACTIVE — FROZEN CROSS-BASIS ROUTING AUDIT.**

## Open task

`T015 — Frozen Sobolev Cross-Basis Routing` in `coordination/CHATGPT_TO_CODEX.md`.

T015 introduces no new training. It tests whether the accepted frozen T014 Sobolev energy is sufficiently calibrated across global, bilinear2 and hard Region2 trajectories to route basis choice per test image on a new 40-image fresh split. The offset boundary condition is now part of primary qualification.

No learned spatial basis, feature expansion, detector coupling, meta-initialization, prompt retraining, or ViT3-style fast model is authorized until T015 resolves whether simple frozen-energy basis routing is already sufficient.
