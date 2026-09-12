# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Processing for Task-Oriented Vision**

## Core scientific question

Can a vision system adapt a **compact spatial correction field per test image**, without test labels or clean targets, so that spatially heterogeneous nuisance degradation is corrected before a frozen downstream task model processes the image?

## Current hypothesis

A useful spatial TTT system requires four pieces to align: (i) a content-safe nuisance signal, (ii) a label-free inner objective whose **gradient field** generates restoration-useful states, (iii) safe action/stopping geometry, and (iv) an appropriate spatial basis. T014 establishes the first three under projected EV+gamma. T015 shows that routing among global, bilinear2 and canonical hard Region2 is not enough because that three-output set has only 1.94% oracle headroom. T016-A now shows a more specific spatial opportunity: keeping the already-selected four Region2 EV/gamma actions fixed, **image-dependent placement of the hard horizontal/vertical boundaries** exposes substantial reference-only headroom. The immediate question is whether the already-frozen Sobolev energy can select that boundary placement label-free before any learned basis or boundary predictor is justified.

## Current validated method

For a test image `x_t`:

1. obtain the frozen T006 CLIP exposure readout and T007 clean-abstention gate;
2. freeze active/dark/bright decisions from the original image;
3. use the frozen T014 Sobolev restoration energy built from the fixed 28 test-time features;
4. optimize a compact projected EV+gamma Region2 fast state from identity using only current pixels/frozen features, never test labels or clean targets;
5. save the full label-free trajectory and choose the minimum predicted-energy checkpoint;
6. attach clean-reference metrics only after all trajectories/decisions are persisted.

The best **fresh-validated deployable** spatial basis remains canonical hard Region2 under the T014 synthetic protocol. T016-A is development-only reference headroom, not a deployed selector and not a fresh qualification.

## Established controlled findings

- **T001–T003:** spatial ISP adaptation is viable on favorable heterogeneous shifts, but spatial capacity alone is insufficient and simple absolute-statistic priors/regularization are not identity-safe.
- **T004–T007:** zero-shot exposure signals are content-confounded; a source-trained frozen-CLIP exposure readout plus joint clean-abstention calibration yields a usable but imperfect label-free degradation signal.
- **T008:** the signal drives real correction, but the original semantic EV+gamma TTT fails clean safety and loses to simple local direct correction.
- **T009:** development geometry audit shows semantic gradients are usually directionally useful at identity, but over-correction/stopping and bilinear spatial coupling are major failure modes. Gamma removal is not justified.
- **T010:** residual semantic targets improve clean safety but erase most advantage over direct correction; no predeclared residual target qualifies.
- **T011:** projected gate-consistent Region2 TTT is strong but unqualified on a fresh split. It beats identity/global/direct materially, but misses clean p95 and the required margin over matched discrete. Projection is strongly active and hard Region2 does not win boundary-offset stress.
- **T012:** learned checkpoint selection cannot rescue the frozen T011 trajectory; even the reference-only checkpoint oracle misses the required margins over fixed16/discrete.
- **T013:** scalar source-trained restoration energy is a controlled development negative. Scalar value fit does not sufficiently constrain `∇_phi E`; gradient alignment and reachable trajectory quality remain weak.
- **T014:** source-supervised Sobolev restoration energy is the first fully qualified learned-inner-objective result. Stage A passed **8/8** development clauses and immutable Stage B passed **12/12** fresh clauses on 40 unseen images. Fresh heterogeneous MSE is `0.03385803`, 19.17% below the same-source value-only energy, 11.72% below matched projected discrete and 8.31% below frozen semantic step16. Unseen-calibration gradient alignment is `73/74` positive with median cosine `0.93606`, versus `59/74` and `0.42464` for value-only.
- **T015:** frozen raw-energy routing among global, bilinear2 and Region2 is a **controlled fresh negative**. Only **4/10** clauses pass. Routed spatial-pool MSE is `0.03780638`, versus evaluation-only best fixed Region2 `0.03504357` (routing 7.88% worse). Reference-only oracle among the same three selected outputs is `0.03436452`, only **1.94%** better than the best fixed basis. A smarter router over the same three outputs is therefore not justified.
- **T016-A:** fixed-action renderer-transfer is a **positive development-only headroom diagnostic**. On the same 120 previously inspected T015 spatial episodes, exactly 27 predeclared shiftable/soft four-region renderers reuse the identical saved selected Region2 EV/gamma corners. The per-input reference oracle reaches `0.03250770` spatial MSE: `0.92764×` canonical Region2 and `0.94597×` the T015 three-basis oracle, so all three strong adaptive-headroom clauses pass. The best single fixed renderer remains canonical Region2. Oracle selections are hard (`tau=0`) in `113/120` episodes, including `72` shifted-hard selections; only `7/120` use smoothing. Offset cases benefit most (`0.84483×` Region2), while exact quadrants are essentially unchanged. Thus the demonstrated capacity is primarily **adaptive boundary placement**, not generic sigmoid smoothing.

## Interpretation of the strongest results

T014 supports the narrow causal claim that **derivative supervision matters**. The Sobolev head fits scalar source restoration values worse than the matched value-only head yet transfers much better derivative geometry and produces better fresh restoration. The positive mechanism remains **Sobolev inner objective + frozen projected action geometry**: projection is heavily active, so this is not an unconstrained-energy claim.

T015 and T016-A refine the remaining spatial limitation. T015 rules out spending effort on a smarter selector over only global/bilinear2/canonical-Region2 outputs because the oracle candidate set is too weak. T016-A then shows that the same selected Region2 corner actions become materially better when the hard split location is allowed to vary per image. Because the globally best fixed renderer is still canonical Region2, the evidence favors **adaptation of geometry**, not replacement by one universal soft basis. This is reference-only development evidence; whether a label-free criterion can recover it is unresolved.

## Non-negotiable design principles

- Test-time adaptation/selection must never consume test labels, clean targets, degradation masks/gain maps, condition IDs, annotations, image IDs as shortcuts, source-only Jacobians/reference gradients, or evaluation metrics.
- Source clean references are permitted only for explicitly source/development-side training or diagnostics whose learned/frozen outputs are fixed before later held-out evaluation.
- Label-free outputs/trajectories/decisions must be finalized and persisted before clean-reference metrics or oracles are attached.
- Keep identity, global, direct and matched discrete controls in every major spatial/TTT claim.
- Do not infer useful gradients from low self-supervised loss or scalar value-regression accuracy; derivative quality must be checked explicitly.
- A gradient-based TTT claim must beat matched non-gradient direct/discrete and strong frozen semantic controls.
- A selector/routing claim must be checked against a reference-only oracle; if oracle headroom is weak, do not train the selector.
- Fresh/development splits become permanently unavailable for corrective tuning after inspection.
- Hard Region2 remains limited by boundary-misaligned evidence; any general spatial claim must address this directly.
- Fresh-run launchers must fail closed by binding declared source SHAs to the actual runtime scientific files; retrospective evidence does not substitute for future pre-launch enforcement.
- Development-only oracle diagnostics may decide whether a direction is worth pursuing, but they are not deployable methods and must never be described as fresh qualification.

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
- **M10 / T015:** **COMPLETED — FROZEN CROSS-BASIS ROUTING NEGATIVE (4/10); THREE-BASIS ORACLE HEADROOM ONLY 1.94%.**
- **M11 / T016-A:** **COMPLETED — DEVELOPMENT-ONLY SHIFTABLE-BOUNDARY ORACLE HEADROOM POSITIVE; 7.24% BELOW REGION2, WITH HARD BOUNDARY PLACEMENT DOMINANT.**
- **M11 / T016-B:** **ACTIVE — FROZEN SOBOLEV LABEL-FREE HARD-BOUNDARY SELECTION AUDIT.**

## Current open task

`T016-B — frozen-Sobolev hard-boundary selection audit` in `coordination/CHATGPT_TO_CODEX.md`.

T016-B uses only the same 120 already-inspected T015 spatial episodes and the nine predeclared hard boundaries `b_x,b_y ∈ {0.4,0.5,0.6}`. It does not train or optimize anything. It asks whether the accepted frozen T014 Sobolev energy can select a shifted hard boundary from candidate pixels without clean/reference information. The scoring artifact must be finalized before any T016-A reference MSE table is read. No learned boundary predictor, learned basis, feature expansion, detector coupling, meta-initialization, prompt retraining, or ViT3-style fast model is authorized in this cycle.
