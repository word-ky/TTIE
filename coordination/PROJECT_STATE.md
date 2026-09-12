# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Processing for Task-Oriented Vision**

## Core scientific question

Can a vision system adapt a **compact spatial correction field per test image**, without test labels or clean targets, so that spatially heterogeneous nuisance degradation is corrected before a frozen downstream task model processes the image?

## Main hypothesis

A single global enhancement state is insufficient when degradation varies across an image. A compact spatial field can help, but only if four pieces align: (i) a content-safe nuisance signal, (ii) a label-free **inner objective whose gradient creates useful restoration states**, (iii) a safe stopping/selection rule, and (iv) a spatial basis whose support matches the degradation without unnecessary coupling. T012 now shows that checkpoint selection alone cannot repair a trajectory that does not contain enough advantage over strong non-gradient/fixed-depth controls.

## Current method abstraction

For a test image `x_t`:

1. extract four fixed local quadrant views;
2. obtain the frozen T006 source-trained exposure readout in CLIP feature space;
3. apply the frozen T007 joint clean-abstention gate to determine active quadrants and dark/bright winner;
4. optimize a compact projected EV+gamma state, with inactive regions exact identity and dark/bright EV direction constrained by the frozen gate;
5. replace the hand-designed zero-envelope loss with a **source-trained differentiable restoration energy** that sees only test-available gate/current-CLIP/current-ISP features;
6. backpropagate through the frozen energy/readout to the ISP fast state, never through test labels or clean targets;
7. compare against identity, global learned-energy TTT, direct correction, matched projected discrete search, the original semantic trajectory, frozen source-fixed step 16, and renderer controls;
8. attach clean-reference metrics/oracles only after all label-free trajectories, selections and outputs are finalized and persisted.

This learned task-aligned inner objective is the T013 candidate. The T006/T007 signal and T011 projected action geometry remain frozen.

## Established controlled findings

- **T001:** episodic label-free optimization of a bounded spatial ISP field is stable and can strongly outperform one global ISP vector on a favorable heterogeneous exposure toy.
- **T002:** spatial gain is conditional: useful for compatible low-frequency heterogeneous shifts, negligible on homogeneous shifts, and potentially harmful for high-frequency/content-confounded cases. A 96-latent uniform control rules out a trivial parameter-count explanation.
- **T003:** the absolute patch-mean-to-0.5 prior is not identity-safe; identity anchoring + plain TV do not rescue it without destroying useful correction.
- **T004:** zero-shot absolute frozen-CLIP exposure scores show useful dark sensitivity but weak bright discrimination and large content offsets.
- **T005:** a fixed same-image ±0.25 EV relative CLIP probe does not solve the content problem.
- **T006:** source-trained three-prototype readout on frozen CLIP features yields strong exposure ranking/localization, but its first independent gate has excessive clean false activation. These vectors are a discriminative frozen-CLIP readout, not preserved text semantics.
- **T007:** a predeclared joint clean-abstention envelope reduces clean all-view FPR to **6%** and clean image-any activation to **20%**, while retaining usable dark/bright recall.
- **T008:** the first frozen-signal EV+gamma pilot proves the signal can drive real correction, but the original semantic TTT is not qualified: it improves dark/bright and heterogeneous restoration yet fails clean safety and loses to a simple local direct action.
- **T009:** development-only geometry audit shows the semantic gradient is usually directionally useful at identity (94.77% positive alignment; median cosine 0.7973), while stopping/over-correction is a dominant failure. 92.81% improve after the first update and 69.28% have an earlier reference-MSE optimum than the final semantic stop. Gamma removal is not justified. Exact-quadrant `region2` removes major bilinear coupling, but the geometry is synthetic-boundary matched.
- **T010:** residual-evidence target relaxation is a controlled negative result. Zero of 16 fixed rho pairs qualifies: clean safety improves, but heterogeneous advantage over `region2_direct` largely disappears. The original zero-envelope objective retains stronger restoration pressure but is unsafe in the clean tail.
- **T011:** gate-consistent projected action geometry is a controlled fresh-split negative qualification. `region2_ttt_projected` passes 8/10 clauses and reaches heterogeneous MSE **0.02559107**, improving **55.03%** over identity, **46.00%** over projected global TTT and **12.64%** over direct correction, but only **4.46%** over matched projected discrete and misses clean p95 safety (**0.005844413 > 0.005**). Projection changes **67.19%** of updates and **52.72%** of active final coordinates lie on a trust-box boundary. Full TTT beats one-step on 79/80 heterogeneous episodes, so one universal shallow depth is not enough. The 40%-boundary stress makes hard `region2` **1.30% worse** than projected bilinear, limiting any general renderer claim.
- **T012:** source-trained checkpoint-quality selection is a **controlled development-only negative result**, PR #12 squash-merged as `a20c45f429a89dc4ca73f1060de33523f80b7b86`. Stage A passes only 1/5 clauses: learned stop clean p95 **0.00511147**, dark/identity **0.65746**, bright/identity **0.40333**, heterogeneous/discrete **1.09773**, heterogeneous/fixed16 **1.13846**. More importantly, the per-image reference oracle over the exact same frozen T011 checkpoints has heterogeneous MSE **0.02744631**, versus **0.02782951** for source-fixed step 16 and **0.02886186** for projected discrete. Oracle/fixed16 is **0.98623** (1.38% gain) and oracle/discrete is **0.950954** (4.9046% gain), so even perfect checkpoint selection cannot meet either required 5% margin on this calibration set. The learned/oracle ratio **1.15435** shows additional selector error, but selector improvement alone cannot satisfy the T012 contract. Stage B was correctly not opened and no fresh T012 evaluation images were touched.

## Non-negotiable design principles

- Test-time adaptation, learned-objective optimization and checkpoint selection must never consume test labels, clean targets, degradation masks, gain maps, condition IDs, annotations, image IDs as semantic shortcuts, or evaluation metrics.
- Clean references may be used only after adaptation for controlled evaluation, except in explicitly source/development-side training/diagnostic modules whose learned outputs are frozen before later evaluation.
- Keep identity/no-adaptation and a global counterpart in every major spatial claim.
- Do not infer restoration or task utility from lower self-supervised/learned energy alone.
- Spatial capacity should remain compact/coarse rather than a free full-resolution correction tensor.
- Separate representation quality, gate safety, inner-objective gradient alignment, stopping/selection, action geometry, spatial basis and optimization mechanism before adding larger fast models/meta-learning.
- After a result motivates a new hypothesis, use a fresh deterministic source/calibration or evaluation split; inspected images are unavailable for corrective tuning.
- Source-trained controllers/objectives/signals are allowed only with strict development/evaluation separation; source labels or synthetic degradation metadata must never become test-time inputs.
- A gradient-based TTT claim must beat non-gradient direct/discrete actions using the same frozen signal. If simple policies match or win, do not attribute value to TTT.
- A stopping/selector claim must be checked against a reference-only trajectory oracle. If the oracle itself cannot clear the target baseline margin, do not keep tuning the selector on that trajectory.
- Hard region-aligned fields remain explicitly limited by boundary-misaligned stress evidence until a more general spatial basis is tested.

## Milestone M0 — mechanism scaffold

Status: **COMPLETED / T001 ACCEPTED**

## Milestone M1 — isolate spatiality and prior confounds

Status: **COMPLETED / T002 ACCEPTED**

## Milestone M2 — simple regularization of absolute prior

Status: **COMPLETED / T003 NEGATIVE DIAGNOSTIC**

## Milestone M3 — degradation-aware semantic readout and first correction pilot

Status: **COMPLETED THROUGH T008 — READOUT QUALIFIED, ORIGINAL TTT NOT QUALIFIED**

## Milestone M4 — objective/action geometry diagnosis

Status: **COMPLETED / T009 DEVELOPMENT-ONLY DIAGNOSTIC**

## Milestone M5 — residual-target stopping repair

Status: **COMPLETED / T010 NEGATIVE DEVELOPMENT GATE**

## Milestone M6 — gate-consistent projected action geometry

Status: **COMPLETED / T011 CONTROLLED NEGATIVE QUALIFICATION**

T011 establishes a strong but still unqualified projected spatial trajectory and shows that projection is genuinely active. It does not justify further hand-tuning of the same action box or a general hard-quadrant renderer claim.

## Milestone M7 — source-trained label-free trajectory stopping

Status: **COMPLETED / T012 NEGATIVE DEVELOPMENT GATE**

T012 freezes the T011 trajectory and tests whether input-adaptive source-trained checkpoint selection can rescue it. The learned selector itself underperforms strong controls, but the decisive result is the reference-only trajectory oracle: even perfect selection among the frozen T011 checkpoints cannot provide the required 5% advantage over source-fixed step 16 or matched projected discrete on the calibration set. Therefore the next step must change the available trajectory rather than continue tuning a stopping head.

The 80 T012 train + 20 calibration images are permanent development data; total inspected IDs through T012 are **408**. No T012 fresh Stage B split exists.

## Milestone M8 — source-trained task-aligned inner objective

Status: **ACTIVE / T013 OPEN**

T013 is the first authorized learned-inner-objective experiment. It trains a compact differentiable restoration energy on a new source-only state bank containing identity/direct/discrete states, selected old semantic checkpoints and fixed off-trajectory Sobol states. The energy receives only frozen gate information, current CLIP exposure evidence and current EV/gamma state. Source clean MSE is a training target only; at test time the energy is frozen and gradients flow from that label-free scalar through the frozen readout into the projected ISP state.

T013 must first pass a new 80-train/20-calibration development gate, including explicit reference-gradient-alignment diagnostics and actual restoration margins over matched projected discrete and the frozen T012 step-16 baseline. Only a passing frozen source receipt can unlock a new 40-image fresh evaluation split.

No detector work, meta-initialization, ViT3-style fast model, prompt retraining or learned spatial basis is authorized until T013 resolves whether a task-aligned learned energy can create a trajectory with genuinely better states than the hand-designed semantic trajectory.

## Open task

`T013 — Source-Trained Differentiable Restoration Energy` in `coordination/CHATGPT_TO_CODEX.md`.

## Candidate later components (not approved unless an OPEN task says otherwise)

- Soft/learned or low-rank spatial basis after the T011 boundary-misaligned stress result is addressed explicitly.
- Jointly learned fast initialization only after a learned inner objective demonstrates value over direct/discrete controls.
- Frozen detector/object-feature consistency only after a corrected restoration mechanism qualifies.
- True CLIP-LIT/CoOp-style prompt learning or a non-CLIP degradation encoder if the current frozen-feature readout becomes limiting again.
- Stronger effective-function-capacity global baseline.
- Meta-learned fast initialization `W_0`.
- ViT3-style fast-weight model mapping local degradation tokens to ISP actions.