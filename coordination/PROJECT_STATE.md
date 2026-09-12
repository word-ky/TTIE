# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Processing for Task-Oriented Vision**

## Core scientific question

Can a vision system adapt a **compact spatial correction field per test image**, without test labels or clean targets, so that spatially heterogeneous nuisance degradation is corrected before a frozen downstream task model processes the image?

## Main hypothesis

A single global enhancement state is insufficient when degradation varies across an image. A compact spatial field can help, but only if four pieces align: (i) a content-safe nuisance signal, (ii) a label-free correction trajectory whose gradient is useful, (iii) a stopping/selection rule that converts that trajectory into a safe output, and (iv) a spatial basis whose support matches the degradation without unnecessary coupling. Spatial capacity alone can amplify a bad objective, stopping rule, or renderer.

## Current method abstraction

For a test image `x_t`:

1. extract four fixed local quadrant views;
2. obtain the frozen T006 source-trained exposure readout in CLIP feature space;
3. apply the frozen T007 joint clean-abstention gate to determine active quadrants and dark/bright winner;
4. generate the frozen T011 projected `region2` EV+gamma trajectory from current pixels only;
5. keep inactive regions exact identity, constrain dark/bright EV direction and magnitude to ±0.5, gamma to [0.8,1.25];
6. use a small **source-trained trajectory-quality head** to score saved label-free checkpoints and select one without test labels/clean targets;
7. compare against identity, global TTT, direct correction, matched discrete search, a source-tuned fixed-step baseline, and renderer controls;
8. attach clean-reference metrics only after every label-free trajectory, quality score, checkpoint selection and output has been finalized and persisted.

The learned stopping/trajectory-selection mechanism is the T012 candidate. The T011 semantic trajectory and projected action geometry remain frozen.

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
- **T011:** gate-consistent projected action geometry is a **controlled fresh-split negative qualification**, PR #11 squash-merged as `4c02ad3d6d44dcce3d2c146a9ebc87a1e2d43ed3`. The primary `region2_ttt_projected` passes 8/10 predeclared clauses. It reaches heterogeneous MSE **0.02559107**, improving **55.03%** over identity, **46.00%** over projected global TTT and **12.64%** over direct correction, but improves only **4.46%** over matched projected discrete search (required ≥5%) and fails clean p95 safety (**0.005844413 > 0.005**). Projection is not cosmetic: **67.19%** of primary updates are altered and **52.72%** of active final coordinates lie on a trust-box boundary. One-step TTT has very low clean drift, but full TTT beats it on **79/80** heterogeneous episodes, so one universal shallow step count does not solve the safety/utility tradeoff. The report-only 40%-boundary stress reverses the exact-quadrant renderer advantage: hard `region2` is **1.30% worse** than projected bilinear there. T011 therefore supports a useful spatial trajectory but not a qualified final stopping rule or a general hard-region basis claim.

## Non-negotiable design principles

- Test-time adaptation/selection must never consume test labels, clean targets, degradation masks, gain maps, condition IDs, annotations, or evaluation metrics.
- Clean references may be used only after adaptation for controlled evaluation, except in explicitly source/development-side training/diagnostic modules whose learned outputs are frozen before held-out evaluation.
- Keep identity/no-adaptation and a global counterpart in every major spatial claim.
- Do not infer restoration or task utility from lower self-supervised loss alone.
- Spatial capacity should remain compact/coarse rather than a free full-resolution correction tensor.
- Separate representation quality, gate safety, gradient alignment, stopping/selection, action geometry, spatial basis, and optimization mechanism before adding larger fast models/meta-learning.
- After a result motivates a new hypothesis, use a fresh deterministic split for the next decisive audit; inspected images are unavailable for corrective tuning.
- Source-trained controllers/signals are allowed only with strict development/evaluation separation; source labels or synthetic degradation metadata must never become test-time inputs.
- A gradient-based TTT claim must beat non-gradient direct/discrete actions using the same frozen signal. If simple policies match or win, do not attribute value to TTT.
- Hard region-aligned fields must remain explicitly limited by boundary-misaligned stress evidence until a more general spatial basis is tested.

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

Status: **COMPLETED / T011 ACCEPTED AS CONTROLLED NEGATIVE QUALIFICATION**

T011 demonstrates substantial spatial utility and a meaningful projected trajectory, but it still fails clean-tail safety and the required margin over matched discrete search. Projection is frequently active, so simply tightening/retuning the same hand-designed box is not the next scientific step. Fixed one-step versus full-step behavior also shows that stopping needs to be input-dependent. The hard quadrant renderer remains limited by the offset-boundary stress result.

## Milestone M7 — source-trained label-free trajectory stopping

Status: **ACTIVE / T012 OPEN**

T012 freezes the entire T011 trajectory generator and trains only a compact source-side quality head that predicts checkpoint restoration quality from test-available trajectory statistics. At test time it may select only among already-generated frozen T011 checkpoints and must not see clean references or condition metadata. A source-tuned fixed-step baseline is mandatory so any gain can be attributed to input-adaptive stopping rather than merely choosing a better universal iteration count.

T012 uses a strict source/fresh barrier: 80 train + 20 calibration images are development-only; the learned head, normalization constants, feature schema and fixed-step baseline must be frozen and hashed before a new 40-image evaluation manifest is created or scored. Stage A must pass source-side safety/utility gates before Stage B is allowed.

No detector work, meta-learning, learned inner objective, or ViT3-style fast model is authorized until T012 resolves whether learned stopping alone can qualify the existing spatial trajectory.

## Open task

`T012 — Source-Trained Label-Free Trajectory Quality Head` in `coordination/CHATGPT_TO_CODEX.md`.

## Candidate later components (not approved unless an OPEN task says otherwise)

- A genuinely source-trained/task-aligned inner objective if learned stopping cannot qualify the frozen T011 trajectory.
- Soft/learned or low-rank spatial basis after the T011 boundary-misaligned stress result is addressed explicitly.
- EV-only simplification only if future fresh evidence shows gamma consistently harmful.
- Frozen detector/object-feature consistency only after a corrected restoration mechanism qualifies.
- True CLIP-LIT/CoOp-style prompt learning or a non-CLIP degradation encoder if the current frozen-feature readout becomes limiting again.
- Stronger effective-function-capacity global baseline.
- Meta-learned fast initialization `W_0`.
- ViT3-style fast-weight model mapping local degradation tokens to ISP actions.