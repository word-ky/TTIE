# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Processing for Task-Oriented Vision**

## Core scientific question

Can a vision system adapt a **compact spatial correction field per test image**, without test labels or clean targets, so that spatially heterogeneous nuisance degradation is corrected before a frozen downstream task model processes the image?

## Main hypothesis

A single global enhancement state is insufficient when degradation varies across an image. A compact spatial field can help, but only if three pieces align: (i) a nuisance/degradation signal that is reasonably content-safe, (ii) a label-free correction objective whose gradient **and magnitude/stopping rule** correlate with useful restoration, and (iii) a spatial action basis whose support does not unnecessarily couple regions that require opposite corrections. Spatial capacity by itself can amplify a bad objective or a bad renderer.

## Current method abstraction

For a test image `x_t`:

1. extract fixed local quadrant views;
2. obtain a frozen source-trained exposure readout in CLIP feature space;
3. apply the frozen T007 joint clean-abstention gate to decide which quadrants are active and whether they look dark or bright;
4. optimize a compact EV+gamma fast state from current pixels only;
5. use the original zero-envelope semantic objective, but constrain the fast state to a **gate-consistent direct-scale trust box**: inactive regions remain identity, dark regions may only increase EV up to +0.5, bright regions may only decrease EV down to -0.5, and gamma stays in [0.8,1.25];
6. compare global, bilinear 2x2, and fixed quadrant-aligned `region2` spatial controls, including direct/discrete non-gradient baselines;
7. attach clean-reference metrics only after every label-free decision/output is finalized.

This projected action geometry is the T011 candidate mechanism. Learned fast networks, meta-initialization, detector coupling and ViT3-style fast weights remain later targets.

## Established controlled findings

- **T001:** episodic label-free optimization of a bounded spatial ISP field is stable and can strongly outperform one global ISP vector on a favorable heterogeneous exposure toy.
- **T002:** the spatial gain is conditional. It helps low-frequency heterogeneous shifts under a compatible objective, provides essentially no advantage on homogeneous shifts, and can become neutral/harmful for high-frequency or content-confounded cases. A 96-latent uniform control rules out a trivial raw-parameter-count explanation.
- **T003:** the absolute patch-mean-to-0.5 prior is not identity-safe, and simple identity anchoring + plain TV do not rescue it without destroying useful correction.
- **T004:** fixed zero-shot absolute frozen-CLIP exposure scores have useful dark sensitivity but poor bright discrimination; cross-image content offsets remain large.
- **T005:** a fixed same-image ±0.25 EV relative CLIP response does not solve the content problem and underperforms the unchanged absolute score.
- **T006:** source-trained three-prototype readout on frozen CLIP features yields strong exposure ranking and mixed-region localization, but its first independent gate has excessive clean false activation. The learned vectors are a discriminative frozen-CLIP readout, not preserved text semantics.
- **T007:** a predeclared image-level joint clean-abstention envelope fixes much of that safety issue: clean all-view FPR falls to **6%**, clean image-any activation to **20%**, while retaining usable dark/bright recall. The recall cost is material.
- **T008:** the first frozen-signal EV+gamma correction pilot shows that the readout can drive real correction, but the current TTT objective is not qualified. Spatial TTT improves homogeneous dark/bright MSE by **61.26% / 58.70%** and pooled heterogeneous MSE by **16.18%** versus identity, yet fails clean-tail safety, beats global TTT by only **4.45%**, is **14.73% worse** than a fixed local ±0.5 EV action, and worsens heterogeneous bright-region MSE by **13.65%**. Lower semantic loss alone is therefore not a sufficient restoration proxy.
- **T009:** development-only geometry audit separates the T008 failure. The pooled spatial EV+gamma semantic gradient is positively aligned with reference restoration at identity in **94.77%** of active non-clean episodes with median cosine **0.7973**, so a blanket objective-direction failure is not supported. However, **92.81%** improve after the first update while **69.28%** have a strictly better earlier MSE step than the final semantic stop: over-correction/stopping is a dominant failure. Gamma removal does **not** meet its predeclared criterion (EV-only pooled MSE gain only **0.45%**, with 94.89% semantic reduction retained). The renderer rule does fire: on exact quadrant shifts, a quadrant-constant field improves MSE by **21.60%** over bilinear under identical direct actions and by **87.96%** under the fixed development-only reference oracle. Important limitation: heterogeneous dark-region gradient median is **-0.2099** despite positive whole-image alignment, and the hard piecewise field is geometrically matched to the synthetic quadrant boundaries.
- **T010:** the source-calibrated two-scalar residual target is a **controlled negative result**. PR #10 is squash-merged as `86c41bd0ff144dcb990f52b4094cea99d40ff7c6`. Zero of 16 predeclared `(rho_dark,rho_bright)` pairs is feasible. All satisfy clean mean/p95 safety, but none achieves the required 5% heterogeneous gain over `region2_direct`. The descriptive minimum `(.25,.25)` has clean p95 **0.00307146**, improves homogeneous dark/bright by **41.35% / 43.81%** vs identity, but improves pooled heterogeneous MSE over direct by only **1.8164%** (`0.03838832` vs `0.03909849`). The original `rho=0` `region2` envelope control is stronger on heterogeneous restoration (`0.03410695`, about **12.8%** better than direct) but narrowly fails clean-tail safety (`p95=0.00541823 > 0.005`). Stage B was correctly not run; no fresh T010 evaluation data was touched. The result establishes a safety–utility frontier for this residual-target family, not a general failure of stopping or spatial TTT.

## Non-negotiable design principles

- Test-time adaptation must never consume test labels, clean targets, degradation masks, gain maps, condition IDs, annotations, or evaluation metrics.
- Clean references may be used only after adaptation for controlled evaluation, except in explicitly source/development-side diagnostic or calibration modules whose outputs are frozen before later held-out evaluation.
- Keep identity/no-adaptation and a global counterpart in every major spatial claim.
- Do not infer restoration or task utility from lower self-supervised loss alone.
- Spatial capacity should remain compact/coarse rather than a free full-resolution correction tensor.
- Separate representation quality, calibration safety, gradient alignment, action magnitude/stopping, spatial basis, and optimization mechanism before adding larger fast models/meta-learning.
- After a result motivates a new hypothesis, use a fresh deterministic split for the next decisive audit; inspected images are unavailable for corrective tuning.
- Source-trained signals or calibration constants are allowed only with strict development/evaluation separation; synthetic source labels must never become test-time metadata inputs.
- A gradient-based TTT claim must beat non-gradient direct/discrete actions using the same frozen signal. If simple policies match or win, do not attribute value to test-time training.
- Hard region-aligned fields must be stress-tested on at least one boundary-misaligned condition before any claim of general spatial parameterization superiority.

## Milestone M0 — mechanism scaffold

Status: **COMPLETED / T001 ACCEPTED**

## Milestone M1 — isolate spatiality and prior confounds

Status: **COMPLETED / T002 ACCEPTED**

## Milestone M2 — can simple regularization rescue the absolute prior?

Status: **COMPLETED / T003 ACCEPTED — NEGATIVE DIAGNOSTIC**

## Milestone M3 — degradation-aware semantic readout and first correction pilot

Status: **COMPLETED THROUGH T008 — READOUT QUALIFIED, ORIGINAL CLEAN-ENVELOPE TTT NOT QUALIFIED**

T004/T005 are negative signal diagnostics; T006 establishes a strong source-trained frozen readout; T007 qualifies the joint abstention gate; T008 shows useful correction signal but fails as a safe/competitive TTT mechanism.

## Milestone M4 — objective/action geometry diagnosis

Status: **COMPLETED / T009 ACCEPTED AS DEVELOPMENT-ONLY DIAGNOSTIC**

T009 identifies over-correction/stopping and bilinear renderer coupling as real failure modes, while not supporting a blanket objective-gradient failure or gamma-removal claim.

## Milestone M5 — residual-target stopping repair

Status: **COMPLETED / T010 ACCEPTED — NEGATIVE DEVELOPMENT GATE**

The fixed residual-evidence target makes the clean tail safer but removes too much heterogeneous correction utility. No predeclared source-side configuration beats `region2_direct` by the required margin, so T010 correctly stops before held-out evaluation. Do not refine the rho grid on the same development data.

## Milestone M6 — gate-consistent projected action geometry

Status: **ACTIVE / T011 OPEN**

T011 keeps the stronger original zero-envelope semantic objective and tests a different repair: constrain the action space itself to the gate-consistent direct-scale range. The primary `region2` method must keep inactive quadrants at exact identity, restrict dark/bright EV direction and magnitude to ±0.5, and restrict gamma to [0.8,1.25]. A fresh 40-image evaluation split is used with no new calibration sweep.

T011 must compare projected spatial TTT against identity, matched global TTT, fixed direct correction, gate-consistent discrete search, the unconstrained envelope control, a bilinear renderer ablation, and an exact one-step projected TTT diagnostic. Qualification still requires clean-tail safety, homogeneous utility, spatial value over global, regional safety, renderer value, and at least 5% improvement over both direct and matched discrete non-gradient baselines.

No detector work, meta-learning, or ViT3-style fast model is authorized until this action-geometry question is resolved.

## Open task

`T011 — Gate-Consistent Projected Spatial TTT` in `coordination/CHATGPT_TO_CODEX.md`.

## Candidate later components (not approved unless an OPEN task says otherwise)

- Source-trained/task-aligned inner objective or learned stopping controller if T011 still cannot make gradient adaptation beat matched direct/discrete actions.
- Soft/learned region basis or low-rank spatial basis after a boundary-misaligned stress result justifies replacing hard `region2`.
- EV-only simplification only if future fresh evidence, not T009 pooled averages alone, shows gamma consistently harmful.
- Frozen detector/object-feature consistency only after a corrected restoration mechanism qualifies.
- True CLIP-LIT/CoOp-style learned prompt tokens or a non-CLIP degradation encoder if the current frozen-feature readout becomes limiting again.
- Stronger effective-function-capacity global baseline.
- Meta-learned fast initialization `W_0`.
- ViT3-style fast-weight model mapping local degradation tokens to ISP actions.
