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
4. adapt only a compact bounded EV+gamma fast state from current pixels;
5. stop against a source-calibrated **residual semantic-evidence target** rather than forcing every active view to the zero-evidence clean envelope;
6. compare a global state, bilinear 2x2 field, and a fixed quadrant-aligned 2x2 region field;
7. attach clean-reference metrics only after every label-free decision/output is finalized.

The residual target and region-aligned renderer are the T010 candidate repair, not yet a qualified result. Learned fast networks, meta-initialization, detector coupling and ViT3-style fast weights remain later targets.

## Established controlled findings

- **T001:** episodic label-free optimization of a bounded spatial ISP field is stable and can strongly outperform one global ISP vector on a favorable heterogeneous exposure toy.
- **T002:** the spatial gain is conditional. It helps low-frequency heterogeneous shifts under a compatible objective, provides essentially no advantage on homogeneous shifts, and can become neutral/harmful for high-frequency or content-confounded cases. A 96-latent uniform control rules out a trivial raw-parameter-count explanation.
- **T003:** the absolute patch-mean-to-0.5 prior is not identity-safe, and simple identity anchoring + plain TV do not rescue it without destroying useful correction.
- **T004:** fixed zero-shot absolute frozen-CLIP exposure scores have useful dark sensitivity but poor bright discrimination; cross-image content offsets remain large.
- **T005:** a fixed same-image ±0.25 EV relative CLIP response does not solve the content problem and underperforms the unchanged absolute score.
- **T006:** source-trained three-prototype readout on frozen CLIP features yields strong exposure ranking and mixed-region localization, but its first independent gate has excessive clean false activation. The learned vectors are a discriminative frozen-CLIP readout, not preserved text semantics.
- **T007:** a predeclared image-level joint clean-abstention envelope fixes much of that safety issue: clean all-view FPR falls to **6%**, clean image-any activation to **20%**, while retaining usable dark/bright recall. The recall cost is material.
- **T008:** the first frozen-signal EV+gamma correction pilot shows that the readout can drive real correction, but the current TTT objective is not qualified. Spatial TTT improves homogeneous dark/bright MSE by **61.26% / 58.70%** and pooled heterogeneous MSE by **16.18%** versus identity, yet fails clean-tail safety, beats global TTT by only **4.45%**, is **14.73% worse** than a fixed local ±0.5 EV action, and worsens heterogeneous bright-region MSE by **13.65%**. Lower semantic loss alone is therefore not a sufficient restoration proxy.
- **T009:** development-only geometry audit separates the T008 failure. The pooled spatial EV+gamma semantic gradient is positively aligned with reference restoration at identity in **94.77%** of active non-clean episodes with median cosine **0.7973**, so a blanket objective-direction failure is not supported. However, **92.81%** improve after the first update while **69.28%** have a strictly better earlier MSE step than the final semantic stop: over-correction/stopping is a dominant failure. Gamma removal does **not** meet its predeclared criterion (EV-only pooled MSE gain only **0.45%**, with 94.89% semantic reduction retained). The spatial renderer rule does fire: on exact quadrant shifts, a quadrant-constant field improves MSE by **21.60%** over bilinear under identical direct actions and by **87.96%** under the fixed development-only reference oracle. Left/right shows the same direction. Important limitation: heterogeneous dark-region gradient median is **-0.2099** despite positive whole-image alignment, and the hard piecewise field is geometrically matched to the synthetic quadrant boundaries.

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

PR #9 squash-merged as `669e258af2f197b2be93302890d976e7d79d62f0`.

T009 fires two predeclared categories:

1. **over-correction / stopping failure** — early gradients are usually useful, but the zero-evidence semantic stop is too aggressive;
2. **renderer failure** — bilinear 2x2 coupling materially hurts exact quadrant correction relative to a quadrant-constant field.

The objective-gradient failure and gamma-removal rules do not fire. These findings motivate a targeted repair rather than optimizer retuning or wholesale objective replacement.

## Milestone M5 — corrected label-free spatial TTT

Status: **ACTIVE / T010 OPEN**

T010 tests exactly two repairs:

- a source-calibrated residual-evidence semantic target that stops before zero-evidence over-correction;
- a fixed quadrant-aligned `region2` renderer whose support matches the fixed local semantic views.

Stage A uses only T009 development images and may use clean references to choose `rho_dark/rho_bright` from a predeclared finite grid. If no configuration is simultaneously identity-safe, useful on homogeneous exposure, and better than fixed region-wise direct correction on heterogeneous development data, T010 stops without touching fresh evaluation outcomes.

If Stage A passes, the selected constants are frozen before a new T010 held-out split is scored. Stage B must compare identity, global TTT, bilinear spatial TTT, region-aligned spatial TTT, fixed direct correction, and discrete search. A gradient-based claim requires the corrected spatial TTT to beat both global TTT and the direct/discrete non-gradient baselines while satisfying clean and regional safety. A boundary-misaligned stress condition is report-only to expose hard-quadrant brittleness.

No detector work, meta-learning, or ViT3-style fast model is authorized until a corrected label-free restoration mechanism qualifies on a fresh split.

## Open task

`T010 — Source-Calibrated Residual-Evidence Target + Region-Aligned Spatial TTT` in `coordination/CHATGPT_TO_CODEX.md`.

## Candidate later components (not approved unless an OPEN task says otherwise)

- Soft/learned region basis or low-rank spatial basis if hard `region2` is brittle under boundary misalignment.
- Source-trained/task-aligned inner objective if the residual-target repair still leaves systematic regional gradient misalignment.
- EV-only simplification only if future fresh evidence, not T009 pooled averages alone, shows gamma consistently harmful.
- Frozen detector/object-feature consistency only after a corrected restoration mechanism qualifies.
- True CLIP-LIT/CoOp-style learned prompt tokens or a non-CLIP degradation encoder if the current frozen-feature readout becomes limiting again.
- Stronger effective-function-capacity global baseline.
- Meta-learned fast initialization `W_0`.
- ViT3-style fast-weight model mapping local degradation tokens to ISP actions.
