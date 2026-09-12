# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Processing for Task-Oriented Vision**

## Core scientific question

Can a vision system adapt a **compact spatial correction field per test image**, without test labels or clean targets, so that spatially heterogeneous nuisance degradation is corrected before a frozen downstream task model processes the image?

## Main hypothesis

A single global enhancement/ISP state is insufficient when degradation varies across an image. A compact spatial field can help when heterogeneous degradation is spatially resolvable, but useful test-time adaptation requires both (i) a degradation signal that separates nuisance shift from legitimate content and (ii) a correction objective whose gradient/action magnitude is aligned with useful restoration. Spatial capacity alone can amplify a misaligned objective.

## Current method abstraction

For a test image `x_t`:

1. extract fixed local image views;
2. obtain a frozen source-trained degradation readout in CLIP feature space;
3. apply a clean-calibrated abstention gate to decide where adaptation is warranted;
4. adapt only a compact bounded ISP fast state from current pixels;
5. render a global or coarse spatial EV/gamma field;
6. evaluate clean-reference or downstream utility only after the label-free adaptation path is finalized.

The current codebase uses direct episodic optimization of a coarse ISP state. T008 established that the frozen readout/gate contains useful exposure information, but the current two-sided clean-envelope objective is not sufficiently aligned with restoration magnitude. T009 is therefore a source-side geometry audit before any new held-out correction pilot. Learned fast networks, meta-initialization, detector coupling and ViT3-style fast weights remain later targets.

## Established controlled findings

- **T001:** episodic label-free optimization of a bounded spatial ISP field is stable and can strongly outperform one global ISP vector on a favorable heterogeneous exposure toy.
- **T002:** the gain is conditional. Spatial freedom helps low-frequency heterogeneous shifts under a compatible objective, provides essentially no advantage on homogeneous shifts, and can become neutral/harmful for high-frequency or content-confounded cases. `uniform96` rules out a trivial raw-latent-count explanation, but not every equal-function-capacity global model.
- The absolute patch-mean-to-0.5 objective is not identity-safe; it edits legitimate intrinsic-dark/high-key content while lowering its own loss.
- **T003:** simple identity anchoring + plain TV do not rescue that absolute prior under the fixed safety/utility diagnostic. Strong anchoring only becomes safe by destroying useful correction.
- **T004:** fixed zero-shot absolute frozen-CLIP exposure scores show useful dark sensitivity but poor bright discrimination; cross-image content offsets remain large.
- **T005:** a fixed same-image ±0.25 EV relative CLIP response does not solve the content problem; it underperforms the unchanged absolute score on the same fresh split.
- **T006:** source-trained three-prototype readout on frozen CLIP features produces strong exposure ranking and spatial localization on fresh images (dark/bright AUC 0.9870/0.9366; homogeneous correct-type TPR 93%/92%; mixed correct recall 91.25%/95%), but the initial independent gate fails clean identity safety at 21% all-view FPR. The learned vectors should be treated as a discriminative frozen-CLIP readout, not preserved text semantics.
- **T007:** with the representation completely frozen, a predeclared image-level joint clean-abstention envelope passes every declared safety/recall gate on a new 40-image audit. It reduces clean all-view FPR from 21.5% to **6%** and clean image-any activation from 45% to **20%**, while retaining homogeneous correct-type TPR **62.5%/74.5%** and mixed dark/bright correct recall **61.875%/75.625%** with only **1.25%/0%** wrong-type mixed activation. The safety gain has a material recall cost.
- **T008:** the first frozen-signal EV+gamma correction pilot is a controlled negative result for the current TTT objective. `spatial2_ttt` improves homogeneous dark/bright MSE by **61.26% / 58.70%** versus identity and pooled heterogeneous MSE by **16.18%**, proving the signal can drive nontrivial correction. However it fails clean-tail safety (p95 drift **0.009289 > 0.005**), beats global TTT by only **4.45%** on heterogeneous cases, is **14.73% worse** than the fixed `spatial2_direct` action, and worsens heterogeneous bright-region MSE by **13.65%** versus identity.
- T008's key mechanism warning is stronger than the aggregate failure: heterogeneous frozen semantic loss falls roughly **6.07 → 0.66** while restoration remains inferior to a simple ±0.5 EV policy. Therefore semantic-loss minimization and restoration magnitude are not sufficiently aligned. This is not yet attributable uniquely to gradient descent because semantic discrete search can also prefer aggressive EV/gamma actions, and the bilinear 2x2 renderer may couple local regions.

## Non-negotiable design principles

- Test-time adaptation must never consume test labels, clean targets, degradation masks, gain maps, condition IDs or evaluation metrics.
- Clean references may be used only after adaptation for controlled held-out evaluation, except in explicitly source/development-side diagnostic or training modules whose outputs are frozen before later held-out evaluation.
- Keep identity/no-adaptation and a global counterpart in every major spatial claim.
- Do not infer restoration or task utility from lower self-supervised loss alone.
- Spatial capacity should remain compact/coarse rather than a free full-resolution correction tensor.
- Separate **representation quality**, **decision/calibration safety**, **objective-gradient alignment**, **action magnitude/stopping**, **spatial parameterization**, and **optimization mechanism** before adding larger fast models/meta-learning.
- After a held-out result motivates a new hypothesis, use a fresh deterministic split for the next decisive audit; inspected evaluation images become unavailable for corrective hyperparameter selection.
- Source-trained degradation signals are allowed only with strict source/development/calibration/evaluation separation; source synthetic labels must never become test-time metadata inputs.
- A gradient-based TTT claim must be compared against non-gradient direct/discrete actions using the same frozen signal; if simpler policies match or beat TTT, do not attribute value to test-time training itself.

## Milestone M0 — mechanism scaffold

Status: **COMPLETED / T001 ACCEPTED**

Spatial 4x4 reduced evaluation MSE by 75.1% relative to global on the predeclared favorable toy. Controlled mechanism result only.

## Milestone M1 — isolate spatiality and prior confounds

Status: **COMPLETED / T002 ACCEPTED**

Low-frequency spatial heterogeneity can justify a spatial field under a compatible objective, but the absolute midtone prior is unsafe and became the dominant bottleneck.

## Milestone M2 — can simple regularization rescue the absolute prior?

Status: **COMPLETED / T003 ACCEPTED — NEGATIVE DIAGNOSTIC**

No predeclared anchor/TV setting achieves both the required clean-drift reduction and heterogeneous-utility retention. Do not tune the same absolute 0.5 prior further.

## Milestone M3 — degradation-aware semantic signal and first correction pilot

Status: **COMPLETED THROUGH T008 — READOUT QUALIFIED, CURRENT CLEAN-ENVELOPE TTT NOT QUALIFIED**

### T004 — absolute frozen-CLIP gate

**COMPLETED / ACCEPTED — NEGATIVE STAGE-A DIAGNOSTIC.** PR #4 merged as `8c9680c56ced11d4433908091a4e4575ca04ab1f`.

### T005 — counterfactual relative frozen-CLIP signal

**COMPLETED / ACCEPTED — NEGATIVE STAGE-A DIAGNOSTIC.** PR #5 merged as `0d2146257135c2b0c3568cbf579d6c78d2354114`.

### T006 — source-trained CLIP exposure prototypes

**COMPLETED / ACCEPTED — STRONG READOUT, INITIAL IDENTITY GATE FAILURE.** PR #6 merged as `5c9d6e61a3f6fdea8d528f26ae6401893bcaf6a2`.

### T007 — joint clean-abstention calibration

**COMPLETED / ACCEPTED — PASSED SAFETY/RECALL GATE.** PR #7 merged as `0713267771f75b4c51e1413ca46b9823004aa6e9`.

### T008 — first semantic global-vs-spatial EV+Gamma TTT pilot

**COMPLETED / ACCEPTED AS NEGATIVE CORRECTION DIAGNOSTIC.** PR #8 squash-merged as `1568f56ac12fd19a642525a81213087b9e9df756`.

Conclusion: the frozen learned exposure readout plus joint abstention gate is strong enough to warrant continued study, but driving every active region all the way into the source clean-envelope with unconstrained EV+gamma TTT is not justified. A simple fixed local EV action is currently a stronger heterogeneous restoration baseline.

## Milestone M4 — objective/action geometry before another held-out pilot

Status: **ACTIVE / T009 OPEN**

T009 will use a new development-only image split to separate four possible causes of the T008 gap:

1. semantic-gradient vs restoration-gradient misalignment;
2. over-correction / semantic stopping beyond the pixel-restoration optimum;
3. harmful or unnecessary gamma adaptation;
4. bilinear 2x2 spatial-field coupling across heterogeneous regions.

T009 may use clean references only in structurally separate offline diagnostic/oracle computations. Its simulated TTT path remains label-free and identical in leakage boundary to deployment. The T009 development images will be excluded from later decisive held-out evaluation.

No detector work, meta-learning, or ViT3-style fast model is authorized until a corrected label-free objective/action mechanism passes a fresh held-out restoration pilot.

## Open task

`T009 — Source-Side Semantic/Restoration Geometry Audit` in `coordination/CHATGPT_TO_CODEX.md`.

## Candidate later components (not approved unless an OPEN task says otherwise)

- Source-calibrated semantic target/trust region if T009 shows positive early gradient alignment but over-correction at the current clean-envelope stop.
- EV-only spatial TTT if gamma is shown to be a dominant harmful coordinate.
- Piecewise/region-aware or learned low-rank spatial basis if bilinear coupling is shown to dominate checkerboard failure.
- Source-trained/task-aligned inner objective if semantic gradients themselves are poorly aligned with restoration gradients.
- Frozen detector / object-feature consistency only after a corrected restoration mechanism qualifies on a fresh split.
- True CLIP-LIT/CoOp-style learned prompt tokens or a non-CLIP degradation encoder if the current frozen-feature readout becomes the limiting factor again.
- Stronger effective-function-capacity global baseline.
- Meta-learned fast initialization `W_0`.
- ViT3-style fast-weight model mapping local degradation tokens to ISP actions.
