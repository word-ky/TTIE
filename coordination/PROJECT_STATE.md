# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Processing for Task-Oriented Vision**

## Core scientific question

Can a vision system adapt a **compact spatial correction field per test image**, without test labels or clean targets, so that spatially heterogeneous nuisance degradation is corrected before a frozen downstream task model processes the image?

## Main hypothesis

A single global enhancement/ISP state is insufficient when degradation varies across an image. A compact spatial field can help when heterogeneous degradation is spatially resolvable, but useful test-time adaptation requires an objective that distinguishes nuisance degradation from legitimate scene content. Spatial capacity alone is not enough and can amplify a bad objective.

## Current method abstraction

For a test image `x_t`:

1. obtain fixed local views/tokens describing current-image appearance;
2. use a compact global or spatial ISP state to produce bounded correction parameters;
3. adapt only this fast state from a label-free objective computed on the current image;
4. render a coarse spatial correction field and corrected image;
5. later, after objective safety is established, evaluate with a frozen detector.

The current codebase still uses direct episodic optimization of a coarse ISP grid. Learned degradation tokens, a learned fast network `F_W`, detector coupling, meta-learned initialization and ViT3-style fast weights remain later targets.

## Established controlled findings

- **T001:** episodic label-free optimization of a bounded spatial ISP field is stable and can strongly outperform one global ISP vector on a favorable heterogeneous exposure toy.
- **T002:** the gain is conditional. On prior-compatible midtone content, spatial freedom strongly helps low-frequency left/right, quadrant and smooth-gradient shifts; it provides essentially no advantage on homogeneous shifts and can become neutral/harmful for high-frequency or content-confounded cases.
- `uniform96` rules out a trivial raw-latent-count explanation, but it is not an equal effective-function-capacity baseline; no claim against arbitrary equally expressive global nonlinear correction is accepted.
- The absolute patch-mean-to-0.5 self-supervised objective is **not identity-safe**. It edits valid intrinsic-dark and high-key content substantially while its own loss decreases; more spatial freedom can amplify this failure.
- **T003:** simple normalized identity anchoring and plain spatial TV do not rescue that absolute prior under the fixed synthetic diagnostic. No one predeclared setting achieves both >=5x reduction of worst clean-content drift and >=70% retention of heterogeneous utility. Strong anchoring can make the field safer only by destroying useful correction. This is a negative result about the tested objective/regularizers/budget, not a universal impossibility result.

## Non-negotiable design principles

- Test-time adaptation must never consume test labels, clean targets, degradation masks, gain maps or evaluation metrics.
- Clean references may be used only after adaptation for controlled evaluation, except that disjoint source/calibration data may be used to freeze development constants before held-out evaluation.
- Keep identity/no-adaptation and a global counterpart in every major spatial claim.
- Do not infer task utility from lower self-supervised loss alone.
- Spatial capacity should remain compact/coarse rather than a free full-resolution correction tensor.
- Separate **objective quality** from **spatial parameterization**: first establish a trustworthy degradation signal, then add larger fast models/meta-learning.

## Milestone M0 — mechanism scaffold

Status: **COMPLETED / T001 ACCEPTED**

Evidence: deterministic heterogeneous toy, 13 CPU tests, A6000 CPU/CUDA validation, no-label adaptation boundary verified. Spatial 4x4 reduced evaluation MSE by 75.1% relative to global on the predeclared T001 toy. Controlled mechanism result only.

## Milestone M1 — isolate spatiality and prior confounds

Status: **COMPLETED / T002 ACCEPTED**

Evidence: fixed 504-row matrix, homogeneous/heterogeneous/grid-frequency/content controls and a 96-latent spatially uniform control. Conclusion: low-frequency spatial heterogeneity can justify a spatial field under a compatible objective, but the absolute midtone prior is unsafe and became the dominant bottleneck.

## Milestone M2 — can simple regularization rescue the absolute prior?

Status: **COMPLETED / T003 ACCEPTED — NEGATIVE DIAGNOSTIC**

Evidence: fixed 936-row / 864-episode A6000-host CPU sweep with focused CUDA validation, 27 tests, exact 11-setting anchor/TV sweep and predeclared 5x-drift/70%-utility conjunction. No setting qualifies. PR #3 merged as `862e3401cf28e9d08582996dad2eb974ad5f9617`.

Conclusion: do not spend the next milestone tuning the same absolute 0.5 prior. Move to degradation-aware semantic evidence.

## Milestone M3 — degradation-aware semantic test-time objective

Status: **ACTIVE / T004 OPEN**

Goal: determine whether a frozen CLIP prior can provide a local exposure-degradation signal that is measurably more content-aware/identity-safe than the absolute 0.5 prior, before learning prompts or introducing a detector.

T004 is intentionally gated:

1. audit the frozen CLIP signal on held-out natural images with fixed prompts and a disjoint clean calibration split;
2. proceed to a global-vs-spatial semantic TTT pilot only if the predeclared signal-quality gate passes;
3. otherwise preserve the negative result and consider learned CLIP-LIT-style prompts in a later task rather than post-hoc prompt tuning.

## Open task

`T004 — Frozen CLIP Local Degradation Signal Audit + Gated Spatial TTT Pilot` in `coordination/CHATGPT_TO_CODEX.md`.

## Candidate later components (not approved unless an OPEN task says otherwise)

- CLIP-LIT-style learned positive/negative degradation prompts on source data.
- Frozen detector / object-feature consistency for task-semantic preservation.
- Edge-aware spatial regularization after the semantic objective is validated.
- Stronger effective-function-capacity global baseline.
- Learned low-rank spatial regions instead of a fixed grid.
- Meta-learned fast initialization `W_0`.
- ViT3-style fast-weight model mapping local degradation tokens to ISP actions.
