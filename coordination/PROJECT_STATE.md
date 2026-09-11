# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Processing for Task-Oriented Vision**

## Core scientific question

Can a vision system adapt a **compact spatial correction field per test image**, without test labels, so that spatially heterogeneous degradation is corrected before a frozen downstream detector processes the image?

## Main hypothesis

A single global enhancement/ISP state is insufficient when degradation varies across an image. A compact spatially varying field can help on heterogeneous degradations that are spatially resolvable, but the test-time objective must distinguish nuisance degradation from valid scene content and must be constrained against destructive drift.

## Current method abstraction

For image `x_t`:

1. Build local degradation tokens `z_i = E_deg(P_i(x_t))`.
2. Use a lightweight fast model `F_W` to map local tokens to correction parameters `phi_i`.
3. Adapt only fast parameters at test time:

   `W_t* = W_0 - eta * grad_W L_TTT(x_t; W_0)`

4. Interpolate/local-compose the resulting parameters into a spatial correction field `Phi_t(u)`.
5. Produce corrected image `x_t* = G(x_t; Phi_t)`.
6. Run a frozen downstream detector `D_theta(x_t*)`.

The current codebase is still at the simpler direct-fast-grid mechanism stage; learned degradation tokens / fast model / detector remain later targets.

## Established controlled findings

- T001: episodic label-free optimization of a bounded spatial ISP field is stable and can strongly outperform one global ISP vector on a favorable heterogeneous exposure toy.
- T002: the gain is **conditional**. On prior-compatible midtone content, spatial4 improves strongly over a many-latent but spatially uniform control for low-frequency left/right, quadrant, and smooth-gradient shifts; the advantage essentially disappears for homogeneous shifts.
- Higher spatial frequency and intrinsic-content confounding can erase or reverse the spatial gain.
- The requested `uniform96` control rules out a trivial raw-latent-count explanation, but it does **not** match effective function capacity; no claim against arbitrary equally expressive global nonlinear correction is accepted.
- Most importantly, the current absolute patch-mean-to-0.5 objective is not identity-safe. It edits undegraded dark/high-key content substantially even as its own self-supervised loss decreases; more spatial freedom can amplify this failure.

## Non-negotiable design principles

- Test-time adaptation must never consume test labels or clean targets.
- Clean references may be used only after adaptation for controlled evaluation.
- Detector weights remain frozen when downstream experiments begin.
- Keep identity/no-adaptation and a global counterpart in all major comparisons.
- Do not infer task utility from lower self-supervised loss alone.
- Spatial capacity should be compact/coarse rather than a free full-resolution correction tensor.

## Candidate future components (not yet approved unless an OPEN task says otherwise)

- CLIP/CLIP-LIT-inspired local degradation direction loss.
- Object/feature consistency to preserve task semantics.
- Edge-aware smoothness and identity anchoring.
- Meta-learned fast initialization `W_0`.
- Learned low-rank spatial regions instead of a fixed grid.
- ViT3-style fast-weight parameterization of the spatial field.
- A stronger effective-function-capacity global nonlinear baseline for later claim validation.

## Milestone M0 — mechanism scaffold

Status: **COMPLETED / T001 ACCEPTED**

Evidence: deterministic global-vs-spatial heterogeneous toy, 13 CPU tests, A6000 CPU/CUDA validation, no-label adaptation boundary verified. Spatial 4x4 reduced evaluation MSE by 75.1% relative to global on the predeclared T001 toy. This is only a controlled mechanism result.

## Milestone M1 — isolate spatiality and objective confounds

Status: **COMPLETED / T002 ACCEPTED**

Evidence: fixed 504-row A6000 matrix, 20 tests, homogeneous/heterogeneous/grid-frequency/content controls, uniform96 latent-count control, and explicit identity-drift failures. Conclusion: low-frequency spatial heterogeneity can justify a spatial field under a compatible objective, but the current absolute midtone prior is unsafe and is now the dominant bottleneck.

## Milestone M2 — objective safety before semantic supervision

Goal: determine whether simple label-free trust-region and spatial smoothness regularization can make the current TTT objective identity-safe without destroying most of its useful heterogeneous-shift gains.

Status: **ACTIVE**

Decision logic: if one fixed regularized configuration materially reduces valid-content drift while retaining useful spatial correction, keep it as a safety baseline. If not, the negative result motivates the next milestone to introduce degradation-aware/semantic self-supervision rather than further tuning the absolute 0.5 prior.

## Open task

`T003 — Can Simple Label-Free Regularization Make Spatial TTT Identity-Safe?` in `coordination/CHATGPT_TO_CODEX.md`.
