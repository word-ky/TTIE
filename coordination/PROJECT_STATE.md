# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Processing for Task-Oriented Vision**

## Core scientific question

Can a vision system adapt a **compact spatial correction field per test image**, without test labels, so that spatially heterogeneous degradation is corrected before a frozen downstream detector processes the image?

## Main hypothesis

A single global enhancement/ISP state is insufficient when degradation varies across an image. A compact spatially varying field should outperform global correction especially under heterogeneous degradations, provided that the test-time objective is constrained to preserve task semantics.

## Current method abstraction

For image `x_t`:

1. Build local degradation tokens `z_i = E_deg(P_i(x_t))`.
2. Use a lightweight fast model `F_W` to map local tokens to correction parameters `phi_i`.
3. Adapt only fast parameters at test time:

   `W_t* = W_0 - eta * grad_W L_TTT(x_t; W_0)`

4. Interpolate/local-compose the resulting parameters into a spatial correction field `Phi_t(u)`.
5. Produce corrected image `x_t* = G(x_t; Phi_t)`.
6. Run a frozen downstream detector `D_theta(x_t*)`.

The current codebase is still at the simpler direct-fast-grid mechanism stage; the learned token encoder / fast model above remains a later method target, not yet implemented.

## Initial design principles

- Do **not** start with a free `H x W x d` correction tensor.
- Start with a low-dimensional/coarse-grid spatial field.
- Start with four differentiable operators only: exposure, gamma, white balance, contrast.
- Detector weights remain frozen during test-time adaptation.
- Test-time loss must not use target labels.
- Always compare against identity/no adaptation and a global-parameter counterpart.
- Before adding semantic teachers or downstream tasks, isolate spatial structure from parameter-count and self-supervised-prior confounds.

## Candidate future components (not yet approved for implementation)

- CLIP/CLIP-LIT-inspired local degradation direction loss.
- Object/feature consistency to preserve task semantics.
- Edge-aware smoothness and identity anchoring.
- Meta-learned fast initialization `W_0`.
- Learned low-rank spatial regions instead of a fixed grid.
- ViT3-style fast-weight parameterization of the spatial field.

## Milestone M0 — mechanism scaffold

Goal: prove that a differentiable spatial correction field and episodic test-time optimization can be implemented stably, with no test labels and no detector updates.

Status: **COMPLETED / T001 ACCEPTED**

Evidence: deterministic global-vs-spatial heterogeneous toy, 13 CPU tests, A6000 CPU/CUDA validation, no-label adaptation boundary verified. On the predeclared T001 toy, spatial 4x4 adaptation reduced evaluation MSE by 75.1% relative to global adaptation. This is accepted only as a controlled mechanism result, not a natural-image or downstream-task claim.

## Milestone M1 — isolate spatiality and objective confounds

Goal: determine whether the observed gain is truly due to spatially varying correction rather than extra degrees of freedom or a favorable midtone prior; characterize when spatiality is useful and when the current label-free objective fails.

Status: **ACTIVE**

## Open task

`T002 — Isolate Spatiality from Capacity and Prior Confounds` in `coordination/CHATGPT_TO_CODEX.md`.
