# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Enhancement**

This file is the live scientific state. Detailed task-by-task history remains in Git history and `research_log/`.

## Core scientific question

Can a vision system adapt a compact spatial image-processing state per test image, without test labels or clean/normal-light targets, so that unknown and spatially heterogeneous degradation is corrected by a reference-free test-time optimization field?

## Current deployable state

**T014 Sobolev Region2 TTT remains the broad fresh-qualified Ours-Core.** On LOL-v2 Real development data, the accepted fixed-validation deployable base remains **T026-A: `11.1208764 dB / 0.3737918 RGB-SSIM`**. T036-A common gain remains the strongest fresh-qualified deployable action expansion found so far, improving exact T026-A by `+0.9392571 dB / +0.0083560 RGB-SSIM` mean on its deterministic reference-unused development cohort, but with an unresolved unsafe tail.

No T059 matched-detail result is deployable. The official LOL-v2 Real test remains sealed.

## Renderer / action-family diagnosis

Reference-only renderer studies established that the remaining capacity is strongly spatial:

- **T051-A spatial exposure field:** fixed 8×8 RGB-shared exposure field adds `+0.6590052 dB` mean / `+0.4525454 dB` median PSNR over T050, with `100/100` PSNR wins. It missed its intentionally large oracle materiality gate, but it is clearly a larger low-frequency action family than the later deployed Region2/global variants.
- **T054-A local-detail field:** strongest later reference-only renderer extension, reaching `24.3454867 / 0.7577572` and adding `+1.2383356 dB` mean / `+1.2353360 dB` median PSNR plus `+0.1877170` mean SSIM over T052.
- T055-A/T055-V show that simply extending the same one-scale optimization budget adds only about `+0.0043 dB`; T056 second-scale detail and T057 chroma-detail failed their fixed materiality gates.

All of these are non-deployable `REFERENCE_ORACLE_ONLY` diagnostics. Their clean targets, reference gradients/states, PSNR/SSIM, and per-image oracle quantities are forbidden from test-time inference.

## T059 matched-detail optimization-field diagnosis — closed as a practical detail route

The T059 program established a stable asymmetry: **direction transfers much better than calibrated scalar value**, but the tested detail action family is not practically useful enough to continue tuning.

Accepted conclusions remain:

- calibrated unseen-image scalar prediction is unsupported despite strong in-sample capacity;
- simple 28-D locality, bank centering/context, early stopping, frozen-CLIP Euclidean locality, donor smoothing/oracles, dual-head disagreement, optimizer-scale rescue, and low-gradient-norm abstention do not solve the value/safety problem;
- the detail field itself is directionally real: T059-E source inner-held alignment is `0.868874` positive-dot / `0.491553` median cosine; T059-S/T and T059-U show most source anchors have useful local descent direction; T059-V/W show the degraded-image `dx/dv` bridge and one-step action can be reconstructed fully online without reference-dependent caches;
- T059-X transfers to the already-open 100-image LOL-v2 Real development cohort from raw input, with `97/100` MSE improvements, but only **`+0.001135428 dB` mean PSNR**;
- **T059-Y** applies the same frozen field after exact accepted T026-A global correction. It remains numerically positive but fails materiality: `100/100` active, `91/100` MSE improve, mean/median MSE change `-4.3532840e-5 / -1.9162789e-5`, mean RGB-SSIM `+0.0014076024`, but mean paired PSNR only **`+0.0024241736 dB < +0.05 dB`**. The harmful tail is `9/100`.

Therefore the fixed one-step matched-detail integration line is **closed**. No lr/step-count/optimizer/threshold rescue on this opened cohort, no T036+detail combination, and no official-test promotion are authorized.

### Current scientific interpretation after T059-Y

The useful result from T059 is not a deployable detail enhancer; it is evidence that the frozen source-trained feature-space restoration gradient can contain transferable action direction. The next scientifically cleaner question is whether that field transfers to a **more relevant low-frequency action family**.

Severe low-light error is dominated by illumination/exposure, while T054 detail acts mainly on local high-frequency structure. T059-Y's `+0.001408` mean SSIM but only `+0.002424 dB` mean PSNR reinforces that mismatch. T051-A provides a pre-existing, fixed 8×8 RGB-shared spatial exposure renderer with materially larger reference-only capacity and no need to invent a new action family.

The new branch therefore tests **action-family transfer of the same frozen energy field**, not another value model and not another detail-specific rescue.

## Strong baseline development anchors

- **Retinexformer T033-A:** `21.4787864 / 0.7900612`.
- **SNR-Aware T045-A:** `23.3963299 / 0.8237644`.

Both are target-free at inference under the frozen comparison protocol, but their released supervised checkpoints are exposed to LOL-v2 Real training data containing this development split. They are training-exposed development anchors, not independent held-out SOTA evidence.

The final sprint objective remains a clear **`+2–3 dB` PSNR advantage over the strongest fair target-free baseline on the same held-out protocol**, without violating the no-test-target rule. This is an objective, not a current claim.

## Information-boundary rules

- Test-time adaptation and checkpoint/state selection must never consume test labels, clean/normal-light targets, reference gradients/Jacobians, oracle values, PSNR/SSIM, degradation masks/gain maps, condition IDs, annotations, or semantic image IDs.
- A degraded-image feature/action Jacobian such as `dx/dv` or `dx/du`, computed entirely from the current degraded image/current target-free intermediate image and frozen model, is permissible; a Jacobian/gradient that depends on a clean/reference target is not.
- Validation/test outputs and decisions must be finalized and persisted before references or evaluation metrics are attached, except in explicitly isolated non-deployable source/reference diagnostics.
- Source-training clean/reference targets may be used only for source-supervised training or isolated source-domain diagnostics; they are never admissible test-time inputs.
- Reference diagnostics may motivate only global research choices; no per-image oracle quantity may enter deployable inference.
- External baselines admitted to the main comparison must be target-free at inference.
- Fresh/final benchmark sets must remain isolated from model/hyperparameter selection.
- **The official LOL-v2 Real test remains untouched through T059-Y.**
- Fresh/test runs must fail closed on source/checkpoint/cohort/provenance mismatches.

## Best current methods / ceilings

- **Broad fresh-qualified Ours-Core:** T014 Sobolev Region2 TTT.
- **Best fixed-validation deployable candidate:** T026-A, `11.1208764 / 0.3737918`.
- **Fresh-qualified deployable action extension:** T036-A common gain, `+0.9392571 dB` mean on its fresh cohort, unsafe tail unresolved.
- **Best promoted non-deployable mechanism state:** T055-A, `24.3497922 / 0.7591279`; T056/T057 are valid but failed materiality gates.
- **T059 matched-detail:** direction transfer is scientifically supported, but raw and T026-conditioned real-development gains are practically negligible; the integration line is closed.

## Integration note

Historical PRs may contain inherited history/integration complications. Preserve accepted scientific/evidence states rather than repairing history inside experiment cycles. T059-P through T059-Y are scientifically reviewable through their pinned source/evidence even while PRs remain open. T059-Y is PR #124.

## Current open task

**T060-A — frozen-energy spatial-exposure direction transfer audit** in `coordination/CHATGPT_TO_CODEX.md`.

Reuse exactly the 80 fixed source outer state-0 anchors. Keep the T059-E head/features frozen and project `q=dE/dx` through the exact T051-style 8×8 RGB-shared spatial-exposure Jacobian `J_exp=dx/du`. Freeze all target-free `x/J_exp/q/g_hat` tensors before any source clean/reference read, then compute source-only true restoration gradients for a direction-alignment diagnostic. No optimizer step, no real-development rollout, no target-domain fitting, and no official-test access are authorized in this cycle.
