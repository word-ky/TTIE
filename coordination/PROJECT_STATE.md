# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Enhancement**

This file is the live scientific state. Detailed task-by-task history remains in Git history and `research_log/`.

## Core scientific question

Can a vision system adapt a compact spatial image-processing state per test image, without test labels or clean/normal-light targets, so that unknown and spatially heterogeneous degradation is corrected by a reference-free test-time optimization field?

## Current deployable state

**T014 Sobolev Region2 TTT remains the broad fresh-qualified Ours-Core.** On LOL-v2 Real development data, the accepted fixed-validation deployable base remains **T026-A: `11.1208764 dB / 0.3737918 RGB-SSIM`**. T036-A common gain remains the strongest fresh-qualified deployable action expansion found so far, improving exact T026-A by `+0.9392571 dB / +0.0083560 RGB-SSIM` mean on its deterministic reference-unused development cohort, but with an unresolved unsafe tail (`29/100` PSNR regressions; worst `-5.614 dB`).

No T059/T060 action-transfer result is deployable yet. The official LOL-v2 Real test remains sealed.

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

## T060 action-family transfer diagnosis

### T060-A — direct 8×8 spatial-exposure projection: negative

T060-A tested whether the same frozen T059-E feature-space field transfers directly into the exact T051-style 8×8 RGB-shared spatial-exposure action family on the fixed 80-anchor source outer cohort.

The experiment is scientifically admissible: all target-free `x/J_exp/q/g_hat` tensors froze before any source clean/reference read; the T051 renderer math, T059-E head/features/normalization, gate/state inputs, and cohort were pinned; no optimizer step, target-domain/LOL-v2/official-test access, or reference-derived input entered prediction; an independent analytic verifier replayed the exposure derivative/bilinear adjoint and scalar summaries.

Frozen result:

- nondegenerate: **`65/80 < 72/80`** → preregistered classification `spatial-exposure projection is insufficiently active`;
- all 80 predicted norms are nonzero; the missing 15 are exactly-zero source-reference gradients, so the failure is not numerical collapse of the predicted field;
- among the 65 nondegenerate anchors, positive-dot fraction is **`60/65 = 0.9230769`**;
- cosine mean / median / p10 / p90 = **`0.3367 / 0.3524 / 0.0301 / 0.6649`**;
- the median cosine is also below the fixed `0.40` directional-quality gate.

Therefore the **direct full 8×8 exposure-field projection route is closed**. The residue is a useful coarse descent sign, not enough transferred spatial shape for a full low-frequency field rollout.

### T060-B — exact T036 common-gain subspace comparison: positive first-order result

T060-B asked whether the frozen T059-E field is more faithful than the original deployed T014/T036 energy when both are projected through the **same exact 4-D Region2 common-gain Jacobian**.

The audit is scientifically admissible. The target-free audit set is selected only by already-frozen `any(gate.active)` metadata from the exact 80 T060-A source-outer state-0 anchors. `60/80` enter the audit and all 60 satisfy the unchanged nondegenerate criterion. The same fresh degraded-image `J_gain` feeds both frozen heads with their own accepted normalization/loading semantics. All `x/J_gain/q_E/q_014/g_E/g_014` tensors were frozen and hashed before the first source-clean read; only then did the separate source diagnostic compute true restoration gradients. No optimizer step, real-domain/LOL-v2/official-test access, PSNR/SSIM, or reference quantity entered prediction.

On the same 60 nondegenerate anchors:

- positive-dot: **T059-E `58/60 = 0.9667` vs T014 `53/60 = 0.8833`**;
- wrong-sign count: **T059-E `2` vs T014 `7`**;
- cosine mean: **`0.7775` vs `0.6290`**;
- cosine median: **`0.9064` vs `0.8009`**, delta `+0.1055`;
- cosine p10: **`+0.4259` vs `-0.0607`**.

T059-E passes every preregistered comparative gate, including both material-advantage alternatives (median cosine `>= +0.05` over T014 and at least two fewer wrong-sign anchors). Therefore the accepted classification is **`T059-E is a common-gain direction candidate for one later finite-step test`**.

Scientific interpretation: T059-E does **not** support a high-dimensional 8×8 exposure field, but in the already useful low-dimensional T036 common-gain action subspace it gives substantially better first-order restoration direction than the deployed T014 energy. This is the first direct evidence that the learned transferable field may improve the strongest current deployable action family rather than merely adding a new renderer. It is still source-only first-order evidence: no finite-step quality, real-domain safety, tail reduction, deployability, or final-benchmark improvement is established yet.

The next experiment is therefore one fixed head-swap validation: preserve the exact accepted T036 finite-step common-gain procedure and replace only T014 guidance with frozen T059-E guidance. No tuning is authorized.

## Strong baseline development anchors

- **Retinexformer T033-A:** `21.4787864 / 0.7900612`.
- **SNR-Aware T045-A:** `23.3963299 / 0.8237644`.

Both are target-free at inference under the frozen comparison protocol, but their released supervised checkpoints are exposed to LOL-v2 Real training data containing this development split. They are training-exposed development anchors, not independent held-out SOTA evidence.

The final sprint objective remains a clear **`+2–3 dB` PSNR advantage over the strongest fair target-free baseline on the same held-out protocol**, without violating the no-test-target rule. This is an objective, not a current claim.

## Information-boundary rules

- Test-time adaptation and checkpoint/state selection must never consume test labels, clean/normal-light targets, reference gradients/Jacobians, oracle values, PSNR/SSIM, degradation masks/gain maps, condition IDs, annotations, semantic image IDs, or per-image baseline outcome/harm labels.
- A degraded-image feature/action Jacobian such as `dx/dv`, `dx/du`, or `dx/d(raw_gain)`, computed entirely from the current degraded image/current target-free intermediate image and frozen model, is permissible; a Jacobian/gradient that depends on a clean/reference target is not.
- Validation/test outputs and decisions must be finalized and persisted before references or evaluation metrics are attached, except in explicitly isolated non-deployable source/reference diagnostics.
- Source-training clean/reference targets may be used only for source-supervised training or isolated source-domain diagnostics; they are never admissible test-time inputs.
- Reference diagnostics may motivate only global research choices; no per-image oracle quantity may enter deployable inference.
- External baselines admitted to the main comparison must be target-free at inference.
- Fresh/final benchmark sets must remain isolated from model/hyperparameter selection.
- **The official LOL-v2 Real test remains untouched through T060-B.**
- Fresh/test runs must fail closed on source/checkpoint/cohort/provenance mismatches.

## Best current methods / ceilings

- **Broad fresh-qualified Ours-Core:** T014 Sobolev Region2 TTT.
- **Best fixed-validation deployable candidate:** T026-A, `11.1208764 / 0.3737918`.
- **Fresh-qualified deployable action extension:** T036-A common gain, `+0.9392571 dB` mean on its fresh cohort, unsafe tail unresolved (`29/100` regressions; worst `-5.614 dB`).
- **Best promoted non-deployable mechanism state:** T055-A, `24.3497922 / 0.7591279`; T056/T057 are valid but failed materiality gates.
- **T059 matched-detail:** direction transfer is scientifically supported, but raw and T026-conditioned real-development gains are practically negligible; the detail integration line is closed.
- **T060-A spatial exposure projection:** coarse sign transfer is visible, but full-field coverage/shape alignment fails the fixed audit; no finite-step spatial-exposure rollout is authorized.
- **T060-B common-gain direction:** positive comparative first-order result; T059-E is markedly better aligned than T014 in the exact 4-D common-gain subspace, but no finite-step/deployable claim yet.

## Integration note

Historical PRs may contain inherited history/integration complications. Preserve accepted scientific/evidence states rather than repairing history inside experiment cycles. T059-P through T060-B are scientifically reviewable through their pinned source/evidence even while PRs remain open. T060-B is PR #126.

## Current open task

**T060-C — fixed T059-E-for-T014 common-gain finite-step swap on the accepted T036 development cohort** in `coordination/CHATGPT_TO_CODEX.md`.

Reuse the exact accepted T036-A 100-image development cohort and all T026/T036 initial states, operator, masks, optimizer, learning rates, step count, stopping semantics, and renderer settings. Change only the gradient guidance from original T014 energy to frozen T059-E, recomputing the current target-free feature/Jacobian/gradient online. Freeze/hash all 100 outputs before any normal/reference or per-image baseline outcome is read. The fixed test asks whether the new direction retains material T036 mean gain while materially reducing the `29/100` regression tail and `-5.614 dB` worst case. No tuning and no official-test access are authorized in this cycle.
