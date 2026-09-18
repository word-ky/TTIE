# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Enhancement**

This file is the live scientific state. Detailed task-by-task history remains in Git history and `research_log/`.

## Core scientific question

Can a vision system adapt a compact spatial image-processing state per test image, without test labels or clean/normal-light targets, so that unknown and spatially heterogeneous degradation is corrected by a reference-free test-time optimization field?

## Current deployable state

**T014 Sobolev Region2 TTT remains the broad fresh-qualified Ours-Core.** It uses the frozen nuisance readout + clean-abstention gate, source-supervised Sobolev restoration energy, hard Region2 EV+gamma adaptation, projected label-free updates, and minimum predicted-energy checkpoint selection.

On LOL-v2 Real development data, the accepted fixed-validation deployable base remains **T026-A: `11.1208764 dB / 0.3737918 RGB-SSIM`**. T036-A common gain remains the strongest fresh-qualified deployable action expansion found so far, improving exact T026-A by `+0.9392571 dB / +0.0083560 RGB-SSIM` mean on its deterministic reference-unused development cohort, but with an unresolved unsafe tail.

No T059 matched-detail result is deployable. The official LOL-v2 Real test remains sealed.

## Renderer / action-family diagnosis

- **T054-A local-detail field** is the strongest missing renderer capability identified in reference-only mechanism studies: `24.3454867 / 0.7577572`, adding `+1.2383356 dB` mean / `+1.2353360 dB` median PSNR and `+0.1877170` mean SSIM over T052 on that oracle/reference cohort.
- **T055-A/T055-V** show that merely extending the same one-scale optimization budget adds only about `+0.0043 dB` beyond T054-A.
- **T056-A** second-scale detail and **T057-A** chroma-detail are valid but failed their frozen materiality gates.

These renderer studies are non-deployable `REFERENCE_ORACLE_ONLY` diagnostics. Their clean targets, reference gradients/states, PSNR/SSIM, and per-image oracle quantities are forbidden from test-time inference.

## Matched-detail optimization-field diagnosis

The matched-detail program has established a stable asymmetry: **detail-direction information transfers much better than calibrated scalar energy/value, but the field is not safety-calibrated and its first real-domain raw-input gain is extremely small.**

Accepted negative/closed conclusions from T059-C2 through T059-U remain unchanged:

- calibrated unseen-image scalar prediction is unsupported despite strong in-sample capacity;
- simple 28-D locality, bank centering/context, early stopping, raw frozen-CLIP Euclidean locality, and local-donor smoothing/oracles do not solve scalar localization;
- dual-head argmin/full-curve disagreement does not expose the catastrophic shared-bias tail reliably;
- optimizer-scale/curvature is not the explanation for the harmful one-step source cases;
- low predicted-gradient norm does not transfer as a useful target-free abstention rule on the independent source outer cohort.

At the same time, the detail field is consistently more promising than scalar value prediction:

- T059-E source inner-held detail alignment is much healthier than value transfer (`0.868874` positive-dot / `0.491553` median cosine).
- T059-S/T show that most source anchors have the correct local descent sign, although the minority harmful cases already have the wrong first-order sign.
- T059-U shows that the **ungated** fixed one-step action is aggregate-favorable on the independent source outer cohort: `55 improve / 5 harm / 20 tie`, mean absolute MSE change `-3.7988555e-5`, median `-2.2112635e-5`, maximum harm `7.3056247e-6`.
- T059-V/W establish that the degraded-image feature/action Jacobian and one-step detail action can be reconstructed fully online from the current degraded image and frozen model, including 16/16 active nonzero-gradient source replay cases, without cached reference-dependent quantities.

### T059-X — first real-domain transfer result

T059-X is accepted as **validation-only directional mechanism evidence** on the exact already-open 100-image LOL-v2 Real development cohort, not as deployment or benchmark evidence.

The fixed image-only T059-W path was executed from each literal raw low image with exactly one Adam detail step (`lr=0.05`). All 100 outputs and decisions were frozen before any development normal was opened. The original evaluator then stopped because it incorrectly asserted bitwise equality between literal raw and the inherited zero-state ISP output; the difference was only float32 roundoff (`<=5.960464477539063e-8`). An evaluation-only repair preserved the already-frozen inference outputs and evaluated the literal raw baseline. Independent replay verifies the ISP chain, Adam/detail render, metrics, counts, and classification. No inference/action/output rerun or tuning occurred.

Frozen T059-X results:

- active actions: **`100/100`**;
- strict full-RGB MSE improve/harm/tie: **`97 / 3 / 0`**;
- mean MSE change: **`-2.7407772160e-5`**;
- median MSE change: **`-1.4010135270e-5`**;
- mean RGB-SSIM change: **`+0.0005802802`**;
- mean paired PSNR gain: **`+0.001135428 dB`**;
- maximum MSE harm: **`5.5341140e-6`**.

T059-X therefore passes its preregistered aggregate-transfer gates, but the effect magnitude is **practically negligible**. The correct interpretation is not that matched-detail is now a deployable enhancer; it is that the source-trained detail direction survives the source→real development shift often enough to produce a tiny consistent gain.

### Current scientific interpretation after T059-X

The scalar/value branch remains closed as a practical route under the tested formulations. The detail branch remains alive, but the evidence now separates **directional validity** from **useful effect size**.

A local high-frequency/detail renderer applied directly to severely underexposed raw images cannot repair the dominant low-frequency/global illumination error, so the tiny `+0.0011 dB` raw-input gain is not surprising. The scientifically relevant remaining question is whether the same frozen detail field is **complementary to the already accepted target-free T026-A global correction**, where the image is closer to the regime in which local detail processing can matter.

The matched-detail line remains **non-deployable**. No official-test matched-detail action, multi-step rollout, target-domain fitting, or per-image metric/oracle selection is authorized.

## Strong baseline development anchors

- **Retinexformer T033-A:** `21.4787864 / 0.7900612`.
- **SNR-Aware T045-A:** `23.3963299 / 0.8237644`.

Both are target-free at inference under the frozen comparison protocol, but their released supervised checkpoints are exposed to LOL-v2 Real training data containing this development split. They are training-exposed development anchors, not independent held-out SOTA evidence.

The final sprint objective remains a clear **`+2–3 dB` PSNR advantage over the strongest fair target-free baseline on the same held-out protocol**, without violating the no-test-target rule. This is an objective, not a current claim.

## Information-boundary rules

- Test-time adaptation and checkpoint/state selection must never consume test labels, clean/normal-light targets, reference gradients/Jacobians, oracle values, PSNR/SSIM, degradation masks/gain maps, condition IDs, annotations, or semantic image IDs.
- A degraded-image feature/action Jacobian such as `dx/dv`, computed entirely from the current degraded image/current target-free intermediate image and frozen model, is permissible; a Jacobian/gradient that depends on a clean/reference target is not.
- Validation/test outputs and decisions must be finalized and persisted before references or evaluation metrics are attached, except in explicitly isolated non-deployable source/reference diagnostics.
- Reference diagnostics may motivate only global research choices; no per-image oracle quantity may enter deployable inference.
- Source-training clean/reference targets may be used only for source-supervised training or isolated source-domain diagnostics; they are never admissible test-time inputs.
- External baselines admitted to the main comparison must be target-free at inference.
- Fresh/final benchmark sets must remain isolated from model/hyperparameter selection.
- **The official LOL-v2 Real test remains untouched through T059-X.**
- Fresh/test runs must fail closed on source/checkpoint/cohort/provenance mismatches.

## Best current methods / ceilings

- **Broad fresh-qualified Ours-Core:** T014 Sobolev Region2 TTT.
- **Best fixed-validation deployable candidate:** T026-A, `11.1208764 / 0.3737918`.
- **Fresh-qualified deployable action extension:** T036-A common gain, `+0.9392571 dB` mean on its fresh cohort, unsafe tail unresolved.
- **Best promoted non-deployable mechanism state:** T055-A, `24.3497922 / 0.7591279`; T056/T057 are valid but failed materiality gates.
- **Matched-detail candidate:** calibrated scalar localization and the tested safety gates are unsupported. The detail direction is source-validated, online-reproducible, and now shows a tiny positive raw-input real-development transfer (`+0.001135428 dB`, `97/100` MSE improvements), but this is not material or deployable.

## Integration note

Historical PRs may contain inherited history/integration complications. Preserve accepted scientific/evidence states rather than repairing history inside experiment cycles. T059-P through T059-X are scientifically reviewable through their pinned source/evidence even while PRs remain open. T059-X is PR #123.

## Current open task

**T059-Y — fixed one-step matched-detail integration on frozen T026-A outputs** in `coordination/CHATGPT_TO_CODEX.md`.

Use the exact accepted target-free T026-A selected outputs/states on the same already-open 100-image development cohort as immutable bases. Apply exactly one frozen T059-E/T054 detail step while keeping the T026-A global state fixed. Freeze/hash all 100 integrated outputs before any reference read, then evaluate only against the already-open development normals. The preregistered materiality gates are mean PSNR gain `>= +0.05 dB`, median PSNR gain `>0`, at least `60/100` strict MSE improvements, and mean RGB-SSIM gain `>= +0.001`. No tuning, second step, T036 integration, official-test access, or clean/reference information in inference is authorized.
