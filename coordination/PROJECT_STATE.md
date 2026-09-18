# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Enhancement**

This file is the live scientific state. Detailed earlier task-by-task history remains in Git history and `research_log/`.

## Core scientific question

Can a vision system adapt a compact spatial image-processing state per test image, without test labels or clean/normal-light targets, so that unknown and spatially heterogeneous degradation is corrected by a reference-free test-time optimization field?

## Current deployable state

**T014 Sobolev Region2 TTT remains the broad fresh-qualified Ours-Core.** It uses the frozen nuisance readout + clean-abstention gate, source-supervised Sobolev restoration energy, hard Region2 EV+gamma adaptation, projected label-free updates, and minimum predicted-energy checkpoint selection.

On LOL-v2 Real development data, the official 100-pair test remains sealed. The accepted fixed-validation deployable base remains **T026-A: `11.1208764 dB / 0.3737918 RGB-SSIM`**. T036-A common gain remains the strongest fresh-qualified deployable action expansion found so far, improving exact T026-A by `+0.9392571 dB / +0.0083560 RGB-SSIM` mean on its deterministic reference-unused development cohort, but with an unresolved unsafe tail.

No T059 matched-detail result is deployable. **No real-domain detail rollout is authorized.**

## Renderer / action-family diagnosis

- **T054-A local-detail field** is the strongest missing renderer capability identified: `24.3454867 / 0.7577572`, adding `+1.2383356 dB` mean / `+1.2353360 dB` median PSNR and `+0.1877170` mean SSIM over T052 on the reference-only mechanism cohort, with both metrics improving on all 100 images.
- **T055-A/T055-V** closes simple one-scale optimization-budget rescue: `24.3497922 / 0.7591279`; another fixed +1000 updates add only `+0.0043055 dB` mean.
- **T056-A** second-scale detail and **T057-A** chroma-detail extensions are scientifically valid but fail their frozen materiality gates; neither supersedes T054/T055 as the main mechanism.

These mechanism results are non-deployable `REFERENCE_ORACLE_ONLY` diagnostics. Clean targets, reference gradients/states, PSNR/SSIM, and per-image oracle quantities are forbidden from deployable inference.

## Matched-detail optimization-field diagnosis

- **T058-AF:** the frozen T014 energy is not ready for direct T054-detail integration. On all `7,346` canonical source states, local-detail alignment is `68.4567%` positive-dot / `0.230516` median cosine, below the fixed `>=75% / >=0.50` readiness gates.
- **T059-BF:** with the unchanged 28-D T014 `EnergyHead`, one fixed `1:1:1` value/legacy/detail recipe can fit all source rows in-sample: detail `98.7990% / 0.626633`, legacy `99.2826% / 0.966527`, value Huber `0.0722323 <= 0.0765085`. This establishes in-source capacity only.
- **T059-C2:** corrected unique-image holdout is negative. A single fixed fit on `64` source images / `5,886` rows evaluated `16` unseen source images / `1,460` rows. Detail positive-dot `0.845031` passes, detail median cosine `0.479706` fails, legacy `0.950694 / 0.938040` passes, and value Huber `0.193253` fails. Thus the fixed matched recipe does not establish image-held-out source generalization.
- **T059-D:** subtracting each held-out bank's deterministic state-0 anchor reduces value Huber from `0.1932531` to `0.1030146` but still fails. Additive bank offset is only part of the problem.
- **T059-E:** directly training bank-relative value still fails the fixed nested source-development split. On `48` inner-train images / `4,357` rows and `16` inner-held images / `1,529` rows, held relative Huber is `0.221076`, detail `0.868874 / 0.491553`, and legacy `0.957011 / 0.905764`; only detail median cosine and scalar value fail. Inner-train relative Huber is `0.056903`.
- **T059-F2:** after offset removal, a strict-positive bankwise scale decomposition still leaves held Huber `0.2147133`, rejecting positive scalar-scale miscalibration as a sufficient explanation. Genuine within-bank shape/order error remains in the tail.
- **T059-G:** fixed cross-image 1-NN in the standardized 28-D feature space separates detail support from scalar transfer. Train-LOO passes relative Huber/detail gates at `0.064456 / 0.946449 / 0.597402`. Inner-held detail still passes at `0.913245 / 0.528418`, but scalar Huber fails at `0.231723`. Thus the 28-D representation contains transferable local information for detail direction while scalar geometry does not transfer under fixed 1-NN.
- **T059-H:** reweighting train-LOO scalar error by the held nearest-distance distribution gives Huber `0.068549`, still below the gate while actual held remains `0.231723`. A modest marginal nearest-support-distance shift is therefore not sufficient to explain the scalar collapse.
- **T059-I:** fixed unweighted averaging over the nearest rows from five distinct training images is not a viable rescue. It worsens train-LOO scalar Huber from `0.064456` to `0.084737` and leaves held at `0.223014`.
- **T059-J:** the source-target oracle over those same frozen five donors is also negative on inner-held images. Train-LOO oracle-floor Huber is `0.0539017` (passes), but inner-held oracle-floor Huber is `0.2128655` (fails). Thus the frozen local five-donor neighborhood itself usually lacks an accurate held scalar value.
- **T059-K:** the source-target oracle over the **entire fixed 48-image / 4,357-row inner-train pool** is strongly positive: train-LOO global-oracle Huber `0.000322648`, inner-held `0.000511841`, both far below the unchanged `0.0765085` gate. Global scalar-range coverage is `0.997705 / 0.996730`. However, the oracle donor is in T059-I's frozen five-donor neighborhood only `0.006885 / 0.007848` of the time, and its median rank under the existing absolute standardized-28D Euclidean geometry is `1972 / 1953` (median percentile `0.4616 / 0.4481`). Therefore the scalar **values** are present in the fixed source pool, but the current absolute 28-D geometry does not localize the appropriate value for held images.
- **T059-L:** the obvious bank-relative feature-coordinate fix is negative. Replacing absolute `x` by `dx = x - x_state0` under the same fixed cross-image 1-NN worsens train-LOO scalar Huber from `0.064456` to `0.099402`, already failing the `0.0765085` source-control gate; inner-held is `0.232549`, effectively unchanged from the absolute-feature `0.231723`. Thus simple state-0 feature centering is not a viable scalar-localization mechanism.
- **T059-M:** removing the legacy/detail Sobolev terms does **not** rescue scalar transfer. Scalar-only relative Huber falls to `0.00525188` on the 48-image / 4,357-row inner-train set but remains `0.2274732` on the 16-image / 1,529-row inner-held set. Relative to T059-E, train improves by `-0.0516511` while held slightly worsens by `+0.00639746`. Harmful joint-objective interference is therefore not supported as the cause of the scalar failure; the unchanged 28-D head has ample in-sample capacity but unresolved cross-image scalar generalization.

**Current scientific interpretation after T059-M:** the scalar marginal values exist in the fixed source pool (T059-K), but target-free scalar conditioning remains unresolved. T059-G/J show that the current absolute local neighborhood does not contain the right held scalar support, T059-H shows marginal distance shift is insufficient, T059-I rejects simple multi-image averaging, T059-L rejects the obvious bank-relative displacement coordinate, and T059-M rejects harmful legacy/detail multi-task interference as the primary explanation. The very low scalar-only train loss together with high held loss raises a remaining, narrower question: whether a strictly source-only image-held checkpoint selector can control late-epoch overfitting under the same representation/head before any representation or architecture redesign is justified.

The matched-detail line therefore remains **non-deployable**. The renderer/action family has strong detail headroom, and detail-direction information itself transfers surprisingly well, but the target-free optimization field has not yet demonstrated sufficiently reliable unseen-image scalar geometry for the detail-expanded state space.

## Strong baseline development anchors

- **Retinexformer T033-A:** `21.4787864 / 0.7900612`.
- **SNR-Aware T045-A:** `23.3963299 / 0.8237644`.

Both are target-free at inference under the frozen comparison protocol, but their released supervised checkpoints are exposed to LOL-v2 Real training data containing this development split. They are training-exposed development anchors, not independent held-out SOTA evidence.

The final sprint objective remains a clear **`+2–3 dB` PSNR advantage over the strongest fair target-free baseline on the same held-out protocol**, without violating the no-test-target rule. This is an objective, not a current claim.

## Information-boundary rules

- Test-time adaptation and checkpoint selection must never consume test labels, clean/normal-light targets, reference gradients/Jacobians, oracle values, PSNR/SSIM, degradation masks/gain maps, condition IDs, annotations, or semantic image IDs.
- Validation/test outputs and decisions must be finalized and persisted before references or evaluation metrics are attached, except in explicitly isolated non-deployable reference diagnostics.
- Reference diagnostics may motivate only global research choices; no per-image oracle quantity may enter deployable inference.
- Source-training clean/reference targets may be used only for source-supervised training or isolated source-domain diagnostics; they are never admissible test-time inputs.
- External baselines admitted to the main comparison must be target-free at inference.
- Fresh/final benchmark sets must remain isolated from model/hyperparameter selection. **The official LOL-v2 Real test remains untouched through T059-M.**
- Fresh/test runs must fail closed on source/checkpoint/cohort/provenance mismatches.

## Best current methods / ceilings

- **Broad fresh-qualified Ours-Core:** T014 Sobolev Region2 TTT.
- **Best fixed-validation deployable candidate:** T026-A, `11.1208764 / 0.3737918`.
- **Fresh-qualified deployable action extension:** T036-A common gain, `+0.9392571 dB` mean on its fresh cohort, unsafe tail unresolved.
- **Best promoted non-deployable mechanism state:** T055-A, `24.3497922 / 0.7591279`; T056/T057 are valid but failed materiality gates.
- **Matched-detail candidate:** T059-BF through T059-M. In-source capacity exists and detail-direction information transfers nonparametrically, but unseen-image scalar geometry repeatedly fails. Offset, positive-scale, marginal distance shift, five-image smoothing, five-donor oracle, simple state-0 feature centering, and harmful joint-objective interference have been rejected as sufficient explanations/rescues; T059-K shows scalar marginal support exists globally. The next unresolved mechanism is whether source-only checkpoint selection can control scalar-head overfitting under the current representation. This line remains non-deployable.
- **Training-exposed development anchors:** Retinexformer `21.4787864 / 0.7900612`; SNR-Aware `23.3963299 / 0.8237644`.

## Integration note

Historical PRs may contain inherited history/integration complications. Preserve accepted scientific/evidence states rather than repairing history inside experiment cycles. T059-M is PR #112 and is scientifically reviewable through its pinned source/evidence commits under that rule.

## Current open task

**T059-N — nested source-only early-stopping control** in `coordination/CHATGPT_TO_CODEX.md`.

Using only the 48-image T059-E inner-train parent cohort, deterministically split 40 fit / 8 source-selector images by sorted `image_group_index` rank `% 6 == 5`. Run exactly one scalar-only 100-epoch fit with fit-only normalization and persist/hash all 100 checkpoints before opening selector scalar targets. Select the best eligible frozen epoch using only the 8-image source selector and the unchanged scalar gate. Do not open the T059-E inner-held cohort, C2 outer cohort, target-domain data, LOL-v2, or official test in this cycle; no representation/architecture/loss search or real-domain rollout is allowed.
