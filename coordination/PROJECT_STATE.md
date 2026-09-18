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
- **T059-J:** the source-target oracle over those same frozen five donors is also negative on inner-held images. Train-LOO oracle-floor Huber is `0.0539017` (passes), but inner-held oracle-floor Huber is `0.2128655` (fails by `0.1363570`). Oracle-best donor rank-1 fractions are `0.315814 / 0.270765`, and target-in-local-donor-range coverage is `0.723663 / 0.710922`. All `5,886` oracle decisions / `29,430` donor values were independently replayed, the donor table was SHA-frozen before held targets opened, the C2 outer holdout stayed sealed, and all training/model/target/test leakage counters were zero.

**Current scientific interpretation after T059-J:** the failure is no longer plausibly explained by a bad target-free selector operating inside an otherwise adequate five-donor neighborhood. For many inner-held rows, the frozen local 28-D neighborhood itself does not contain an accurate scalar value. This does **not** yet distinguish whether the required scalar values are absent from the entire fixed 48-image inner-train pool or merely not localized by the current feature geometry. That global-support-vs-localization distinction is the next justified question before any feature redesign or new training.

The matched-detail line therefore remains **non-deployable**. The renderer/action family has strong detail headroom, but the target-free optimization field has not yet demonstrated sufficiently reliable unseen-image scalar geometry for the detail-expanded state space.

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
- Fresh/final benchmark sets must remain isolated from model/hyperparameter selection. **The official LOL-v2 Real test remains untouched through T059-J.**
- Fresh/test runs must fail closed on source/checkpoint/cohort/provenance mismatches.

## Best current methods / ceilings

- **Broad fresh-qualified Ours-Core:** T014 Sobolev Region2 TTT.
- **Best fixed-validation deployable candidate:** T026-A, `11.1208764 / 0.3737918`.
- **Fresh-qualified deployable action extension:** T036-A common gain, `+0.9392571 dB` mean on its fresh cohort, unsafe tail unresolved.
- **Best promoted non-deployable mechanism state:** T055-A, `24.3497922 / 0.7591279`; T056/T057 are valid but failed materiality gates.
- **Matched-detail candidate:** T059-BF through T059-J. In-source capacity exists and detail-direction information transfers nonparametrically, but unseen-image scalar geometry repeatedly fails; simple offset, positive-scale, nearest-distance, five-image smoothing, and five-donor oracle explanations have all been rejected. This line remains non-deployable.
- **Training-exposed development anchors:** Retinexformer `21.4787864 / 0.7900612`; SNR-Aware `23.3963299 / 0.8237644`.

## Integration note

Historical PRs may contain inherited history/integration complications. Preserve accepted scientific/evidence states rather than repairing history inside experiment cycles. T059-J is PR #109 and is scientifically reviewable under that rule.

## Current open task

**T059-K — frozen global inner-train scalar-support oracle floor** in `coordination/CHATGPT_TO_CODEX.md`.

Using exactly the accepted inner split, first freeze/hash the complete `4,357`-row inner-train scalar/feature candidate table before any inner-held scalar target is opened. Then run a source-only oracle that chooses the minimum-Huber scalar candidate globally: train-LOO candidates must exclude the query image; inner-held candidates may use all 48 training images. The unchanged `0.07650849781930447` scalar gate determines whether the scalar values exist somewhere in the fixed training pool but are not localized by the current 28-D neighborhood, or whether the held scalar targets are inadequately supported even globally. This is `REFERENCE_ORACLE_ONLY`; it must never become test-time logic. C2 outer supervision, target-domain data, LOL-v2, and official test remain inaccessible.
