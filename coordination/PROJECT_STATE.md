# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Enhancement**

## Core scientific question

Can a vision system adapt a compact spatial image-processing state per test image, without test labels or clean/normal-light targets, so that unknown and spatially heterogeneous degradation is corrected by a reference-free test-time optimization field?

## Current deployable state

**T014 Sobolev Region2 TTT remains the broad fresh-qualified Ours-Core.** It uses the frozen nuisance readout + clean-abstention gate, source-supervised Sobolev restoration energy, hard Region2 EV+gamma adaptation, projected label-free updates, and minimum predicted-energy checkpoint selection.

On LOL-v2 Real development data, the official 100-pair test remains sealed. The accepted fixed-validation deployable base is **T026-A: `11.1208764 dB / 0.3737918 RGB-SSIM`**. T026-B showed that extending the same trajectory from 40 to 80 steps adds only `+0.1210491 dB / +0.0042526`, so the 40-step base remains fixed.

**T036-A common gain is the strongest fresh-qualified deployable action expansion so far.** On its deterministic reference-unused 100-pair training-development cohort it improves exact T026-A by `+0.9392571 dB / +0.0083560 RGB-SSIM` mean, but remains unsafe per image. It is a mechanism result, not the final deployable Ours.

## Optimization-field diagnosis

- **T028-A:** exact EV+gamma reference-only reachability `17.4599918 / 0.4317158`, proving substantial headroom beyond the learned path.
- **T029-A:** learned-field direction is strong early and collapses late: step-10 median cosine `0.7266` with `97%` positive-dot versus selected-state median cosine `-0.2785` with `24%` positive-dot.
- **T031/T032:** source-support distance predicts invalidity moderately well, but the direct global trust-radius controller fails.
- **T039-A:** on all `7,346` accepted canonical T014 source states and gains `{0.75,1.00,1.25}`, legacy alignment is `93.3545%` positive-dot / `0.955998` median cosine and later common-gain alignment is `85.2274% / 0.802533`; the frozen energy can therefore generalize to at least one later action tangent on source.
- **T041/T042:** late real legacy/feature-state extrapolation is a major local reliability failure; substituting fixed step-10 legacy coordinates restores strong directional alignment.
- **T043/T044:** directional validity does not equal absolute image quality, and raw late excursion is not a useful target-free risk score.
- **T058-AF:** the exact T054 local-detail tangent fails the preregistered frozen-energy source-readiness gate on all `7,346` canonical T014 source states: `4,959/7,244 = 68.4567%` positive-dot and `0.230516` median cosine, versus required `>=75%` and `>=0.50`. The result is a source-only diagnostic with immutable learned gradients frozen before clean-source reference gradients were opened. The frozen T014 field is therefore **not ready** for direct T054-detail integration. Descriptively, homogeneous-dark states are especially weak (`51.6782%`, median cosine `0.0209763`), while homogeneous-bright is stronger (`82.7819%`, `0.456648`); these subgroups do not alter the global verdict.
- **T059-BF:** matched dual-tangent source fitting is feasible with the unchanged 28-D T014 `EnergyHead` and one fixed `1:1:1` value/legacy/detail recipe. After exactly one seed-7 CPU 100-epoch fit on all `7,346` canonical source states, detail alignment reaches `98.7990%` positive-dot / `0.626633` median cosine while legacy remains `99.2826% / 0.966527`; value Huber is `0.0722323`, inside the preregistered `0.0765085` ceiling. This resolves the **in-source capacity** question positively but does not establish cross-image generalization or deployability because the same source-image banks were used for fitting and evaluation.

The deployable bottleneck remains two-sided: the renderer/action family must be expressive enough, and the learned target-free optimization field must remain valid over the states/directions it visits. T054–T057 establish that local detail/noise control is the strongest missing renderer capability found so far. T058-AF shows the frozen T014 field does not transfer strongly enough to that 64-D detail tangent, while T059-BF shows that matched source-side Sobolev supervision can fit both the legacy and detail tangents without changing the 28-D head architecture. **Image-held-out source generalization is now the required gate before any real-domain detail rollout.**

## Capacity sequence through T057

All results here are **non-deployable `REFERENCE_ORACLE_ONLY` diagnostics** on the frozen development cohort. Clean targets, reference gradients/states, PSNR/SSIM, and per-image oracle quantities are forbidden from deployable inference.

- **T046-A common-gain continuation:** `19.5732278 / 0.4093671`; +1000 more updates add only `+0.0202 dB`, rejecting material underconvergence.
- **T047-A additive lift with gain frozen:** `20.0559720 / 0.4142735`; broad-positive but sub-gate.
- **T048-A joint regional affine coupling:** `21.0649800 / 0.4566638`; joint gain+lift adds `+1.0090 dB` mean / `+0.5584 dB` median over T047.
- **T049-A regional monotonic tone LUT:** `21.8722552 / 0.5047579`; adds `+0.8073 dB` mean / `+0.5989 dB` median and `+0.0481` mean SSIM.
- **T050-A tone convergence extension:** `21.8742627 / 0.5051607`; another +1000 tone-only updates add only `+0.0020 dB`, closing pure tone budget rescue.
- **T051-A smooth RGB-shared 8×8 exposure field:** `22.5332679 / 0.5219925`; broad-positive but below the frozen SOTA-scale gate.
- **T052-A joint smooth exposure + additive field:** `23.1071511 / 0.5700402`; relative to T050, `+1.2328884 dB` mean / `+1.0713880 dB` median and `+0.0648795` mean SSIM. It misses only the frozen `+1.50 dB` mean requirement.
- **T053-A additive-range closure:** `23.1137228 / 0.5693885`; widening only `b` gives `+0.00657 dB` mean and slightly worse SSIM, closing additive-range rescue.
- **T054-A local-detail field:** `24.3454867 / 0.7577572`. A single RGB-shared 8×8 coefficient on `D=y0-B5(y0)` adds **`+1.2383356 dB` mean / `+1.2353360 dB` median PSNR and `+0.1877170` mean SSIM** over T052, with both metrics improving on 100/100 images. The selected field is strongly negative, establishing local detail attenuation / denoising as a major missing capability.
- **T055-A/T055-V one-scale convergence closure:** `24.3497922 / 0.7591279`. Another fixed +1000 updates add only **`+0.0043055 dB` mean / `+0.0029189 dB` median and `+0.0013707` SSIM**. Independent replay is accepted after verifier-only FMA-order adjudication; pure one-scale step/LR rescue is closed.
- **T056-A fixed second-scale detail band:** scientifically valid but **not materially supported**. Adding one RGB-shared 8×8 coefficient on `D2=B5(y0)-B9(y0)` reaches **`24.4701676 / 0.7698747`**, only `+0.1203754 dB` mean / `+0.1019722 dB` median PSNR and `+0.0107468` mean SSIM over T055, below the frozen `+0.50/+0.25 dB/+0.020` gate. Gains are broad but too small to promote coarse-band stacking as the next major mechanism.
- **T057-A fixed chroma-detail marginal oracle:** scientifically valid but **not materially supported**. Starting exact T055 and adding only one zero-RGB-mean chroma-detail 8×8 field reaches **`24.4353944 / 0.7673667`**, for only `+0.0856022 dB` mean / `+0.0472394 dB` median PSNR and `+0.0082388` mean SSIM versus T055, far below the same `+0.50/+0.25 dB/+0.020` gate despite `100/100` PSNR and SSIM wins. Only `5/100` selected states are at step 500. Independent replay passes 100 images / 50,100 history states / 200 metrics / 724 scalar checks; basis/interpolation/renderer errors remain below `1e-6`, all old coordinates and inactive pixels are exact, and official test access is zero.

The capacity diagnosis is now sharper. T054 establishes **local detail/noise control** as a high-value missing capability, while T055 closes simple one-scale budget rescue and T056/T057 show only small broad-positive returns from a coarser frequency band or chroma/luminance detail decoupling. T058-AF then shows that the current frozen T014 energy does not supply a sufficiently aligned gradient for the T054 direction. T059-BF demonstrates that this incompatibility is trainable rather than an architectural impossibility: the same 28-D head can fit legacy and detail source tangents jointly under the fixed matched Sobolev loss. The unresolved question is whether that fit transfers across unseen source images; renderer micro-extension and real-domain rollout remain paused until that gate is answered.

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
- Fresh/final benchmark sets must remain isolated from model/hyperparameter selection. **The official LOL-v2 Real test remains untouched through T059-BF.**
- Fresh/test runs must fail closed on source/checkpoint/cohort/provenance mismatches.

## Best current methods / ceilings

- **Broad fresh-qualified Ours-Core:** T014 Sobolev Region2 TTT.
- **Best fixed-validation deployable candidate:** T026-A, `11.1208764 / 0.3737918`.
- **Fresh-qualified deployable action extension:** T036-A common gain, `+0.9392571 dB` mean on its fresh cohort, unsafe tail unresolved.
- **Matched-detail source-fit candidate:** T059-BF dual-tangent head; all five in-source gates pass, but it is **not deployable** until image-held-out source generalization is established.
- **Best promoted non-deployable mechanism state:** T055-A, `24.3497922 / 0.7591279`; T056 (`24.4701676 / 0.7698747`) and T057 (`24.4353944 / 0.7673667`) are valid higher finite-budget observations but both failed their materiality gates and are not promoted as mechanisms.
- **Training-exposed development anchors:** Retinexformer `21.4787864 / 0.7900612`; SNR-Aware `23.3963299 / 0.8237644`.
- **Heterogeneous-only adaptive geometry extension:** T019; universal geometry repair remains paused after T020.

## Integration note

PR #75/T050 is merged as `276c0b1fa5c9e6548bef90048ec6fcc43da439c3`. PRs #76–#83 inherit evidence-history/integration complications; preserve exact accepted scientific/evidence states rather than rewriting history during experiment cycles. PR #83/T057 may remain non-mergeable for history reasons; that does not invalidate its reviewed scientific evidence. PR #95/T058-AF and PR #99/T059-BF are likewise scientifically reviewable despite inherited non-mergeable history; do not repair history inside experiment cycles.

## Current open task

**T059-C — fixed image-held-out source generalization** in `coordination/CHATGPT_TO_CODEX.md`.

Use the immutable 80-bank source manifest with the preregistered manifest-order split `bank_index % 5 == 0` as a 16-image holdout and the other 64 banks for one fresh fixed dual-tangent fit. Hold-out source targets/reference gradients must not be read before the epoch-100 checkpoint is persisted. Then evaluate only the held-out banks with the same five source-readiness gates. This is still a source-only generalization test; it does not authorize target-domain TTT, LOL-v2, official-test access, PSNR/SSIM selection, or any use of clean/reference information at test time.
