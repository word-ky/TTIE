# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Enhancement**

This file is the live scientific state. Detailed task history remains in Git history and `research_log/`.

## Core scientific question

Can a vision system adapt a compact spatial image-processing state per test image, without test labels or clean/normal-light targets, so that unknown and spatially heterogeneous degradation is corrected by a reference-free test-time optimization trajectory with reliable target-free stopping?

## Frozen Final Ours

T070-A froze the unchanged T067-B candidate as Final Ours. Immutable manifest SHA256:

`e7f129d931d27531e5f6a14cd72e8c94734c474c40c3c403959cc8f5764e5ab9`

Scientific source:

`aa4d920dff4b5b76751c24266e95ac9696d55d90`

The frozen method is:

- degraded-image-only scientific input;
- CommonRegion2/CommonBox 12-D EV/gamma/gain renderer from identity;
- 27 float32 Adam updates, `lr=0.03`;
- fixed low-only objective `L_spa + 10 L_exp + 5 L_col`;
- frozen T066-A 19-D safety model, probability threshold `0.5`, defining first-safe entry;
- normalized-progress constant `rho=0.9857470621423519`;
- T067-B safety–utility interpolation `lambda=0.875` with frozen endpoint/tie conventions.

No clean/reference target, test label, PSNR/SSIM, baseline outcome, oracle range, degradation annotation or condition ID enters test-time adaptation or checkpoint selection.

## Official LOL-v2 Real held-out result — T071-A

T071-A is accepted as `OFFICIAL_LOLV2_REAL_TEST_RESULT_FROZEN` on the complete official 100-image LOL-v2 Real test split:

- Final Ours mean PSNR: **`18.5322668614 dB`**
- median PSNR: `18.1927883320 dB`
- mean RGB-SSIM: **`0.5734772618`**
- selected step min / median / max: `17 / 22 / 24`
- mean synchronized inference time: `0.7003712362 s/image`

The complete target-free output/decision table was frozen before any clean-reference payload was opened, with inference-stage `reference_reads=0`. The official LOL-v2 Real test is now evaluation-frozen and forbidden for tuning.

## Matched official LOL-v2 Real baselines — T071-B

T071-B is accepted as `OFFICIAL_LOLV2_REAL_BASELINES_FROZEN`. On the exact same complete 100-image test split and exact T071-A metric implementation:

| Method | Mean PSNR | Median PSNR | Mean RGB-SSIM |
|---|---:|---:|---:|
| Frozen Final Ours | `18.5322668614` | `18.1927883320` | `0.5734772618` |
| Retinexformer | `22.7952619433` | `22.9109248988` | `0.8390188290` |
| SNR-Aware (`ttie_native_pad16`) | `21.3659020740` | `21.5756290394` | `0.8495393436` |

Paired Ours-minus-baseline PSNR gaps:

- vs Retinexformer: mean **`-4.2629950819 dB`**, median `-5.0813503531 dB`;
- vs SNR-Aware: mean **`-2.8336352125 dB`**, median `-1.7602761007 dB`.

Both baselines reused exact previously accepted source/checkpoint/config bindings, ran once on all 100 low images, froze outputs before reference evaluation, and had inference-stage `reference_reads=0`. Independent verification reproduces cohort identity, output hashes, metrics and paired deltas to machine precision.

### Scientific interpretation

The current Final Ours is **not competitive with target-domain supervised LOL-v2 Real checkpoints in-domain** on PSNR/SSIM. No in-domain superiority claim is supported. This result is nevertheless not the decisive test of the unknown-degradation motivation because Retinexformer and SNR-Aware use paired supervised LOL-v2 Real source checkpoints, while Ours can adapt per target image using only the degraded target image.

The next decisive comparison is therefore source-frozen domain transfer: keep Final Ours and the exact LOL-v2-trained baseline checkpoints unchanged, move all methods to a different target dataset, allow only Ours' already-frozen degraded-image-only per-image TTT, and prohibit all target-specific retraining/tuning.

Do not use the official-test gap to tune Final Ours.

## Historical mechanism conclusions retained

- T062/T063 established a strong zero-reference trajectory, roughly `+3.5` to `+3.9 dB` mean PSNR over T036 replicates across independent internal cohorts.
- T063-A/T064-A showed historical severe failures were selection-limited rather than trajectory-capacity-limited.
- T066-A dynamics are useful primarily as a lower safety-entry signal, not a reliable late-stage catastrophic-quality monitor.
- T067-B's frozen `lambda=0.875` interpolation materially improved safety while retaining utility.
- Predeclared late-tail diagnostics were closed as insufficient: absolute-step cap, cumulative objective-motion knee, tail-local motion/objective inefficiency, aggregate component-value regret, final projection pressure and symmetric component-gradient cancellation.
- Continuing to fit development/exposed/official-test failures is post-hoc overfitting; method design remains closed until the fair cross-dataset program is completed and reviewed.

## Fair-comparison program

1. Frozen Final Ours on complete official LOL-v2 Real test — **completed, T071-A**.
2. Matched Retinexformer/SNR-Aware on the same official split — **completed, T071-B**.
3. Source-frozen cross-dataset/domain-shift evaluation — **current priority**.
   - LSRW complete canonical paired test split — **authorized as T072-A, not yet evaluated**.
   - UHD-LL complete held-out split — remains sealed for a later research-lead cycle.
4. Only after these fair tables exist may the research lead decide whether another Ours-development phase is justified.

Development baseline anchors and exposed-transfer numbers remain mechanism/diagnostic evidence only and must not be used as final Ours-vs-baseline gaps.

## Cross-dataset protocol

For LSRW and later UHD-LL:

- Final Ours stays exactly at the T070-A immutable artifact.
- Retinexformer and SNR-Aware stay at the exact T071-B accepted **LOL-v2 Real source** checkpoints/configs; no target-domain checkpoint substitution.
- No target-specific retraining, fine-tuning, calibration, threshold selection or hyperparameter tuning is allowed.
- Ours may only perform its already-frozen per-image degraded-image-only test-time adaptation.
- All method outputs must be frozen and hashed before any target clean/reference payload is read.
- References are metrics-only after output freeze.
- Complete canonical target test splits are required; no convenient subset or outcome-driven exclusion.

This source-frozen protocol directly tests the unknown-degradation/domain-shift motivation rather than target-domain supervised specialization.

## Information-boundary rules

- Test-time adaptation and checkpoint/state selection must never consume test labels, clean/normal-light targets, reference gradients/Jacobians, oracle values, PSNR/SSIM, degradation masks/gain maps, condition IDs, annotations, semantic image IDs, or per-image baseline outcome/harm labels.
- Quantities computed only from the degraded image/current target-free intermediate image, frozen state and frozen global/development-only assets are permissible.
- Held-out clean/reference targets may be read only after the corresponding complete output/decision tables for all compared methods are irrevocably frozen and hashed.
- Held-out results cannot authorize method tuning, threshold changes, sample exclusion or altered-setting reruns.
- Runs fail closed on source/checkpoint/cohort/provenance/geometry mismatch.

## Current open task

**T072-A — complete LSRW source-frozen cross-domain evaluation** in `coordination/CHATGPT_TO_CODEX.md`.

Resolve the complete canonical paired LSRW test split; keep Final Ours and the exact T071-B LOL-v2-trained Retinexformer/SNR-Aware artifacts unchanged; run all three on all target lows; freeze all outputs before reference access; then compute matched PSNR/RGB-SSIM and paired gaps. No target-specific training/tuning and no UHD-LL access in this cycle.