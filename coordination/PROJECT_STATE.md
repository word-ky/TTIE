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

T071-A is accepted as `OFFICIAL_LOLV2_REAL_TEST_RESULT_FROZEN`.

The exact frozen Final Ours was run once on the complete official 100-image LOL-v2 Real test split, with no exclusions and no scientific-setting changes:

- **Mean PSNR:** `18.53226686142791 dB`
- **Median PSNR:** `18.192788332030815 dB`
- **Mean RGB-SSIM:** `0.5734772617839705`
- **Images:** `100`
- **Selected step min / median / max:** `17 / 22 / 24`
- **Total synchronized inference time:** `70.03712362400256 s`
- **Mean inference time:** `0.7003712362400256 s/image`

Selected-step histogram: `17:5, 19:5, 20:7, 21:10, 22:29, 23:34, 24:10`.

The complete target-free output/decision table was frozen before any clean-reference payload was opened, with SHA256:

`7b00345e2437b558f40e196eba4ce8114a50c219d24dd856c1da94c2596e32e3`

Inference-stage accounting was exactly 100 low reads and `reference_reads=0`; only after the output freeze were the 100 official normal-light references read for post-hoc metrics. Independent verification reproduced cohort provenance, output-table hash, metric aggregation and the information-boundary ordering to machine precision.

**Scientific interpretation:** this is the first genuine held-out in-domain Final-Ours result and is now immutable evaluation evidence. It is not a development signal and cannot authorize method/hyperparameter changes or outcome-driven reruns. The official LOL-v2 Real test has therefore moved from sealed to **evaluation-frozen / forbidden-for-tuning** status.

## Historical mechanism conclusions retained

- T062/T063 established a strong zero-reference trajectory, roughly `+3.5` to `+3.9 dB` mean PSNR over T036 replicates across independent internal 100-image cohorts.
- T063-A/T064-A showed severe historical failures were selection-limited rather than trajectory-capacity-limited: safe prefix checkpoints existed for every diagnosed image.
- T066-A dynamics are useful primarily as a lower safety-entry signal, not a reliable late-stage catastrophic-quality monitor.
- T067-B's frozen `lambda=0.875` interpolation between first-safe entry and normalized progress materially improved safety while retaining utility.
- An already-exposed rare tail motivated several predeclared diagnostics, all closed as insufficient: absolute-step cap, cumulative objective-motion knee, tail-local motion/objective inefficiency, aggregate component-value regret, final projection pressure and symmetric component-gradient cancellation.
- Continuing to fit exposed or official-test failures would be post-hoc overfitting; the method-design phase remains closed.

## Fair-comparison program

The user's stated priority and research-lead lock are now:

1. frozen Final Ours on complete official LOL-v2 Real test — **completed by T071-A**;
2. matched baseline evaluation on that exact same complete official split — **current priority**;
3. later, frozen cross-dataset/domain-shift evaluation on complete held-out sets (at minimum LSRW and UHD-LL), with no target-specific retraining/tuning.

Development baseline anchors exist (Retinexformer T033-A `21.4787864 / 0.7900612`; SNR-Aware T045-A `23.3963299 / 0.8237644`) but are **diagnostic development numbers only** and must not be compared numerically with the T071-A official-test result as the final Ours-vs-baseline gap.

The matched official-test baseline task must resolve the exact previously accepted Retinexformer and SNR-Aware source/checkpoint/config/training provenance, use the same complete 100-image test manifest and the same T071-A metric implementation, and prohibit test-set retraining/tuning or reference-conditioned inference.

## Final-evaluation protocol

- Final Ours and all scientific settings are immutable after T070-A.
- Official-test references are metrics-only evidence after output freeze; official-test outcomes cannot change Ours.
- A fair baseline comparison must use the exact complete T071-A official split and matched metric semantics, with each baseline's training/exposure condition stated explicitly.
- Cross-dataset/domain-shift held-out evaluation remains required for the unknown-degradation motivation. LSRW and UHD-LL remain sealed until separately authorized.
- No target-specific retraining/tuning is allowed on held-out cross-dataset sets; Ours may only perform its already-frozen per-image target-free adaptation.
- Development or exposed-transfer cohorts may support historical mechanism analysis but cannot support the final Ours-vs-baseline gap.

## Information-boundary rules

- Test-time adaptation and checkpoint/state selection must never consume test labels, clean/normal-light targets, reference gradients/Jacobians, oracle values, PSNR/SSIM, degradation masks/gain maps, condition IDs, annotations, semantic image IDs, or per-image baseline outcome/harm labels.
- Quantities computed only from the degraded image/current target-free intermediate image, frozen state and frozen global/development-only assets are permissible.
- Held-out clean/reference targets may be read only after the corresponding output/decision table is irrevocably frozen and hashed.
- Held-out results cannot authorize method tuning, threshold changes, sample exclusion or reruns with altered scientific settings.
- Runs fail closed on source/checkpoint/cohort/provenance mismatch.

## Current open task

**T071-B — matched official LOL-v2 Real baseline table** in `coordination/CHATGPT_TO_CODEX.md`.

Resolve the exact accepted T033-A Retinexformer and T045-A SNR-Aware artifacts/configurations, run them without retraining/tuning on all and only the exact T071-A official low images, freeze outputs before reference evaluation, and compute matched PSNR/RGB-SSIM using the exact T071-A evaluator. Record training/exposure provenance and paired differences relative to the already-frozen Final Ours. Do not modify Ours and do not access LSRW/UHD-LL in this cycle.