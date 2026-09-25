# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Enhancement**

This file is the live scientific state. Detailed task history remains in Git history and `research_log/`.

## Core scientific question

Can a vision system adapt a compact spatial image-processing state per test image, without test labels or clean/normal-light targets, so that unknown and spatially heterogeneous degradation is corrected by a reference-free test-time optimization trajectory with reliable target-free stopping?

## Frozen Final Ours

T070-A froze the unchanged T067-B candidate as Final Ours for the current Phase-1 fair-evaluation program. Immutable manifest SHA256:

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

The current T070-A method is **not competitive with target-domain supervised LOL-v2 Real checkpoints in-domain** on PSNR/SSIM. No in-domain superiority claim is supported. This result is nevertheless not the decisive test of the unknown-degradation motivation because Retinexformer and SNR-Aware use paired supervised LOL-v2 Real source checkpoints, while T070-A can adapt per target image using only the degraded target image.

The next decisive Phase-1 comparison is therefore source-frozen domain transfer: keep T070-A and the exact LOL-v2-trained baseline checkpoints unchanged, move all methods to a different target dataset, allow only T070-A's already-frozen degraded-image-only per-image TTT, and prohibit all target-specific retraining/tuning.

Do not use the official-test gap to tune T070-A.

## Paper positioning update — 2026-09-24

The paper is **not** positioned as an in-domain LLIE SOTA method. Its primary question is deployment robustness under source-to-target degradation shift: can a frozen source method adapt at test time to an unseen target distribution using only degraded target images and no clean/reference information?

Therefore:

- the official LOL-v2 Real result remains an honest **source-domain reference / sanity check**, not a headline superiority claim;
- no in-domain rescue or SOTA-targeting branch is required for the current paper;
- the decisive evidence is source-frozen cross-domain performance on complete unseen target datasets;
- the main competitor set should emphasize TTA / cross-domain / zero-shot or generalization-oriented restoration methods, while Retinexformer and SNR-Aware remain useful fixed-source anchors;
- Ours-static vs Ours-TTT is required on target domains to isolate the gain caused by test-time adaptation itself;
- no official-test result may be used to tune T070-A.

This positioning is intentionally narrower than universal LLIE SOTA: source-domain specialization may outperform T070-A in-domain, while the paper claims value only if T070-A demonstrates superior or clearly stronger robustness/adaptation under unseen degradation shift.

## Historical mechanism conclusions retained

- T062/T063 established a strong zero-reference trajectory, roughly `+3.5` to `+3.9 dB` mean PSNR over T036 replicates across independent internal cohorts.
- T063-A/T064-A showed historical severe failures were selection-limited rather than trajectory-capacity-limited.
- T066-A dynamics are useful primarily as a lower safety-entry signal, not a reliable late-stage catastrophic-quality monitor.
- T067-B's frozen `lambda=0.875` interpolation materially improved safety while retaining utility.
- Predeclared late-tail diagnostics were closed as insufficient: absolute-step cap, cumulative objective-motion knee, tail-local motion/objective inefficiency, aggregate component-value regret, final projection pressure and symmetric component-gradient cancellation.
- Continuing to fit development/exposed/official-test failures is post-hoc overfitting; T070-A method design remains closed during the Phase-1 fair cross-dataset program.

## Fair-comparison program

1. Frozen T070-A on complete official LOL-v2 Real test — **completed, T071-A**.
2. Matched Retinexformer/SNR-Aware on the same official split — **completed, T071-B**.
3. Source-frozen cross-dataset/domain-shift evaluation — **current priority**.
   - **UHD-LL is the active target.** T072-E-R1 sealed the freeze-before-reference harness; T072-I sealed the canonical 150-image native-`3840×2160` low-only cohort and exact 450-job dispatch; T072-L is accepted as `UHDLL_ANALYSIS_SPEC_SEALED`, fixing the exact T071-B RGB PSNR/RGB-SSIM provenance, complete-sample policy, paired Ours-minus-baseline statistics, and one shared 10,000-resample paired bootstrap stream with seed `20260922` before any target reference is opened.
   - LSRW remains deferred until the user-provided canonical archive is available and does not block UHD-LL.
4. After the source-frozen cross-dataset tables exist, review whether the cross-domain/TTA claim is sufficiently supported; do not open an in-domain-SOTA rescue branch unless the user explicitly changes the paper objective again.

Development baseline anchors and exposed-transfer numbers remain mechanism/diagnostic evidence only and must not be used as final Ours-vs-baseline gaps.

## Cross-dataset protocol

For UHD-LL and later LSRW:

- T070-A stays exactly at the immutable artifact for Phase-1 evaluation.
- Retinexformer and SNR-Aware stay at the exact T071-B accepted **LOL-v2 Real source** checkpoints/configs; no target-domain checkpoint substitution.
- No target-specific retraining, fine-tuning, calibration, threshold selection or hyperparameter tuning is allowed.
- T070-A may only perform its already-frozen per-image degraded-image-only test-time adaptation.
- All method outputs must be frozen and hashed before any target clean/reference payload is read.
- References are metrics-only after output freeze.
- Complete canonical target test splits are required; no convenient subset or outcome-driven exclusion.
- For UHD-LL, the T072-L preregistered analysis specification is mandatory and cannot be altered after target outcomes are observed.

This source-frozen protocol directly tests the unknown-degradation/domain-shift motivation rather than target-domain supervised specialization.

## Information-boundary rules

- Test-time adaptation and checkpoint/state selection must never consume test labels, clean/normal-light targets, reference gradients/Jacobians, oracle values, PSNR/SSIM, degradation masks/gain maps, condition IDs, annotations, semantic image IDs, or per-image baseline outcome/harm labels.
- Quantities computed only from the degraded image/current target-free intermediate image, frozen state and frozen global/development-only assets are permissible.
- Held-out clean/reference targets may be read only after the corresponding complete output/decision tables for all compared methods are irrevocably frozen and hashed.
- Held-out results cannot authorize method tuning, threshold changes, sample exclusion or altered-setting reruns.
- Runs fail closed on source/checkpoint/cohort/provenance/geometry mismatch.

## Current task status

**T072-AZ-B accepted:** RetinexFormer and SNR-Aware each have 150/150 independently verified native UHD-LL low-only outputs on the paid RTX 4090; combined manifest SHA256 `73c8d304c8bb88c4612a13598b5d4dd76378e0c3107bd97011941037a4427001` (PR #220). No UHD-LL reference or metric was accessed.

**T073-A-R1 locally sealed under the user-approved long-horizon plan; hourly research-lead audit remains active.** The initial T073-A seal claim was superseded by R1 findings. Revised `research_log/T073A/` fixes exactly eight Tier-1 rows; ZERO-IG is preferred additional and GM-MoE secondary. PromptIR and PromptIR+DCTTA now share the prospectively chosen official five-task `epoch=80.ckpt` (SHA256 `206baf0dd10f636f025b33b5ee7eb63a353fcbf4d50858b62f9480a6d4be9d4a`), selected from task provenance before target execution, not from UHD-LL outcomes. DCTTA has `LOW_ONLY_DCTTA_REQUIRED`; its official paired loader opens GT, so a task-owned low-only wrapper and source-side equivalence proof remain required before model execution. T072-L metric/bootstrap and accepted T072-AZ-B manifest are unchanged; the independent verifier and 19 rejection tests pass. Only RetinexFormer/SNR-Aware are `FROZEN_OUTPUTS`; eight rows remain `PENDING_OUTPUTS`. `reference_reads=0`, `metrics=0`, `model_runs=0`. UHD-LL references remain sealed; continue Phase B source-side wrapper preparation, not target execution until its gate passes.

**Phase B progress through T073-C (2026-09-25):** The T073-A paragraph above records its historical preregistration state; the preregistration artifact has not been retroactively edited. Ours-Step0 now has 150/150 native UHD-LL low-only outputs independently verified and frozen via `research_log/T073C/step0_freeze_receipt.json` (output manifest SHA256 `76a1ee7b12f41991499802024811321dcce3b0af0bae2c06ab41dba719180cce`). Ours-TTT has passed a one-image native-4K execution smoke but not the full row. PromptIR static/DCTTA B3a low-only binding passed; B3b native-4K static smoke encountered PyTorch 32-bit convolution indexing and remains provisionally blocked without an execution-equivalent rescue. RetinexFormer/SNR-Aware accepted outputs remain untouched. UHD-LL reference reads and metrics remain zero; do not open references before the all-Tier-1 output gate.

**Phase B material blockers (2026-09-25 15:48+08):** Ours-TTT full execution stopped at image 2 because frozen T070-A asserts on an inactive gate; low-only Step0 receipts show 49/150 inactive images, and a fresh original-CLI run reproduces the assertion. Ours-TTT remains `BLOCKED_SCIENTIFIC_NO_ACTIVE_GATE` pending research-lead protocol decision, not frozen. PromptIR's synthetic row-scheduling probes are not an accepted native-4K rescue: after the index error they reach 48 GB GPU OOM; both PromptIR rows remain blocked with no outputs. MR. Illuminate and QuadPrior official weights are still pending acquisition. RetinexFormer, SNR-Aware and Ours-Step0 remain the only three verified/frozen Tier-1 output rows. UHD-LL target references and metrics remain sealed/zero.
