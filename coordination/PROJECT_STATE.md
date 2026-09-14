# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Enhancement**

## Core scientific question

Can a vision system adapt a compact spatial image-processing state per test image, without test labels or clean/normal-light targets, so that unknown and spatially heterogeneous degradation is corrected by a reference-free test-time optimization field?

## Current scientific state

**T014 Sobolev Region2 TTT remains the broad fresh-qualified Ours-Core.** It combines the frozen nuisance readout + clean-abstention gate, source-supervised Sobolev restoration energy, hard Region2 EV+gamma adaptation, projected label-free updates, and minimum predicted-energy checkpoint selection. T021-A showed the matched Sobolev-over-value-only advantage also transfers to frozen RGB-SSIM.

**T019 remains a heterogeneous-only fresh-qualified adaptive-geometry extension.** T020 showed that the geometry selector is not universally safe on non-spatial inputs; simple repairs did not close that gap, so universal geometry repair remains paused.

### LOL-v2 Real convergence state

**T022-A established the first leakage-safe validation anchor.** On a deterministic 100-pair validation split carved from the 689 official training pairs, untuned T014 improved raw mean PSNR `8.1097227→9.2728689 dB` and mean RGB-SSIM `0.1600228→0.2730060`, but absolute quality was weak. The official 100-pair LOL-v2 Real test set has remained untouched throughout tuning.

**T022-B ruled out checkpoint selection as the main old-family bottleneck.** The 41-state saved-trajectory PSNR oracle adds only `+0.1472 dB` and the SSIM oracle only `+0.00489`.

**T022-C showed the original real-domain action range was too conservative.** Widening only active dark-winner EV from `[0,+0.5]` to `[0,+2.0]` improved validation to `10.2295540 dB / 0.3282315 SSIM`.

**T022-D closed simple budget extension for the old gamma-0.8 trajectory.** `40→80` updates produced `-0.0138358 dB` PSNR and `-0.0016949` SSIM despite `82/100` images selecting step 80.

**T023-A was a negative/insufficient 16-pair real-domain Sobolev recalibration pilot.** It strongly fit its 656-state source bank but reached only `10.5166353 dB / 0.3144893 SSIM`: `+0.2870812 dB` PSNR and `-0.0137422` SSIM versus T022-C. Automatic scaling of that exact tiny-source recipe is not justified.

**T025-A found `+3.3164 dB` non-deployable reference-oracle headroom inside the older T022-C action family.** This exposed real reachability plus old-bound pressure, but all oracle quantities remained quarantined.

**T026-A is the best deployable LOL-v2 validation candidate.** Changing exactly one setting from T022-C — active Region2 gamma lower `0.8→0.5` — improved validation to **`11.1208764 dB / 0.3737918 SSIM`**, i.e. **`+0.8913224 dB / +0.0455603 SSIM`**, passing the predeclared joint gate. The sole 100-image run stayed low-only through output/decision/trajectory freeze; task-specific normal references were deployed only afterward; no oracle quantity entered inference.

**T026-B closed the simple longer-budget route on the promoted gamma-0.5 trajectory.** Changing only `40→80` updates gives `11.2419255 dB / 0.3780444 SSIM`, only `+0.1210491 dB / +0.0042526 SSIM` versus T026-A, failing the `+0.50 dB` materiality gate while mean runtime roughly doubles (`2.274494→4.703975 s/image`). T026-A 40-step remains promoted.

**T028-A establishes very large non-deployable within-family reachability in the exact promoted T026-A family.** The fixed two-start/500-update `REFERENCE_ORACLE_ONLY` audit reaches **`17.459991778 dB / 0.431715830 SSIM`** versus T026-A `11.120876417 dB / 0.373791825`. Paired oracle-minus-T026-A PSNR is **`+6.339115361 dB` mean / `+5.771085393 dB` median**. All 100 images improve PSNR; 18/100 lose SSIM because the oracle optimizes MSE. `94/100` oracle winners occur at step 500, so this is demonstrated reachability rather than a convergence certificate. Active gamma has no boundary hits; active EV has 75/392 upper-bound hits.

**T029-A diagnoses a temporal gradient-direction failure rather than a uniformly bad field.** Across all 4,100 frozen T026-A states, learned-energy versus reference-MSE gradient alignment is only weak/mixed (median cosine `0.171865263`, positive-dot `58.34%`). Early trajectory alignment is strong (`step 10`: median `0.726601211`, positive-dot `97%`) but collapses later (`step 30`: median `-0.220573222`, positive-dot `22%`; `step 40`: median `-0.245285485`, positive-dot `24%`). At the already-frozen selected states, median cosine is `-0.278469368` and only `24%` have positive dot. All active gradients are nondegenerate. The audit performed zero optimizer updates and zero reference-driven selections; all 4,100 states/outputs were bound before task-specific reference deployment. Two aborted GPU attempts and a CPU-only verifier repair are fully disclosed procedural deviations, but no scientific settings/outcomes were selected across attempts and the complete pass plus independent checks are internally consistent.

**T030-A rules out the fixed learned-gradient self-reversal guard as a useful deployable repair.** On a deterministic fresh 100-pair cohort drawn from previously reference-unused non-validation LOL-v2 Real training pairs, unchanged T026-A gives **`10.3286568 dB / 0.3228188 SSIM`**. The predeclared low-only guard — step-10 learned-gradient anchor, first later cosine reversal `<=0`, then minimum predicted energy over the eligible prefix — gives **`10.1223728 dB / 0.3150107`**, i.e. paired **`-0.2062840 dB / -0.0078081 SSIM`**. It changes 59/100 selections with PSNR win/equal/loss `5/41/54` and adds about `+99.48%` compute because all learned gradients are recomputed. Both selectors used the same unchanged 40-update low-only trajectory; all outputs/decisions were frozen before normal deployment, independent replay was exact, and no T029 reference artifact entered selection. These 100 pairs are now development-used and cannot serve as another fresh qualification cohort.

**Scientific consequence of T030-A:** T029's late reference-gradient failure is real, but learned-gradient rotation relative to an early learned-gradient anchor is not a reliable target-free detector. Do not sweep anchor/threshold/fallback on the consumed cohort. The leading next mechanism question is whether directional failure tracks departure from the frozen T014 source-training feature support, which would provide a more causal low-only trust signal.

### Benchmark readiness

**T024-A froze the target-free comparison protocol.** Retinexformer `GT_mean` is rejected because reference statistics alter outputs. Retinexformer without `GT_mean` and SNR-Aware have admissible target-free paths; SG-LLIE's released checkpoint is NTIRE-bound rather than matched to its LOL-v2 paper result, LLFormer lacks a matched official LOL-v2 recipe/checkpoint, and Zero-DCE++ belongs to an external-SICE stratum.

**T027-A made Retinexformer exporter-ready without evaluation-data access.** Retinexformer commit `1e9a0efce4b306b6701b824768370ff26066c32a` and official `LOL_v2_real.pth` SHA256 `539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b` are bound. On eight deterministic non-validation training lows, the target-disabled official-forward adapter and TTIE `default_no_gt_mean` exporter are float-identical (`max abs diff=0`) at native `400×600`.

**T027-B made SNR-Aware exporter-ready without evaluation-data access.** Canonical commit `1113144c82adc8bcc4a9ec27749ed75f196a4e4d` and official `LOLv2_real.pth` SHA256 `432d29d370e9f674f1b6763d371b4c24569a86d21f0fd45a5797226274d85781` are bound. On eight deterministic non-validation training lows, the independent pinned-source native-pad16 adapter and TTIE `ttie_native_pad16` exporter are exactly float-identical (`max abs diff=0`). This native-pad16 mode is a predeclared protocol adaptation, not the official resize-based `test4` reproduction.

Neither strong baseline has yet been run for quality metrics on the frozen validation or official test. Benchmark readiness is improved, but competitiveness is not established.

## Best current methods

### Broad fresh-qualified Ours-Core

**T014 Sobolev Region2 TTT** — broad deployable method under the accepted controlled/fresh protocol.

### Current LOL-v2 validation candidate

**T026-A = T014 + dark-winner EV upper `+2.0` + active gamma lower `0.5`** — best current leakage-safe deployable validation candidate: **`11.1208764 dB / 0.3737918 SSIM`** on the fixed 100-pair LOL-v2 Real validation split. On the separate T030-A fresh development cohort it scores **`10.3286568 dB / 0.3228188 SSIM`**. Not official-test qualified.

### Heterogeneous-only geometry extension

**T019 = T014 + frozen utility-aware hard-boundary selector** — fresh-qualified only for the prescribed heterogeneous spatial protocol.

## Strongest controlled findings

- T001–T003: spatial ISP capacity helps heterogeneous shifts; simple priors/capacity alone are insufficient.
- T004–T007: zero-shot exposure signals are content-confounded; frozen source-trained nuisance readout + clean abstention is usable.
- T008–T013: semantic TTT provides useful directions but scalar objective fitting/checkpointing alone do not produce a reliable restoration field.
- **T014:** Sobolev derivative supervision establishes the optimization-field principle: restoration-useful derivatives matter more than scalar value fit.
- T018–T019: utility-aware hard geometry is viable on heterogeneous shifts.
- T020: adaptive geometry is not universally safe; simple direction-factorization repairs fail.
- **T021-A:** Sobolev beats the matched value-only control in RGB-SSIM on frozen fresh outputs.
- **T022-C / T026-A:** wider physically useful action ranges materially improve real validation.
- **T022-D / T026-B:** merely doubling trajectory length is not a compelling solution.
- **T023-A:** tiny real-source Sobolev recalibration strongly fits its bank but fails the joint validation gate.
- **T025-A / T028-A:** reference-only oracles reveal large within-family reachable headroom, reaching `+6.3391 dB` mean in the promoted family.
- **T029-A:** learned-field direction is strongly restoration-aligned early but becomes weak/negative late; selected T026-A states have only `24%` positive-dot alignment.
- **T030-A:** a fixed learned-gradient self-reversal proxy fails on a fresh 100-pair cohort and mostly worsens changed selections; the late-drift diagnosis does not translate into this simple self-consistency guard.
- **T027-A/B:** Retinexformer and SNR-Aware strict target-free exporters are checkpoint-bound and numerically verified on non-evaluation smoke inputs.

## Information-boundary rules

- Test-time adaptation/selection must never consume test labels, clean/normal-light targets, degradation masks/gain maps, condition IDs, annotations, semantic image IDs, reference gradients/Jacobians, oracle values, or evaluation metrics.
- Source/development/validation references may be used only in explicitly declared training, calibration, tuning, or diagnostic stages; they may not enter the per-image test-time decision path.
- Validation/test enhanced outputs and decisions must be finalized and persisted before references or metrics are attached, except explicitly isolated non-deployable reference diagnostics.
- Oracle/reference-gradient diagnostics may motivate later **global** research choices, but no per-image oracle state, target statistic, oracle step, reference gradient, or oracle score may be consumed by deployable inference/training unless a later task explicitly redefines a source-training split and preserves a separate holdout.
- Final benchmark test sets must remain isolated from hyperparameter/model selection; tuning belongs only on predeclared train/validation data.
- The official 100 LOL-v2 Real test pairs remain untouched through T030-A and must stay untouched until the final Ours configuration and admitted baseline execution protocols are frozen.
- External baselines admitted to the main comparison must also be target-free at inference; target/reference-based brightness matching or selection is inadmissible.
- Fresh/test runs must fail closed on source/provenance/preparation binding mismatches.

## Interpretation

The paper remains image-enhancement-first. T014 supplies the central scientific contribution: learning a reference-free test-time **optimization field** through source-side derivative supervision. T021-A shows this is not MSE-specific. T022–T030 are convergence/diagnostic work on real LOL-v2.

T026-A remains the best deployable validation configuration, but T030-A gives a second fresh development-cohort score (`10.3287/0.3228`) and shows the simple self-reversal guard is counterproductive. T028-A demonstrates that the exact current action family contains dramatically better states, while T029-A localizes a late directional failure. The immediate question is now whether this failure corresponds to leaving the T014 source-training feature support; any candidate trust signal must remain strictly low-only and must be frozen before reference diagnostics.

The official test remains sealed. Downstream detection is not required. Remaining paper-level gaps are competitive real-benchmark performance, recent matched target-free SOTA comparison, perceptual metrics, and efficiency/quality tradeoffs.

## Milestones

- T001–T013: completed mechanism/diagnostic sequence.
- **T014: COMPLETED — broad controlled/fresh Sobolev Ours-Core.**
- **T019-D: COMPLETED — heterogeneous adaptive-geometry fresh positive.**
- **T020-A–E: COMPLETED — universal geometry safety boundary diagnosed; repair route paused.**
- **T021-A: COMPLETED — frozen fresh RGB-SSIM transfer positive.**
- **T022-A–D: COMPLETED — real validation anchor, selector diagnosis, EV widening positive, old longer-budget negative.**
- **T023-A: COMPLETED — 16-pair real-domain Sobolev pilot negative/insufficient.**
- **T024-A: COMPLETED — target-free baseline protocol frozen.**
- **T025-A: COMPLETED — old-family reference-oracle reachability audit.**
- **T026-A: COMPLETED — active gamma lower `0.8→0.5` materially positive.**
- **T026-B: COMPLETED — promoted gamma-0.5 80-step budget negative/insufficient.**
- **T027-A/B: COMPLETED — Retinexformer and SNR-Aware target-free exporters ready.**
- **T028-A: COMPLETED — substantial within-family reference-oracle headroom (`+6.3391 dB` mean).**
- **T029-A: COMPLETED — weak/mixed overall alignment with strong late/selected-state directional mismatch.**
- **T030-A: COMPLETED — fixed low-only self-reversal guard negative/insufficient on a fresh 100-pair cohort.**
- **T031-A: ACTIVE — source-support distance versus reference-gradient validity audit.**

## Current open task

`T031-A — source-support distance versus reference-gradient validity audit` in `coordination/CHATGPT_TO_CODEX.md`.

Use only the accepted T014 source-training feature bank and frozen T030-A 4,100 low-only state features to freeze a fixed nearest-source distance in the T014 standardized 28-D feature space. Only after all support scores are frozen may the already-development-used T030 normals be attached in an isolated diagnostic to compute reference-gradient validity. Do not build a selector, tune a threshold, retrain the field, select a new cohort, or open the official test.