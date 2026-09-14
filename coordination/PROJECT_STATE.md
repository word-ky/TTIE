# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Enhancement**

## Core scientific question

Can a vision system adapt a compact spatial image-processing state per test image, without test labels or clean/normal-light targets, so that unknown and spatially heterogeneous degradation is corrected by a reference-free test-time optimization field?

## Current scientific state

**T014 Sobolev Region2 TTT remains the broad fresh-qualified Ours-Core.** It combines the frozen nuisance readout + clean-abstention gate, source-supervised Sobolev restoration energy, hard Region2 EV+gamma adaptation, projected label-free updates, and minimum predicted-energy checkpoint selection. T021-A showed the matched Sobolev-over-value-only advantage also transfers to frozen RGB-SSIM.

**T019 remains a heterogeneous-only fresh-qualified adaptive-geometry extension.** T020 showed that the geometry selector is not universally safe on non-spatial inputs; simple repairs did not close that gap, so universal geometry repair remains paused.

### LOL-v2 Real convergence state

**T022-A established the first leakage-safe validation anchor.** On a deterministic 100-pair validation split carved from the 689 official training pairs, untuned T014 improved raw mean PSNR `8.1097227→9.2728689 dB` and mean RGB-SSIM `0.1600228→0.2730060`. The official 100-pair LOL-v2 Real test set has remained untouched throughout tuning.

**T022-B ruled out checkpoint selection as the main old-family bottleneck.** The 41-state saved-trajectory PSNR oracle adds only `+0.1472 dB` and the SSIM oracle only `+0.00489`.

**T022-C showed the original real-domain action range was too conservative.** Widening only active dark-winner EV from `[0,+0.5]` to `[0,+2.0]` improved validation to `10.2295540 dB / 0.3282315 SSIM`.

**T022-D closed simple budget extension for the old gamma-0.8 trajectory.** `40→80` updates produced `-0.0138358 dB` PSNR and `-0.0016949` SSIM despite `82/100` images selecting step 80.

**T023-A was a negative/insufficient 16-pair real-domain Sobolev recalibration pilot.** It strongly fit its 656-state source bank but reached only `10.5166353 dB / 0.3144893 SSIM`: `+0.2870812 dB` PSNR and `-0.0137422` SSIM versus T022-C. Automatic scaling of that exact tiny-source recipe is not justified.

**T025-A found `+3.3164 dB` non-deployable reference-oracle headroom inside the older T022-C action family.** This exposed real reachability plus old-bound pressure, but all oracle quantities remained quarantined.

**T026-A is the best deployable LOL-v2 validation candidate.** Changing exactly one setting from T022-C — active Region2 gamma lower `0.8→0.5` — improved validation to **`11.1208764 dB / 0.3737918 SSIM`**, i.e. **`+0.8913224 dB / +0.0455603 SSIM`**, passing the predeclared joint gate. The sole 100-image run stayed low-only through output/decision/trajectory freeze; task-specific normal references were deployed only afterward; no oracle quantity entered inference.

**T026-B closed the simple longer-budget route on the promoted gamma-0.5 trajectory.** Changing only `40→80` updates gives `11.2419255 dB / 0.3780444 SSIM`, only `+0.1210491 dB / +0.0042526 SSIM` versus T026-A, failing the `+0.50 dB` materiality gate while mean runtime roughly doubles (`2.274494→4.703975 s/image`). T026-A 40-step remains promoted.

**T028-A establishes very large non-deployable within-family reachability in the exact promoted T026-A family.** The fixed two-start/500-update `REFERENCE_ORACLE_ONLY` audit reaches **`17.4599918 dB / 0.4317158 SSIM`** versus T026-A `11.1208764 dB / 0.3737918`. Paired oracle-minus-T026-A PSNR is **`+6.3391154 dB` mean / `+5.7710854 dB` median**. All 100 images improve PSNR; 18/100 lose SSIM because the oracle optimizes MSE. `94/100` oracle winners occur at step 500, so this is demonstrated reachability rather than a convergence certificate. Active gamma has no boundary hits; active EV has 75/392 upper-bound hits.

**T029-A diagnoses a temporal gradient-direction failure rather than a uniformly bad field.** Across all 4,100 frozen T026-A states, learned-energy versus reference-MSE gradient alignment is only weak/mixed (median cosine `0.171865263`, positive-dot `58.34%`). Early alignment is strong (`step 10`: median `0.726601211`, positive-dot `97%`) but collapses later (`step 30`: median `-0.220573222`, positive-dot `22%`; `step 40`: median `-0.245285485`, positive-dot `24%`). At the already-frozen selected states, median cosine is `-0.278469368` and only `24%` have positive dot. No deployable path consumed reference derivatives.

**T030-A rules out the fixed learned-gradient self-reversal guard as a useful deployable repair.** On a deterministic fresh 100-pair cohort drawn from previously reference-unused non-validation LOL-v2 Real training pairs, unchanged T026-A gives **`10.3286568 dB / 0.3228188 SSIM`**. The predeclared low-only guard gives **`10.1223728 dB / 0.3150107`**, i.e. paired **`-0.2062840 dB / -0.0078081 SSIM`**. It changes 59/100 selections with PSNR win/equal/loss `5/41/54` and adds about `+99.48%` compute. These 100 pairs are now development-used and cannot serve as another fresh qualification cohort.

**T031-A identifies source-support distance as a promising low-only diagnostic of field validity.** Using exactly the accepted T014 source-training bank (80 IDs / 7,346 feature rows) and the frozen T030 100×41 state features, nearest-source distance in the frozen standardized 28-D feature space predicts reference-gradient invalidity with **AUROC `0.720469831`**. Spearman correlation between distance and reference-gradient cosine is **`-0.482299254`**. Valid states have median distance `0.821024194`; invalid states have `1.104813257`. The temporal relation is strong: median distance / invalid fraction are `0.7096 / 2%` at step 10, `0.9597 / 34%` at step 20, `1.1939 / 76%` at step 30, and `1.3022 / 76%` at step 40. Original T026-A selected states have median distance `1.3022`, invalid fraction `77%`, and median reference-gradient cosine `-0.2731`.

T031-A is an association, not a causal certificate: trajectory time and within-image repeated states co-vary with distance. Nevertheless, it is the first strictly low-only signal to pass a predeclared relation gate against the diagnosed late reference-gradient failure. All 4,100 support scores and independent replay were frozen before task normals were decoded; reference gradients were isolated afterward with zero optimizer updates and zero selections. Review follow-up added an external support-freeze binding check before reference access and fixed the portable evidence exporter; no scientific rerun or deployable change occurred.

**Scientific consequence through T031-A:** the leading real-domain failure hypothesis is now *trajectory-induced support departure*: the source-trained Sobolev field is useful early but becomes unreliable as fast states leave the feature region represented by source training. T030 shows that learned-gradient self-reversal is not a useful detector; T031 shows source-support distance is a materially better diagnostic candidate. The next required test is a single fresh, source-derived trust-region intervention with no reference-tuned threshold.

### Benchmark readiness

**T024-A froze the target-free comparison protocol.** Retinexformer `GT_mean` is rejected because reference statistics alter outputs. Retinexformer without `GT_mean` and SNR-Aware have admissible target-free paths; SG-LLIE's released checkpoint is NTIRE-bound rather than matched to its LOL-v2 paper result, LLFormer lacks a matched official LOL-v2 recipe/checkpoint, and Zero-DCE++ belongs to an external-SICE stratum.

**T027-A made Retinexformer exporter-ready without evaluation-data access.** Retinexformer commit `1e9a0efce4b306b6701b824768370ff26066c32a` and official `LOL_v2_real.pth` SHA256 `539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b` are bound. On eight deterministic non-validation training lows, the target-disabled official-forward adapter and TTIE `default_no_gt_mean` exporter are float-identical (`max abs diff=0`) at native `400×600`.

**T027-B made SNR-Aware exporter-ready without evaluation-data access.** Canonical commit `1113144c82adc8bcc4a9ec27749ed75f196a4e4d` and official `LOLv2_real.pth` SHA256 `432d29d370e9f674f1b6763d371b4c24569a86d21f0fd45a5797226274d85781` are bound. On eight deterministic non-validation training lows, the independent pinned-source native-pad16 adapter and TTIE `ttie_native_pad16` exporter are exactly float-identical (`max abs diff=0`). This native-pad16 mode is a predeclared protocol adaptation, not the official resize-based `test4` reproduction.

Neither strong baseline has yet been run for quality metrics on the frozen validation or official test. Benchmark readiness is high enough for execution, but competitiveness is not established. Official test remains sealed until final Ours and admitted baseline protocols are frozen.

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
- **T030-A:** a fixed learned-gradient self-reversal proxy fails on a fresh 100-pair cohort and mostly worsens changed selections.
- **T031-A:** strictly low-only nearest-source distance predicts reference-gradient invalidity with AUROC `0.7205` and rho `-0.4823`; support departure is now the leading late-drift hypothesis, but no deployable rule is qualified yet.
- **T027-A/B:** Retinexformer and SNR-Aware strict target-free exporters are checkpoint-bound and numerically verified on non-evaluation smoke inputs.

## Information-boundary rules

- Test-time adaptation/selection must never consume test labels, clean/normal-light targets, degradation masks/gain maps, condition IDs, annotations, semantic image IDs, reference gradients/Jacobians, oracle values, or evaluation metrics.
- Source/development/validation references may be used only in explicitly declared training, calibration, tuning, or diagnostic stages; they may not enter the per-image test-time decision path.
- Validation/test enhanced outputs and decisions must be finalized and persisted before references or metrics are attached, except explicitly isolated non-deployable reference diagnostics.
- Oracle/reference-gradient diagnostics may motivate later **global** research choices, but no per-image oracle state, target statistic, oracle step, reference gradient, oracle score, or reference-validity label may be consumed by deployable inference/training unless a later task explicitly redefines a source-training split and preserves a separate holdout.
- Any support/trust signal intended for deployment must be computable from source-training artifacts plus the current low-light image/state only. T031 reference validity labels may not set the T032 threshold.
- Final benchmark test sets must remain isolated from hyperparameter/model selection; tuning belongs only on predeclared train/validation data.
- The official 100 LOL-v2 Real test pairs remain untouched through T031-A and must stay untouched until the final Ours configuration and admitted baseline execution protocols are frozen.
- External baselines admitted to the main comparison must also be target-free at inference; target/reference-based brightness matching or selection is inadmissible.
- Fresh/test runs must fail closed on source/provenance/preparation binding mismatches.

## Interpretation

The paper remains image-enhancement-first. T014 supplies the central scientific contribution: learning a reference-free test-time **optimization field** through source-side derivative supervision. T021-A shows this is not MSE-specific. T022–T031 are convergence/diagnostic work on real LOL-v2.

T026-A remains the best deployable validation configuration. T028-A proves its exact action family contains dramatically better states, T029-A localizes a late directional failure, T030-A rejects a naive self-reversal detector, and T031-A shows that the failure is strongly associated with leaving frozen source feature support. This creates a coherent mechanism hypothesis: **test-time optimization can induce its own OOD fast states even when the initial image is manageable, and the learned field becomes unreliable outside its source-supported region.**

The immediate experiment is T032-A: one fixed support radius derived only from source-training geometry, one first-exit trust-region selector, and one genuinely fresh 100-pair qualification. If it fails, source-support distance remains a useful diagnosis rather than a deployable repair and the project should return to benchmark execution / field redesign instead of sweeping thresholds. If it passes, it becomes the first fresh-qualified target-free controller for the late-drift mechanism.

The official test remains sealed. Remaining paper-level gaps are competitive real-benchmark performance, recent matched target-free SOTA comparison, perceptual metrics, and efficiency/quality tradeoffs.

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
- **T031-A: COMPLETED — promising source-support proxy (AUROC `0.7205`, rho `-0.4823`) with strict pre-reference score freeze.**
- **T032-A: ACTIVE — fresh source-support trust-region qualification.**

## Current open task

`T032-A — fresh source-support trust-region qualification` in `coordination/CHATGPT_TO_CODEX.md`.

Use exactly one source-only radius: the 95th percentile of each T014 source row's nearest standardized 28-D neighbor from a **different source image ID**. On one new deterministic 100-pair reference-unused non-validation cohort, run the exact unchanged T026-A 40-step low-only trajectory once and compare baseline minimum-energy selection against a first-exit support-prefix minimum-energy selector. Freeze both decisions/outputs before any normal is deployed. No threshold sweep, no T031 reference-label use, no second cohort, no baseline quality run, and no official test.