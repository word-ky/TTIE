# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Enhancement**

## Core scientific question

Can a vision system adapt a compact spatial image-processing state per test image, without test labels or clean/normal-light targets, so that unknown and spatially heterogeneous degradation is corrected by a reference-free test-time optimization field?

## Current scientific state

**T014 Sobolev Region2 TTT remains the broad fresh-qualified Ours-Core.** It combines the frozen nuisance readout + clean-abstention gate, source-supervised Sobolev restoration energy, hard Region2 EV+gamma adaptation, projected label-free updates, and minimum predicted-energy checkpoint selection. T021-A showed the matched Sobolev-over-value-only advantage also transfers to frozen RGB-SSIM.

**T019 remains a heterogeneous-only fresh-qualified adaptive-geometry extension.** T020 showed that the geometry selector is not universally safe on non-spatial inputs; simple repairs did not close that gap, so universal geometry repair remains paused.

### LOL-v2 Real convergence state

**T022-A established the leakage-safe validation anchor.** On a deterministic 100-pair validation split carved from the 689 official training pairs, untuned T014 improved raw mean PSNR `8.1097227→9.2728689 dB` and mean RGB-SSIM `0.1600228→0.2730060`. The official 100-pair LOL-v2 Real test set has remained untouched throughout tuning.

**T022-C showed the original real-domain action range was too conservative.** Widening only active dark-winner EV from `[0,+0.5]` to `[0,+2.0]` improved validation to `10.2295540 dB / 0.3282315 SSIM`. T022-D showed that merely extending the old trajectory `40→80` steps was not useful.

**T023-A was a negative/insufficient 16-pair real-domain Sobolev recalibration pilot.** It reached `10.5166353 dB / 0.3144893 SSIM`, only `+0.2870812 dB` PSNR and `-0.0137422` SSIM versus T022-C; automatic scaling of that exact tiny-source recipe is not justified.

**T026-A is the best deployable LOL-v2 validation candidate.** Changing exactly one setting from T022-C — active Region2 gamma lower `0.8→0.5` — improved validation to **`11.1208764 dB / 0.3737918 SSIM`**, i.e. **`+0.8913224 dB / +0.0455603 SSIM`**, passing the predeclared joint gate. The 100-image run stayed low-only through output/decision/trajectory freeze; task-specific normal references were deployed only afterward.

**T026-B closed the simple longer-budget route on the promoted gamma-0.5 trajectory.** `40→80` updates gives `11.2419255 dB / 0.3780444 SSIM`, only `+0.1210491 dB / +0.0042526 SSIM` versus T026-A while mean runtime roughly doubles. T026-A 40-step remains promoted.

**T028-A establishes very large non-deployable within-family reachability in the exact promoted T026-A family.** The isolated two-start/500-update reference oracle reaches **`17.4599918 dB / 0.4317158 SSIM`**. Paired oracle-minus-T026-A PSNR is **`+6.3391154 dB` mean / `+5.7710854 dB` median**; all 100 images improve PSNR. This proves substantial reachable headroom without changing the deployable action family.

**T029-A diagnoses a temporal gradient-direction failure rather than a uniformly bad field.** Across 4,100 frozen T026-A states, learned-energy versus reference-MSE gradient alignment is weak/mixed overall, but early alignment is strong (`step 10`: median cosine `0.7266`, positive-dot `97%`) and collapses later (`step 30`: `-0.2206`, `22%`; `step 40`: `-0.2453`, `24%`). At the already-frozen selected states, median cosine is `-0.2785` and only `24%` have positive dot. Reference derivatives remained diagnostic-only.

**T030-A rules out the fixed learned-gradient self-reversal guard.** On a fresh 100-pair reference-unused cohort, unchanged T026-A gives **`10.3286568 dB / 0.3228188 SSIM`** while the predeclared low-only guard gives **`10.1223728 dB / 0.3150107`**, i.e. paired **`-0.2062840 dB / -0.0078081 SSIM`**.

**T031-A identifies source-support distance as a promising low-only diagnostic of field validity.** Using the accepted T014 source-training bank (80 IDs / 7,346 rows) and frozen T030 state features, nearest-source distance predicts reference-gradient invalidity with **AUROC `0.720469831`** and Spearman correlation to reference-gradient cosine **`-0.482299254`**. Valid-state median distance is `0.821024194`; invalid-state median is `1.104813257`. The relation tracks trajectory time strongly, so this is diagnostic association rather than causal proof.

**T032-A shows that this diagnostic does not directly yield a usable source-only trust region.** The single predeclared global radius, the cross-image source-bank 95th percentile `r_support=0.6469400522`, was tested on one new deterministic 100-pair reference-unused cohort with the exact T026-A low-only trajectory. Baseline T026-A scores **`10.203176520 dB / 0.322824726 SSIM`**; the first-exit support-prefix rule scores **`8.444270861 dB / 0.206756358`**, a paired mean **`-1.758905659 dB / -0.116068368 SSIM`**. All 100 selections change and 33/100 trajectories exit already at step 0. The rule is therefore **negative/insufficient** and must not be rescued by threshold/percentile sweep on this evidence.

The T032 information boundary is valid: source geometry sets the radius; 100 low-only trajectories, support distances, 200 decisions, and both output sets were frozen before successful normal deployment. A failed initial deployment attempt caused a CPU evaluator to stop on a missing receipt before normal decode/metrics and did not change settings or rerun GPU inference. Official test remains untouched.

**T033-A establishes the first strong target-free quality anchor on the frozen development validation split.** Using the unchanged accepted T027-A Retinexformer exporter with `GT_mean=false` and self-ensemble disabled, the exact 100 validation lows produce **`21.478786404 dB / 0.790061209 RGB-SSIM`** versus accepted T026-A **`11.120876417 / 0.373791825`**. The paired Retinexformer-minus-T026-A mean gap is **`+10.357909987 dB / +0.416269384 SSIM`**; Retinexformer wins PSNR on `99/100` and SSIM on `98/100`. All 100 float outputs were frozen before any normal decode and metrics exactly replay the T026 convention.

**Critical T033 limitation:** this 100-pair split is carved from the official LOL-v2 training set, and the released Retinexformer checkpoint was trained supervised on that same training set. Therefore T033 is a descriptive development capacity anchor, **not** an independent held-out generalization or SOTA comparison.

**T034-A shows that expanding the oracle action family with per-region RGB gains materially raises PSNR capacity, but the mechanism is not yet attributable specifically to chromatic correction.** The isolated WB-expanded reference oracle reaches **`19.907949481 dB / 0.408435396 RGB-SSIM`** versus T028-A **`17.459991778 / 0.431715830`**. Paired PSNR improves by **`+2.447957703 dB` mean / `+1.909172977 dB` median**, with `100/100` PSNR wins, passing the predeclared substantial-headroom gate. However RGB-SSIM decreases by **`-0.023280434` mean** and falls on `90/100` images. The active winning WB means are approximately `R=1.4300, G=1.4666, B=1.4301`; because these gains contain a strong shared intensity mode after gamma, T034 proves substantial **WB-family/post-gamma gain** PSNR capacity but does not yet prove that channel-specific chromatic correction causes the gain. The expanded oracle is still non-deployable and reference-only.

Relative to the training-exposed Retinexformer descriptive anchor, T034 narrows the development PSNR gap from about `4.02 dB` (T028) to about `1.57 dB`, but this is not a fair held-out ranking and the large SSIM gap remains.

### Scientific consequence through T034-A

The strongest current mechanism interpretation is:

1. The T014 learned optimization field is restoration-useful in controlled/fresh settings and often directionally useful early on real trajectories.
2. The promoted T026 EV+gamma action family has large reachable real-domain headroom relative to the current deployable selector, so field/trajectory failure is genuine.
3. Late real-domain states frequently exhibit field-direction failure.
4. Source-support distance is correlated with that failure, but a global source-only support radius is too restrictive/mismatched for real LOL-v2 and is **not** a qualified controller.
5. T033 establishes a very strong, but training-exposed, supervised development anchor; it cannot be used as held-out SOTA evidence.
6. T034 shows that action-family capacity also matters: adding post-gamma RGB gains raises the MSE/PSNR oracle ceiling by `+2.45 dB`, reducing the descriptive PSNR gap to the supervised anchor to about `1.57 dB`.
7. The T034 gain cannot yet be called genuinely chromatic because the winning RGB gains are strongly common-mode and can act as an extra intensity degree. Therefore the next diagnostic is a single matched shared-gain oracle control before any deployable WB/field redesign.

## Benchmark readiness

**T024-A froze the target-free comparison protocol.** Retinexformer `GT_mean` is inadmissible because reference statistics alter outputs. Retinexformer without `GT_mean` and SNR-Aware have admissible target-free paths.

**T027-A made Retinexformer exporter-ready without evaluation-data access.** Retinexformer commit `1e9a0efce4b306b6701b824768370ff26066c32a` and official `LOL_v2_real.pth` SHA256 `539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b` are bound. On eight deterministic non-validation training lows, the target-disabled official-forward adapter and TTIE `default_no_gt_mean` exporter are float-identical (`max abs diff=0`) at native `400×600`.

**T027-B made SNR-Aware exporter-ready without evaluation-data access.** Canonical commit `1113144c82adc8bcc4a9ec27749ed75f196a4e4d` and official `LOLv2_real.pth` SHA256 `432d29d370e9f674f1b6763d371b4c24569a86d21f0fd45a5797226274d85781` are bound. The pinned-source native-pad16 adapter and TTIE exporter are float-identical on eight non-evaluation smoke lows. This native-pad16 mode is a predeclared protocol adaptation, not the original resize-based `test4` reproduction.

**T033-A completed the Retinexformer target-free frozen-development benchmark.** It gives `21.478786404 dB / 0.790061209` on the exact frozen 100 development-validation images under the T026 metric convention, with pre-reference output freeze and no target-dependent inference. Because those images belong to the official supervised training set used by the released checkpoint, treat this only as a development capacity anchor.

SNR-Aware remains exporter-ready but has not yet been run for quality metrics. The official LOL-v2 Real test remains sealed until final Ours and baseline protocols are frozen; do not consume official-test metrics to guide further tuning.

## Best current methods

- **Broad fresh-qualified Ours-Core:** T014 Sobolev Region2 TTT.
- **Current LOL-v2 deployable validation candidate:** T026-A = T014 + dark-winner EV upper `+2.0` + active gamma lower `0.5`; **`11.1208764 dB / 0.3737918 SSIM`** on the fixed validation split. Not official-test qualified.
- **Non-deployable exact-family ceiling:** T028-A reference oracle = **`17.4599918 dB / 0.4317158 SSIM`**.
- **Non-deployable WB-expanded ceiling:** T034-A reference oracle = **`19.9079495 dB / 0.4084354 SSIM`**; PSNR `+2.4480 dB` versus T028 with SSIM tradeoff and unresolved common-gain/chromatic attribution.
- **Descriptive strong supervised anchor with training exposure:** T033-A Retinexformer = **`21.4787864 dB / 0.7900612 SSIM`** on the same development split.
- **Heterogeneous-only geometry extension:** T019 = T014 + frozen utility-aware hard-boundary selector.

## Information-boundary rules

- Test-time adaptation/selection must never consume test labels, clean/normal-light targets, degradation masks/gain maps, condition IDs, annotations, semantic image IDs, reference gradients/Jacobians, oracle values, or evaluation metrics.
- Validation/test enhanced outputs and decisions must be finalized and persisted before references or metrics are attached, except explicitly isolated non-deployable reference diagnostics.
- Oracle/reference-gradient diagnostics may motivate only global research choices; no per-image oracle quantity may enter deployable inference.
- External baselines admitted to the main comparison must be target-free at inference; reference-based brightness matching or selection is inadmissible.
- Final benchmark test sets must remain isolated from model/hyperparameter selection. The official 100 LOL-v2 Real test pairs remain untouched through T034-A.
- Fresh/test runs must fail closed on source/checkpoint/cohort/provenance mismatches.

## Milestones

- T001–T013: completed mechanism/diagnostic sequence.
- **T014: COMPLETED — broad controlled/fresh Sobolev Ours-Core.**
- **T019-D: COMPLETED — heterogeneous adaptive-geometry fresh positive.**
- **T020-A–E: COMPLETED — universal geometry safety boundary diagnosed; repair route paused.**
- **T021-A: COMPLETED — frozen fresh RGB-SSIM transfer positive.**
- **T022-A–D: COMPLETED — real validation anchor, selector diagnosis, EV widening positive, longer-budget negative.**
- **T023-A: COMPLETED — 16-pair real-domain Sobolev pilot negative/insufficient.**
- **T024-A: COMPLETED — target-free baseline protocol frozen.**
- **T025-A: COMPLETED — old-family reference-oracle reachability audit.**
- **T026-A: COMPLETED — active gamma lower `0.8→0.5` materially positive.**
- **T026-B: COMPLETED — promoted gamma-0.5 80-step budget negative/insufficient.**
- **T027-A/B: COMPLETED — Retinexformer and SNR-Aware target-free exporters ready.**
- **T028-A: COMPLETED — substantial within-family reference-oracle headroom (`+6.3391 dB` mean).**
- **T029-A: COMPLETED — strong late/selected-state directional mismatch.**
- **T030-A: COMPLETED — fixed low-only self-reversal guard negative/insufficient.**
- **T031-A: COMPLETED — source-support distance is a promising diagnostic proxy.**
- **T032-A: COMPLETED — fixed source-only first-exit trust-region rule negative/insufficient (`-1.7589 dB / -0.1161 SSIM`).**
- **T033-A: COMPLETED — Retinexformer target-free development anchor (`21.4788 dB / 0.7901 SSIM`), with explicit training-exposure limitation.**
- **T034-A: COMPLETED — WB-family reference oracle adds `+2.4480 dB` PSNR versus T028, with `-0.0233` SSIM and unresolved common-mode attribution.**
- **T035-A: ACTIVE — shared post-gamma gain attribution control.**

## Current open task

`T035-A — common-gain control for T034 WB attribution` in `coordination/CHATGPT_TO_CODEX.md`.

Reuse the exact T028/T034 two-start/500-update reference-oracle protocol on the original frozen 100 validation pairs, but add only one per-region scalar gain shared identically across RGB at the WB position. Compare this control read-only against both T028 and full-WB T034 to determine whether common intensity explains most of the `+2.45 dB` PSNR gain. This remains `REFERENCE_ORACLE_ONLY`; no reference quantity may enter deployable TTT, and the official LOL-v2 Real test remains sealed.