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

### Scientific consequence through T032-A

The strongest current mechanism interpretation is:

1. The T014 learned optimization field is restoration-useful in controlled/fresh settings and often directionally useful early on real trajectories.
2. The promoted T026 action family has large reachable real-domain headroom, so action-space capacity alone is not the main current explanation.
3. Late real-domain states frequently exhibit field-direction failure.
4. Source-support distance is correlated with that failure, but a global source-only support radius is too restrictive/mismatched for real LOL-v2 and is **not** a qualified controller; 33% step-0 exits make this especially clear.
5. Therefore, do **not** sweep support thresholds or declare support-aware stopping solved. The project should first establish the actual strong-baseline gap, then choose between optimization-field redesign and broader image-formation/action redesign based on competitive evidence.

## Benchmark readiness

**T024-A froze the target-free comparison protocol.** Retinexformer `GT_mean` is inadmissible because reference statistics alter outputs. Retinexformer without `GT_mean` and SNR-Aware have admissible target-free paths.

**T027-A made Retinexformer exporter-ready without evaluation-data access.** Retinexformer commit `1e9a0efce4b306b6701b824768370ff26066c32a` and official `LOL_v2_real.pth` SHA256 `539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b` are bound. On eight deterministic non-validation training lows, the target-disabled official-forward adapter and TTIE `default_no_gt_mean` exporter are float-identical (`max abs diff=0`) at native `400×600`.

**T027-B made SNR-Aware exporter-ready without evaluation-data access.** Canonical commit `1113144c82adc8bcc4a9ec27749ed75f196a4e4d` and official `LOLv2_real.pth` SHA256 `432d29d370e9f674f1b6763d371b4c24569a86d21f0fd45a5797226274d85781` are bound. The pinned-source native-pad16 adapter and TTIE exporter are float-identical on eight non-evaluation smoke lows. This native-pad16 mode is a predeclared protocol adaptation, not the original resize-based `test4` reproduction.

Neither strong baseline has yet been run for quality metrics on the frozen validation or official test. Benchmark execution is now the immediate priority. Official test stays sealed until final Ours and baseline protocols are frozen.

## Best current methods

- **Broad fresh-qualified Ours-Core:** T014 Sobolev Region2 TTT.
- **Current LOL-v2 validation candidate:** T026-A = T014 + dark-winner EV upper `+2.0` + active gamma lower `0.5`; **`11.1208764 dB / 0.3737918 SSIM`** on the fixed validation split. Not official-test qualified.
- **Heterogeneous-only geometry extension:** T019 = T014 + frozen utility-aware hard-boundary selector.

## Information-boundary rules

- Test-time adaptation/selection must never consume test labels, clean/normal-light targets, degradation masks/gain maps, condition IDs, annotations, semantic image IDs, reference gradients/Jacobians, oracle values, or evaluation metrics.
- Validation/test enhanced outputs and decisions must be finalized and persisted before references or metrics are attached, except explicitly isolated non-deployable reference diagnostics.
- Oracle/reference-gradient diagnostics may motivate only global research choices; no per-image oracle quantity may enter deployable inference.
- External baselines admitted to the main comparison must be target-free at inference; reference-based brightness matching or selection is inadmissible.
- Final benchmark test sets must remain isolated from model/hyperparameter selection. The official 100 LOL-v2 Real test pairs remain untouched through T032-A.
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
- **T033-A: ACTIVE — Retinexformer target-free frozen-validation benchmark.**

## Current open task

`T033-A — Retinexformer target-free frozen-validation benchmark` in `coordination/CHATGPT_TO_CODEX.md`.

Use the already accepted T027-A exporter and pinned official checkpoint on exactly the frozen 100-image LOL-v2 Real validation split. Freeze all 100 target-free outputs before normals/metrics are opened, then evaluate with exactly the T026 metric convention and report the paired gap versus accepted T026-A. No tuning, no SNR-Aware in this cycle, and no official test.