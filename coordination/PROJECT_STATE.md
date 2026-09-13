# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Enhancement**

## Core scientific question

Can a vision system adapt a compact spatial image-processing state per test image, without test labels or clean targets, so that unknown and spatially heterogeneous degradation is corrected by a reference-free test-time optimization field?

## Current scientific state

**T014 Sobolev Region2 TTT remains the broad fresh-qualified Ours-Core.** It combines the frozen T006/T007 nuisance readout + clean-abstention gate, the source-supervised T014 Sobolev restoration energy, canonical hard Region2 EV+gamma adaptation, 40 projected label-free updates when active, and minimum predicted-energy checkpoint selection. T014 passed 8/8 development and 12/12 fresh clauses; fresh heterogeneous MSE is `0.03385803`, and unseen-calibration gradient alignment is `73/74` positive with median cosine `0.93606`.

**T019 remains a heterogeneous-only fresh-qualified geometry extension, not the broad default.** T019-D passed 5/5 on a new 40-image / 120-episode heterogeneous fresh cohort: pooled `H1/H0=0.952464`, oracle proximity `1.027465`, offset `0.874063`, left/right `1.004090`, quadrants `1.000000`, with `43 beneficial / 73 equal / 4 harmful` and zero quadrant movement.

**T020 defines the boundary of universal adaptive geometry.** T020-A is a valid fresh non-spatial negative; T020-B shows the ideal fixed-1% non-spatial target itself is development-safe; T020-C remains unsafe with the unchanged three-way learner; T020-D attributes the first-order failure to direction sign; and T020-E shows that simple binary lower-vs-upper factorization still does not repair the safety failure. Universal geometry repair is therefore paused rather than repeatedly patched on development data.

**T021-A is a positive frozen-fresh structural-metric transfer result for the central T014 causal comparison.** On the exact accepted T014 Stage-B frozen outputs, with `H0 = region2_ttt_energy_value_only` and `H1 = region2_ttt_energy_sobolev` bound before scoring, the 200 primary rows / 40 source images give mean paired RGB-SSIM delta `+0.0225770368`, median `+0.0075782518`, and source-image-cluster bootstrap 95% CI `[0.0175655974, 0.0281076831]`. The lower bound is strictly positive. This supports transfer of the Sobolev optimization-field advantage beyond MSE, while the small clean-condition decrease remains visible and no external SOTA claim is made.

**T022-A is now the first real paired-benchmark anchor and reveals a major benchmark-convergence gap.** On a deterministic 100-pair validation split drawn from the 689 LOL-v2 Real training pairs, the exact untuned T014/Ours-Core improves raw mean PSNR `8.1097227→9.2728689` dB and mean RGB-SSIM `0.1600228→0.2730060`; medians improve PSNR `7.6001615→8.6800929` and SSIM `0.1389773→0.2426305`. All 100 images are active and execute 40 updates; mean selected checkpoint step is `26.95`. A6000 trajectory runtime is mean `2.3006 s`, median `2.4303 s`, p95 `2.4770 s` per image. The canonical 689/100 train/test structure and pairing were verified, the validation split was filename-hash-bound before outcomes, all low-light-only decisions/outputs were frozen before normal-light deployment, and the official 100 test pairs remain untouched. PR #42 was accepted and squash-merged as `98054ad96d87f02ff2b6dea60a9199e0214b41f7`.

T022-A is scientifically encouraging as a transfer sanity check because both PSNR and SSIM improve with zero LOL-v2 tuning, but its **absolute restoration quality is weak for a competitive low-light enhancement benchmark**. Therefore the current priority is not an official-test run or a SOTA claim. The immediate question is whether much better validation states already exist inside the frozen 40-step trajectories and are missed by learned-energy checkpoint selection, or whether the trajectory/objective/action space itself lacks real-domain headroom.

The project is now operating on two lines: (1) narrowly scoped truth/mechanism work only when it directly determines a tuning decision, and (2) higher-priority benchmark/SOTA convergence with validation-only tuning, strong external baselines, and strict one-shot final-test isolation.

## Best current methods

### Broad fresh-qualified Ours-Core

**T014 Sobolev Region2 TTT** — broad deployable image-enhancement method under the accepted controlled/fresh protocol; currently the benchmark method being transferred to LOL-v2 Real.

### Heterogeneous-only fresh-qualified extension

**T019 = T014 + frozen 1%-deadband hard-boundary selector** — fresh-qualified only on the prescribed heterogeneous spatial protocol. T020 prevents promotion to a universal default.

## Strongest controlled findings

- T001–T003: spatial ISP capacity helps heterogeneous shifts; capacity/simple priors alone are insufficient.
- T004–T007: zero-shot exposure signals are content-confounded; frozen source-trained nuisance readout + clean abstention is usable.
- T008–T012: semantic TTT has useful directions but drift/stopping/coupling problems; learned checkpointing cannot rescue oracle-limited trajectories.
- T013: scalar restoration-value fitting does not sufficiently constrain the TTT derivative field.
- **T014:** Sobolev derivative supervision is the first fully qualified learned inner objective; the learned optimization field matters more than scalar value fit.
- T018–T019: utility-aware hard geometry is viable and fresh-qualified for heterogeneous shifts.
- T020: the adaptive-geometry extension is not universally safe; simple direction-classifier repairs do not close the gap.
- **T021-A:** Sobolev beats the matched value-only control by mean RGB-SSIM `+0.02258`, with a 95% source-image-cluster CI wholly above zero on frozen fresh outputs.
- **T022-A:** exact untuned Ours-Core transfers to LOL-v2 Real validation with `+1.1631 dB` mean PSNR and `+0.1130` mean SSIM over raw input, but the absolute benchmark quality remains weak and now becomes the main convergence problem.

## Information-boundary rules

- Test-time adaptation/selection must never consume test labels, clean/normal-light targets, degradation masks/gain maps, condition IDs, annotations, semantic image IDs, reference gradients/Jacobians, oracle values, or evaluation metrics.
- Source/development/validation references may be used only in explicitly declared training, calibration, tuning, or diagnostic stages; they may not enter the per-image test-time decision path.
- Held-out/fresh/test decisions and enhanced outputs must be finalized and persisted before their reference metrics, labels, families, or oracles are attached.
- Inspected fresh/test IDs are permanently excluded from corrective tuning unless the protocol explicitly designated them as development/validation before outcomes were seen.
- Final benchmark test sets must remain isolated from hyperparameter/model selection; tuning belongs on predeclared train/validation data only.
- The 100 official LOL-v2 Real test pairs are still isolated after T022-A and must remain untouched until a final method/configuration is frozen.
- Fresh/test runs must fail closed on source/provenance/preparation binding mismatches.

## Interpretation

The main paper-level method story remains image-enhancement-first. T014 establishes the **optimization-field principle**: a reference-free test-time energy is useful when its derivatives are restoration-useful, not merely when its scalar values fit a reference loss. T021-A shows that this matched Sobolev-over-value-only advantage transfers from MSE to SSIM on frozen fresh outputs. T018–T019 remain a secondary utility-aware geometry principle for heterogeneous spatial degradation.

The project has now entered benchmark convergence. T022-A is the first honest real paired benchmark anchor: untuned Ours-Core helps substantially relative to the dark input, but the resulting `9.27 dB / 0.273 SSIM` validation anchor is not yet a competitive endpoint. Before spending validation budget on learning rate, bounds, step count, checkpointing, or retraining, T022-B will audit the already frozen trajectory to separate checkpoint-selection error from trajectory/action-space limitation. That diagnosis will determine the single tuning axis for the following cycle.

Downstream detection is not required for the current enhancement-focused paper plan. The remaining paper-level gaps are real-benchmark performance, strong baseline/SOTA comparison under matched protocol, LPIPS/perceptual evidence where feasible, and efficiency/quality tradeoffs.

## Milestones

- T001–T013: completed mechanism/diagnostic sequence.
- **T014: COMPLETED — Sobolev inner objective passed 8/8 development and 12/12 fresh qualification.**
- T015–T018: completed routing/geometry diagnostics.
- **T019-D: COMPLETED — one-shot fresh heterogeneous geometry qualification positive 5/5.**
- **T020-A–E: COMPLETED — universal geometry safety boundary diagnosed; simple repair route paused.**
- **T021-A: COMPLETED — frozen fresh RGB-SSIM metric-transfer positive.**
- **T022-A: COMPLETED — leakage-safe untuned LOL-v2 Real validation anchor; improves raw input but absolute benchmark quality is weak.**
- **T022-B: ACTIVE — frozen LOL-v2 validation trajectory headroom audit.**

## Current open task

`T022-B — frozen LOL-v2 validation trajectory headroom audit` in `coordination/CHATGPT_TO_CODEX.md`.

Use only the frozen T022-A 100-image validation trajectories and validation references, with no new TTT and no official-test work. Re-render all saved states, verify selected-state reconstruction, report global fixed-step and per-image PSNR/SSIM oracle headroom plus projection-bound saturation. The goal is to decide whether the next convergence cycle should tune checkpoint selection or the trajectory/objective/action space; no tuning is launched in T022-B itself.
