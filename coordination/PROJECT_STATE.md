# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Enhancement**

## Core scientific question

Can a vision system adapt a compact spatial image-processing state per test image, without test labels or clean targets, so that unknown and spatially heterogeneous degradation is corrected by a reference-free test-time optimization field?

## Current scientific state

**T014 Sobolev Region2 TTT remains the broad fresh-qualified Ours-Core.** It combines the frozen T006/T007 nuisance readout + clean-abstention gate, the source-supervised T014 Sobolev restoration energy, canonical hard Region2 EV+gamma adaptation, 40 projected label-free updates when active, and minimum predicted-energy checkpoint selection. T014 passed 8/8 development and 12/12 fresh clauses; fresh heterogeneous MSE is `0.03385803`, and unseen-calibration gradient alignment is `73/74` positive with median cosine `0.93606`.

**T019 remains a heterogeneous-only fresh-qualified geometry extension, not the broad default.** T019-B changed only supervision from exact direction to a fixed 1% utility-deadband target while holding representation/folds/network/optimizer/seed/epochs fixed; it passed 7/7 grouped-OOF development clauses and reduced harmful episodes `11→5`. T019-C froze the final two-head selector. T019-D passed 5/5 on a new 40-image / 120-episode heterogeneous fresh cohort: pooled `H1/H0=0.952464`, oracle proximity `1.027465`, offset `0.874063`, left/right `1.004090`, quadrants `1.000000`, with `43 beneficial / 73 equal / 4 harmful` and zero quadrant movement.

**T020-A is a valid fresh non-spatial negative (3/4).** On a second new 40-image / 120-episode cohort, the frozen T019-C selector gives pooled `0.986616×`, clean `1.046125×` fail, homogeneous-dark `0.994103×`, homogeneous-bright `0.963036×`. Clean outcomes are `0 beneficial / 39 equal / 1 harmful`; that fresh cohort is burned for corrective tuning.

**T020-B is a development-only non-spatial target positive (5/5).** The unchanged fixed `delta=0.01` per-axis reference deadband target on accepted development images gives pooled `0.900891×`, clean `0.721586×`, dark `0.906699×`, bright `0.889559×`, with `61 beneficial / 59 equal / 0 harmful`.

**T020-C is a development-only non-spatial OOF negative (3/5).** It kept the frozen T019 28-D representation, historical five image-grouped folds, and exact T019-B `84→64→64→3` learner fixed. Results are pooled `0.947906×` pass, clean `1.207265×` fail, dark `0.937079×` pass, bright `0.970188×` pass, and clean harmful count `1` fail.

**T020-D is a development-only mechanistic diagnosis: the T020-C failure is direction-dominant under the predeclared oracle test.** Necessity-oracle / frozen predicted-sign passes `3/5`, while frozen-necessity / direction-oracle passes `5/5`. Among the 20 harmful T020-C episodes, `19` contain at least one wrong-direction axis. This identifies sign prediction as a first-order bottleneck but does not provide a deployable repair.

**T020-E is a controlled development-only negative (3/5): simple binary sign factorization does not repair the non-spatial safety failure.** With every T020-C move/no-move decision frozen and only fixed `84→64→64→2` lower-vs-upper heads learned from fold-training non-center targets, the result is pooled `0.952217×` pass, clean `1.207265×` fail, dark `0.940200×` pass, bright `0.977458×` pass, and clean harmful count `1` fail. Pooled harmful episodes remain `20→20`; wrong-direction moved axes improve only `31→29`; homogeneous-bright harmful episodes worsen `15→16`.

**T021-A is now a positive frozen-fresh structural-metric transfer result for the central T014 causal comparison.** On the exact accepted T014 Stage-B frozen outputs, with `H0 = region2_ttt_energy_value_only` and `H1 = region2_ttt_energy_sobolev` bound before scoring and the 40 report-only offset rows excluded, the 200 primary rows / 40 source images give mean paired RGB-SSIM delta `+0.0225770368`, median `+0.0075782518`, and source-image-cluster bootstrap 95% CI `[0.0175655974, 0.0281076831]`. The lower bound is strictly positive. Per-condition mean deltas are clean `-0.0002281`, homogeneous-dark `+0.0510952`, homogeneous-bright `+0.0113669`, left-right `+0.0334832`, quadrants `+0.0171680`. This supports transfer of the Sobolev optimization-field advantage beyond MSE, but it is not a universal per-condition improvement claim and not an external SOTA claim. PR #41 was accepted and squash-merged as `0bd623cfde6c7fab5a640b8a482897dac23bc5b1`.

The universal adaptive-geometry repair branch remains **paused**. T014 is the broad default and T019 is a heterogeneous-only extension. The project is now operating on two lines: (1) narrowly scoped truth/mechanism work only when it has clear scientific upside, and (2) higher-priority benchmark/SOTA convergence with validation-only tuning and strict final-test isolation.

## Best current methods

### Broad fresh-qualified Ours-Core

**T014 Sobolev Region2 TTT** — broad deployable image-enhancement method under the accepted protocol.

### Heterogeneous-only fresh-qualified extension

**T019 = T014 + frozen 1%-deadband hard-boundary selector** — fresh-qualified only on the prescribed heterogeneous spatial protocol. T020-A/T020-C/T020-E prevent promotion to a universal default.

## Strongest controlled findings

- T001–T003: spatial ISP capacity helps heterogeneous shifts; capacity/simple priors alone are insufficient.
- T004–T007: zero-shot exposure signals are content-confounded; frozen source-trained nuisance readout + clean abstention is usable.
- T008–T012: semantic TTT has useful directions but drift/stopping/coupling problems; learned checkpointing cannot rescue oracle-limited trajectories.
- T013: scalar restoration-value fitting does not sufficiently constrain the TTT derivative field.
- **T014:** Sobolev derivative supervision is the first fully qualified learned inner objective; the learned optimization field matters more than scalar value fit.
- T015–T017: routing/soft-geometry alternatives expose limited or unsafe transfer paths.
- T018–T019: utility-aware hard geometry is viable and fresh-qualified for heterogeneous shifts.
- T020-A: the heterogeneous geometry selector is not broadly safe on a fresh non-spatial cohort.
- T020-B: the ideal 1% non-spatial deadband target itself is development-safe with zero harmful moves.
- T020-C: unchanged in-domain three-way direction learning remains unsafe.
- T020-D: oracle attribution is direction-dominant.
- T020-E: binary lower-vs-upper factorization does not materially close the direction gap; the simple classifier-repair route is closed for now.
- **T021-A:** on frozen T014 fresh outputs, Sobolev beats the matched value-only control by mean RGB-SSIM `+0.02258`, with a 95% source-image-cluster CI wholly above zero; the core optimization-field claim therefore transfers beyond MSE.

## Information-boundary rules

- Test-time adaptation/selection must never consume test labels, clean/normal-light targets, degradation masks/gain maps, condition IDs, annotations, semantic image IDs, reference gradients/Jacobians, oracle values, or evaluation metrics.
- Source/development references may be used only in explicitly declared training/calibration/diagnostic stages.
- Held-out/fresh decisions and enhanced outputs must be finalized and persisted before their reference metrics, labels, families, or oracles are attached.
- Inspected fresh/test IDs are permanently excluded from corrective tuning unless the protocol explicitly designates them as development/validation before outcomes are seen.
- Failed fresh cohorts may motivate a development-only hypothesis but their per-row references/logits/features/outcomes may not tune that hypothesis.
- OOF claims must be image-grouped; held-out decisions must not depend on held-out labels/reference values.
- Development-only target/oracle/OOF evidence is not fresh qualification.
- Final benchmark test sets must remain isolated from hyperparameter/model selection; tuning belongs on predeclared train/validation data only.
- Fresh/test runs must fail closed on source/provenance/preparation binding mismatches.

## Interpretation

The main paper-level method story is image-enhancement-first. T014 establishes the **optimization-field principle**: a reference-free test-time energy is useful when its derivatives are restoration-useful, not merely when its scalar values fit a reference loss. T021-A adds an important metric-transfer check: the matched Sobolev-over-value-only improvement persists on SSIM over the frozen primary fresh protocol, so the mechanism is not only an MSE artifact. T018–T019 remain a secondary **utility-aware geometry principle** for heterogeneous spatial degradation.

T020 defines the boundary of that extension. The ideal non-spatial target is safe and a direction oracle shows headroom, but both the original three-way learner and the simplest binary sign decomposition fail the same clean-safety contract. Continuing with small classifier patches would risk development overfitting without addressing the paper's larger evidence gaps.

The priority now shifts to **benchmark convergence**: establish a leakage-safe real paired benchmark sandbox, obtain an untuned Ours-Core anchor, then use validation-only tuning and fair external-baseline execution before a one-shot official-test comparison. Real-world enhancement quality, SOTA comparison, and efficiency are now higher-priority paper gaps than further geometry repair. Downstream detection is not required for the current enhancement-focused paper plan.

## Milestones

- T001–T013: completed mechanism/diagnostic sequence.
- **T014: COMPLETED — Sobolev inner objective passed 8/8 development and 12/12 fresh qualification.**
- T015–T018: completed routing/geometry diagnostics; exact-direction selector fresh negative.
- T019-A/B/C: completed deadband target, OOF selector, and final freeze.
- **T019-D: COMPLETED — one-shot fresh heterogeneous geometry qualification positive 5/5.**
- **T020-A: COMPLETED — one-shot fresh non-spatial safety negative 3/4.**
- **T020-B: COMPLETED — development-only non-spatial fixed-1% target positive 5/5, zero harmful.**
- **T020-C: COMPLETED — development-only non-spatial grouped-OOF negative 3/5.**
- **T020-D: COMPLETED — frozen OOF attribution is direction-dominant.**
- **T020-E: COMPLETED — frozen-necessity binary-sign repair negative 3/5.**
- **T021-A: COMPLETED — frozen fresh RGB-SSIM metric-transfer positive; 95% clustered CI above zero.**
- **T022-A: ACTIVE — LOL-v2 Real benchmark bootstrap + untuned Ours-Core validation baseline.**

## Current open task

`T022-A — LOL-v2 Real benchmark bootstrap + untuned Ours-Core validation baseline` in `coordination/CHATGPT_TO_CODEX.md`.

Use the canonical LOL-v2 Real release, keep the 100 official test pairs isolated, deterministically bind 100 validation pairs from the 689 training pairs by filename hash before outcomes, run the exact accepted T014/Ours-Core with no tuning and no paired normal-light access during adaptation, freeze all outputs/decisions before metrics, then report raw-input versus Ours-Core full-resolution RGB PSNR/SSIM and A6000 runtime. No performance gate and no SOTA claim in this cycle; this is the benchmark anchor for subsequent validation-only tuning and external-baseline comparison.
