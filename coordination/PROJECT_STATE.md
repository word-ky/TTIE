# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Enhancement**

## Core scientific question

Can a vision system adapt a compact spatial image-processing state per test image, without test labels or clean/normal-light targets, so that unknown and spatially heterogeneous degradation is corrected by a reference-free test-time optimization field?

## Current scientific state

**T014 Sobolev Region2 TTT remains the broad fresh-qualified Ours-Core.** It combines the frozen nuisance readout + clean-abstention gate, source-supervised Sobolev restoration energy, hard Region2 EV+gamma adaptation, projected label-free updates, and minimum predicted-energy checkpoint selection. T014 passed the accepted controlled development/fresh protocol, and T021-A later showed that its matched Sobolev-over-value-only advantage also transfers to frozen RGB-SSIM.

**T019 remains a heterogeneous-only fresh-qualified adaptive-geometry extension.** T020 established that this geometry selector is not universally safe on non-spatial inputs; simple three-way or binary direction repairs did not close that gap. Universal geometry repair remains paused.

**T022-A established the first leakage-safe LOL-v2 Real validation anchor.** On the deterministic 100-pair validation split carved from the 689 official training pairs, untuned T014 improved raw mean PSNR `8.1097227→9.2728689 dB` and mean RGB-SSIM `0.1600228→0.2730060`, but absolute real-benchmark quality was weak. The official 100-pair LOL-v2 Real test set has remained untouched throughout tuning.

**T022-B ruled out checkpoint selection as the main real-domain bottleneck.** Over the exact 41 frozen checkpoints from T022-A, the per-image PSNR oracle improves only `+0.1472 dB` on average and the SSIM oracle only `+0.00489`; the best global fixed step is effectively tied with the learned selector. This redirected tuning toward the trajectory/action space rather than selector design.

**T022-C is the current best deployable LOL-v2 validation candidate and a materially positive action-range result.** Widening only active dark-winner Region2 EV from `[0,+0.5]` to `[0,+2.0]`, with all other scientific settings frozen, improved mean validation PSNR `9.2728689→10.2295540 dB` (`+0.9566851 dB`) and mean RGB-SSIM `0.2730060→0.3282315` (`+0.0552255`). All low-light inference artifacts were frozen before normal-light references were deployed.

**T022-D closed simple update-budget extension.** Changing only `max_steps: 40→80` from T022-C produced `-0.0138358 dB` PSNR and `-0.0016949` SSIM despite `82/100` images selecting step 80. Following the same learned field longer is not the solution.

**T023-A was a negative/insufficient 16-pair real-domain Sobolev recalibration pilot.** The new head fit its 656-state source bank nearly perfectly in gradient space (positive cosine fraction `1.0`, median cosine `0.9881`) but changed validation from T022-C to only `10.5166353 dB / 0.3144893 SSIM`: `+0.2870812 dB` PSNR and `-0.0137422` SSIM. Near-perfect fitting on a tiny real-source bank does not guarantee joint metric generalization, so automatic scaling of that exact recipe is not justified.

**T024-A froze the target-free strong-baseline comparison protocol, but the initial five-method roster is not benchmark-ready.** Retinexformer without `GT_mean` and SNR-Aware have verifiable target-free LOL-v2 paths for later final-test reproduction; Retinexformer `GT_mean` is rejected because reference statistics alter the output. SG-LLIE's released checkpoint is NTIRE-bound rather than matched to its LOL-v2 paper result, LLFormer lacks a matched official LOL-v2 recipe/checkpoint, and Zero-DCE++ belongs to an external-SICE stratum. Recent-SOTA coverage remains incomplete and must resume before final paper claims.

**T025-A completed the non-deployable reference-oracle reachability audit and substantially changes the diagnosis.** Inside the exact frozen T022-C gate + Region2 EV/gamma family, the fixed two-start/500-update reference oracle reaches mean `13.545967 dB / 0.384052 SSIM` versus T022-C `10.229554 dB / 0.328231 SSIM`, a mean gap of `+3.316413 dB / +0.055821 SSIM`; PSNR improves on `100/100` images. Therefore the learned trajectory is far from exhausting useful states already present in the current family. This is not a certified global ceiling because `69/100` winning oracle states occur at step 500.

**T025-A also exposes strong action-bound pressure.** `388/392` active oracle gamma coordinates hit the current lower bound `0.8`; all 86 bright-winner EV coordinates hit their allowed upper bound `0`; the 306 dark-winner EV coordinates have median `1.997626`, near `+2`. The evidence therefore supports two simultaneous facts: the learned optimization field misses substantially better reachable states, and the present action bounds may still be restrictive. T025-A remains strictly `REFERENCE_ORACLE_ONLY`; no oracle state/reference quantity may enter deployable TTT.

**T026-A is now the active validation-only performance probe.** It changes exactly one deployable hyperparameter from T022-C: active gamma lower bound `0.8→0.5`, keeping gamma upper `1.25`, EV bounds, gate, energy, Region2 geometry, optimizer, 40-step budget, and learned-energy checkpoint selection fixed. Validation inference must remain low-light-only and freeze all outputs/decisions before reference evaluation.

## Best current methods

### Broad fresh-qualified Ours-Core

**T014 Sobolev Region2 TTT** — broad deployable method under the accepted controlled/fresh protocol.

### Current LOL-v2 validation candidate

**T022-C = T014 + dark-winner EV upper bound `+2.0`** — best current leakage-safe validation candidate; not yet official-test qualified. T026-A is testing one global gamma-bound adjustment and is not promoted unless it passes the predeclared joint PSNR/SSIM gate.

### Heterogeneous-only geometry extension

**T019 = T014 + frozen utility-aware hard-boundary selector** — fresh-qualified only for the prescribed heterogeneous spatial protocol.

## Strongest controlled findings

- T001–T003: spatial ISP capacity helps heterogeneous shifts; simple priors/capacity alone are insufficient.
- T004–T007: zero-shot exposure signals are content-confounded; frozen source-trained nuisance readout + clean abstention is usable.
- T008–T013: semantic TTT provides useful directions but scalar objective fitting and checkpointing alone do not produce a reliable restoration field.
- **T014:** Sobolev derivative supervision establishes the optimization-field principle: restoration-useful derivatives matter more than scalar value fit.
- T018–T019: utility-aware hard geometry is viable on heterogeneous shifts.
- T020: adaptive geometry is not universally safe; simple direction-factorization repairs fail.
- **T021-A:** Sobolev beats the matched value-only control in RGB-SSIM on frozen fresh outputs with cluster-bootstrap CI wholly above zero.
- **T022-A:** untuned T014 transfers positively to real LOL-v2 validation but with a large absolute-quality gap.
- **T022-B:** saved-trajectory oracle headroom is too small for selector tuning to be the main solution.
- **T022-C:** widening only dark positive exposure to `+2 EV` produces a large real-validation gain (`+0.9567 dB`, `+0.05523 SSIM`).
- **T022-D:** doubling the exact trajectory budget to 80 updates does not improve PSNR/SSIM.
- **T023-A:** tiny real-domain Sobolev recalibration fits its training bank extremely well but fails the joint validation gate.
- **T024-A:** fair target-free baseline protocol is frozen, but recent matched SOTA coverage is incomplete.
- **T025-A:** a fixed reference oracle within the exact T022-C family reveals `+3.3164 dB` mean reachable PSNR headroom and near-universal gamma-lower-bound saturation; the remaining gap is not explained by checkpoint selection or step budget alone.

## Information-boundary rules

- Test-time adaptation/selection must never consume test labels, clean/normal-light targets, degradation masks/gain maps, condition IDs, annotations, semantic image IDs, reference gradients/Jacobians, oracle values, or evaluation metrics.
- Source/development/validation references may be used only in explicitly declared training, calibration, tuning, or diagnostic stages; they may not enter the per-image test-time decision path.
- Validation/test enhanced outputs and decisions must be finalized and persisted before their references or evaluation metrics are attached, except for explicitly labeled non-deployable reference-oracle diagnostics whose outputs must remain isolated from deployable TTT.
- Oracle diagnostics may motivate a **global validation-tuned hyperparameter choice**, but no per-image oracle state, target statistic, oracle step, reference gradient, or oracle score may be consumed by deployable inference/training unless a later task explicitly redefines a source-training split and preserves a separate holdout.
- Final benchmark test sets must remain isolated from hyperparameter/model selection; tuning belongs only on predeclared train/validation data.
- The official 100 LOL-v2 Real test pairs remain untouched through T025-A and must stay untouched until a final configuration is frozen.
- External baselines used in the main comparison must also be target-free at inference; target/reference-based brightness matching or selection is inadmissible in the main table.
- Fresh/test runs must fail closed on source/provenance/preparation binding mismatches.

## Interpretation

The paper remains image-enhancement-first. T014 supplies the central scientific contribution: learning a reference-free test-time **optimization field** through source-side derivative supervision. T021-A shows this advantage is not specific to MSE. T022–T026 are convergence work: identify the real-domain performance bottleneck, make only evidence-driven validation changes, then freeze a final configuration for strong target-free baseline comparison.

T025-A prevents two wrong conclusions. First, the current learned trajectory is not close to the best states already available in its own action family; there is more than `3 dB` mean PSNR headroom under a reference oracle. Second, because gamma and EV coordinates also concentrate near physical bounds, a richer/wider action state can still matter. The immediate deployable probe is therefore the single strongest boundary signal, active gamma lower `0.8→0.5`, with every other T022-C choice frozen.

The project continues on two lines: narrowly scoped truth/performance probes only when they directly determine tuning, and high-priority benchmark/SOTA convergence. After T026-A, priority returns to recent strong-baseline coverage unless the single bound change produces a clearly material improvement. Downstream detection is not required. Remaining paper-level gaps are competitive real-benchmark performance, strong matched recent-SOTA comparison, perceptual metrics, and efficiency/quality tradeoffs.

## Milestones

- T001–T013: completed mechanism/diagnostic sequence.
- **T014: COMPLETED — broad controlled/fresh Sobolev Ours-Core.**
- **T019-D: COMPLETED — heterogeneous adaptive-geometry fresh positive.**
- **T020-A–E: COMPLETED — universal geometry safety boundary diagnosed; repair route paused.**
- **T021-A: COMPLETED — frozen fresh RGB-SSIM transfer positive.**
- **T022-A: COMPLETED — leakage-safe untuned LOL-v2 validation anchor.**
- **T022-B: COMPLETED — selector headroom limited.**
- **T022-C: COMPLETED — materially positive dark-EV action-range probe (`+0.9567 dB`, `+0.05523 SSIM`).**
- **T022-D: COMPLETED — 80-step budget negative/insufficient (`-0.01384 dB`, `-0.001695 SSIM`).**
- **T023-A: COMPLETED — 16-pair real-domain Sobolev pilot negative/insufficient (`+0.2871 dB`, `-0.01374 SSIM`).**
- **T024-A: COMPLETED — target-free baseline protocol frozen; requested five-method roster has insufficient recent matched coverage.**
- **T025-A: COMPLETED — reference-oracle reachability shows `+3.3164 dB / +0.05582 SSIM` headroom inside frozen T022-C family; strong gamma/EV boundary pressure observed.**
- **T026-A: ACTIVE — target-free active-gamma lower-bound `0.8→0.5` validation probe.**

## Current open task

`T026-A — target-free active-gamma lower-bound probe` in `coordination/CHATGPT_TO_CODEX.md`.

On the exact frozen 100-image LOL-v2 Real validation split, run T022-C with only the active gamma lower bound widened from `0.8` to `0.5`; keep all other method settings frozen. Adapt using low-light images only, freeze all outputs/decisions before deploying normal-light references, compare against accepted T022-C, and stop after the single predeclared run. Do not use T025 oracle states or reference-derived per-image information, and do not touch the official LOL-v2 Real test set.