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

**T022-C is the current best LOL-v2 validation candidate and a materially positive action-range result.** Widening only active dark-winner Region2 EV from `[0,+0.5]` to `[0,+2.0]`, with all other scientific settings frozen, improved mean validation PSNR `9.2728689→10.2295540 dB` (`+0.9566851 dB`) and mean RGB-SSIM `0.2730060→0.3282315` (`+0.0552255`). All low-light inference artifacts were frozen before normal-light references were deployed. This identifies the original conservative positive-exposure range as a genuine real-domain bottleneck, but `10.23 dB / 0.328 SSIM` remains far from paper-ready competitive quality.

**T022-D is a leakage-safe validation-only negative and closes simple update-budget extension.** It changed only `max_steps: 40→80` from accepted T022-C. Mean PSNR changed `10.2295540→10.2157182 dB` (`-0.0138358 dB`) and mean RGB-SSIM `0.3282315→0.3265366` (`-0.0016949`). Despite this, `82/100` images selected step 80. Thus the learned energy continues preferring later states after true restoration quality has plateaued or worsened; longer-budget tuning is closed.

**T023-A is a leakage-safe validation-only negative/insufficient real-domain Sobolev recalibration pilot.** A deterministic 16-pair subset from the 589 non-validation LOL-v2 Real training pairs produced 656 EV2-matched source states. The new head fits that source bank extremely well (positive gradient-cosine fraction `1.0`, median cosine `0.9881`, direction loss `0.0168`), but validation quality changes from T022-C `10.2295540 dB / 0.3282315 SSIM` to `10.5166353 dB / 0.3144893 SSIM`: only `+0.2870812 dB` PSNR and `-0.0137422` SSIM. It therefore fails the predeclared joint gate and is not promoted. This shows that near-perfect source value/gradient fitting on a tiny real-source bank does not guarantee joint PSNR/SSIM generalization; the pilot does not justify automatic scaling of the same recipe to all 589 pairs.

Benchmark/SOTA convergence is now the immediate priority. Before any official-test run, T024-A freezes a target-free strong-baseline roster and common evaluation protocol so future comparison cannot drift across incompatible or target-assisted conventions. The official 100-pair LOL-v2 Real test set remains untouched.

## Best current methods

### Broad fresh-qualified Ours-Core

**T014 Sobolev Region2 TTT** — broad deployable method under the accepted controlled/fresh protocol.

### Current LOL-v2 validation candidate

**T022-C = T014 + dark-winner EV upper bound `+2.0`** — best current validation-only real-benchmark configuration; not yet official-test qualified.

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
- **T022-D:** doubling the exact trajectory budget to 80 updates does not improve PSNR/SSIM; simple budget truncation is rejected as the main remaining bottleneck.
- **T023-A:** a 16-pair real-domain Sobolev head achieves near-perfect source gradient fit but only `+0.2871 dB` validation PSNR while reducing SSIM by `0.01374`; small-source field recalibration does not justify full-data scaling by itself.

## Information-boundary rules

- Test-time adaptation/selection must never consume test labels, clean/normal-light targets, degradation masks/gain maps, condition IDs, annotations, semantic image IDs, reference gradients/Jacobians, oracle values, or evaluation metrics.
- Source/development/validation references may be used only in explicitly declared training, calibration, tuning, or diagnostic stages; they may not enter the per-image test-time decision path.
- Validation/test enhanced outputs and decisions must be finalized and persisted before their references or evaluation metrics are attached.
- Final benchmark test sets must remain isolated from hyperparameter/model selection; tuning belongs only on predeclared train/validation data.
- The official 100 LOL-v2 Real test pairs remain untouched after T022-A/B/C/D and T023-A and must stay untouched until a final configuration is frozen.
- External baselines used in the main comparison must also be target-free at inference; target/reference-based brightness matching or selection is not admissible in the main table.
- Fresh/test runs must fail closed on source/provenance/preparation binding mismatches.

## Interpretation

The paper remains image-enhancement-first. T014 supplies the central scientific contribution: learning a reference-free test-time **optimization field** through source-side derivative supervision. T021-A shows this advantage is not specific to MSE. T022–T023 address paper-level benchmark convergence.

T022-C showed that a controlled-study action box was too conservative for severe real low light. T022-D showed that simply following the same learned energy for longer does not improve restoration. T023-A then showed that a tiny real paired source bank can be fit almost perfectly in value/gradient space without delivering the required joint PSNR/SSIM validation transfer. That result argues against blind scaling or seed/loss tinkering and strengthens the need to quantify the competitive benchmark gap under a rigorously matched protocol before deciding the next architectural change.

The project continues on two lines: narrowly scoped truth/mechanism work only when it directly determines tuning, and higher-priority benchmark/SOTA convergence. Downstream detection is not required. Remaining paper-level gaps are competitive real-benchmark performance, strong matched baseline/SOTA comparison, perceptual metrics, and efficiency/quality tradeoffs.

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
- **T024-A: ACTIVE — strong-baseline provenance/fair-comparison protocol freeze.**

## Current open task

`T024-A — LOL-v2 strong-baseline protocol and fair-comparison freeze` in `coordination/CHATGPT_TO_CODEX.md`.

Audit exactly five named strong baseline families (Retinexformer, SG-LLIE, SNR-Aware, LLFormer, Zero-DCE++) using official sources; bind exact code/checkpoint/training/evaluation provenance; identify any target-assisted test normalization; classify validation and final-test fairness; and freeze one common target-free main-table metric protocol. Do not run the official LOL-v2 Real test, train baselines, or modify Ours in this cycle.