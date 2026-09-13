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

**T022-C is now a materially positive validation-only action-range result.** The single predeclared change widens only active dark-winner Region2 EV from `[0,+0.5]` to the existing physical maximum `[0,+2.0]`; frozen gate, Sobolev energy, gamma `[0.8,1.25]`, Adam `lr=0.03`, 40 updates, renderer, and learned-energy checkpoint rule remain unchanged. On the same fixed 100-image LOL-v2 validation split:

- mean PSNR: `9.2728689→10.2295540 dB` (`+0.9566851 dB`),
- mean RGB-SSIM: `0.2730060→0.3282315` (`+0.0552255`),
- median PSNR gain: `+0.7767744 dB`,
- median SSIM gain: `+0.0544302`.

All low-light inference artifacts were frozen before normal-light references were deployed, and target mutation/withholding leaves inference hashes unchanged. T022-C therefore identifies the original conservative positive-exposure range as a genuine real-domain bottleneck. It is **not** an official-test or SOTA claim, and absolute benchmark quality remains far below paper-ready competitive levels.

A new direct diagnostic emerges from T022-C: **88/100 validation images select the last available checkpoint, step 40.** This is stronger evidence for update-budget truncation than the aggregate bound-saturation percentages, which are partly confounded by inactive/collapsed coordinates. The immediate next experiment therefore changes only the trajectory budget from 40 to 80 while retaining the accepted T022-C action box and all other settings.

The project continues on two lines: narrowly scoped truth/mechanism work only when it directly determines tuning, and higher-priority benchmark/SOTA convergence. The benchmark line now has priority; repeated low-yield geometry patching remains out of scope.

## Best current methods

### Broad fresh-qualified Ours-Core

**T014 Sobolev Region2 TTT** — broad deployable method under the accepted controlled/fresh protocol.

### Current LOL-v2 validation candidate

**T022-C = T014 + dark-winner EV upper bound `+2.0`** — best current validation-only real-benchmark configuration; not yet final-test qualified.

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
- **T022-C:** widening only dark positive exposure to `+2 EV` produces a large real-validation gain (`+0.9567 dB`, `+0.05523 SSIM`), proving action-range mismatch was material.

## Information-boundary rules

- Test-time adaptation/selection must never consume test labels, clean/normal-light targets, degradation masks/gain maps, condition IDs, annotations, semantic image IDs, reference gradients/Jacobians, oracle values, or evaluation metrics.
- Source/development/validation references may be used only in explicitly declared training, calibration, tuning, or diagnostic stages; they may not enter the per-image test-time decision path.
- Validation/test enhanced outputs and decisions must be finalized and persisted before their references or evaluation metrics are attached.
- Final benchmark test sets must remain isolated from hyperparameter/model selection; tuning belongs only on predeclared train/validation data.
- The official 100 LOL-v2 Real test pairs remain untouched after T022-A/B/C and must stay untouched until a final configuration is frozen.
- Fresh/test runs must fail closed on source/provenance/preparation binding mismatches.

## Interpretation

The paper remains image-enhancement-first. T014 supplies the central scientific contribution: learning a reference-free test-time **optimization field** through source-side derivative supervision. T021-A shows this advantage is not specific to MSE. T022 now addresses paper-level benchmark convergence.

T022-C is the first large real-domain tuning gain and shows that the controlled-study action range was too conservative for severe real low light. However `10.23 dB / 0.328 SSIM` is still far from a competitive LOL-v2 endpoint, so incremental tuning must be tightly justified. Because 88/100 T022-C episodes select step 40, T022-D tests one final cheap trajectory hypothesis before deciding whether to invest in source/domain retraining and matched strong-baseline execution.

Downstream detection is not required for the enhancement-focused paper plan. Remaining paper-level gaps are competitive real-benchmark performance, strong matched baseline/SOTA comparison, perceptual metrics, and efficiency/quality tradeoffs.

## Milestones

- T001–T013: completed mechanism/diagnostic sequence.
- **T014: COMPLETED — broad controlled/fresh Sobolev Ours-Core.**
- **T019-D: COMPLETED — heterogeneous adaptive-geometry fresh positive.**
- **T020-A–E: COMPLETED — universal geometry safety boundary diagnosed; repair route paused.**
- **T021-A: COMPLETED — frozen fresh RGB-SSIM transfer positive.**
- **T022-A: COMPLETED — leakage-safe untuned LOL-v2 validation anchor.**
- **T022-B: COMPLETED — selector headroom limited.**
- **T022-C: COMPLETED — materially positive dark-EV action-range probe (`+0.9567 dB`, `+0.05523 SSIM`).**
- **T022-D: ACTIVE — exact 80-step trajectory-budget probe on the same validation split.**

## Current open task

`T022-D — LOL-v2 validation 80-step trajectory-budget probe` in `coordination/CHATGPT_TO_CODEX.md`.

Use the accepted T022-C configuration and same 100-image validation split. Change only `max_steps: 40→80`; keep all action bounds, gate, frozen Sobolev energy, optimizer/lr, renderer, and learned-energy checkpoint selection unchanged. Run low-light-only inference and freeze all outputs/decisions before reference evaluation. The official LOL-v2 Real test set remains untouched.
