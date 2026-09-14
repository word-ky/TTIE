# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Enhancement**

## Core scientific question

Can a vision system adapt a compact spatial image-processing state per test image, without test labels or clean/normal-light targets, so that unknown and spatially heterogeneous degradation is corrected by a reference-free test-time optimization field?

## Current scientific state

**T014 Sobolev Region2 TTT remains the broad fresh-qualified Ours-Core.** It combines the frozen nuisance readout + clean-abstention gate, source-supervised Sobolev restoration energy, hard Region2 EV+gamma adaptation, projected label-free updates, and minimum predicted-energy checkpoint selection. T021-A later showed that the matched Sobolev-over-value-only advantage also transfers to frozen RGB-SSIM.

**T019 remains a heterogeneous-only fresh-qualified adaptive-geometry extension.** T020 showed that the geometry selector is not universally safe on non-spatial inputs; simple three-way/binary direction repairs did not close that gap. Universal geometry repair remains paused.

**T022-A established the first leakage-safe LOL-v2 Real validation anchor.** On the deterministic 100-pair validation split carved from the 689 official training pairs, untuned T014 improved raw mean PSNR `8.1097227→9.2728689 dB` and mean RGB-SSIM `0.1600228→0.2730060`, but absolute quality was weak. The official 100-pair LOL-v2 Real test set has remained untouched throughout tuning.

**T022-B ruled out checkpoint selection as the main real-domain bottleneck.** Over the 41 frozen T022-A checkpoints, the per-image PSNR oracle adds only `+0.1472 dB` and the SSIM oracle only `+0.00489`; the best global fixed step is effectively tied with learned selection.

**T022-C established that the original real-domain action range was too conservative.** Widening only active dark-winner EV from `[0,+0.5]` to `[0,+2.0]` improved validation from `9.2728689/0.2730060` to `10.2295540 dB / 0.3282315 SSIM`.

**T022-D closed simple budget extension for the old gamma-0.8 trajectory.** Changing only `40→80` updates from T022-C produced `-0.0138358 dB` PSNR and `-0.0016949` SSIM despite `82/100` images selecting step 80. Following that exact old trajectory longer was not useful.

**T023-A was a negative/insufficient 16-pair real-domain Sobolev recalibration pilot.** The new head fit its 656-state source bank nearly perfectly in gradient space but changed validation from T022-C to only `10.5166353 dB / 0.3144893 SSIM`: `+0.2870812 dB` PSNR and `-0.0137422` SSIM. Automatic scaling of that exact tiny-source recipe is not justified.

**T024-A froze the target-free baseline comparison protocol, but recent matched SOTA coverage remains incomplete.** Retinexformer without `GT_mean` and SNR-Aware have verifiable target-free LOL-v2 paths; Retinexformer `GT_mean` is rejected because reference statistics alter outputs. SG-LLIE's released checkpoint is NTIRE-bound rather than matched to its LOL-v2 paper result, LLFormer lacks a matched official LOL-v2 recipe/checkpoint, and Zero-DCE++ belongs to an external-SICE stratum. Strong recent baseline coverage must resume before final paper claims.

**T025-A completed a non-deployable reference-oracle reachability audit.** Inside the exact frozen T022-C gate + Region2 EV/gamma family, a fixed two-start/500-update reference oracle reached `13.545967 dB / 0.384052 SSIM` versus T022-C `10.229554 dB / 0.328231 SSIM`, a gap of `+3.316413 dB / +0.055821 SSIM`, with PSNR improved on `100/100`. This proved that useful states were being missed by the learned field. It also exposed strong action-bound pressure: `388/392` active gamma coordinates hit the old lower bound `0.8`, dark-winner EV concentrated near `+2`, and all bright-winner EV coordinates hit their allowed upper `0`. T025 remains strictly `REFERENCE_ORACLE_ONLY` and must never feed deployable TTT.

**T026-A is now the best deployable LOL-v2 validation candidate and a materially positive action-range result.** Changing exactly one setting from T022-C — active Region2 gamma lower `0.8→0.5`, with gamma upper `1.25`, EV boxes, gate, T014 energy, Region2 geometry, Adam `lr=0.03`, 40-step budget and learned-energy selection frozen — improved validation from `10.2295540 dB / 0.3282315 SSIM` to **`11.1208764 dB / 0.3737918 SSIM`**, i.e. **`+0.8913224 dB / +0.0455603 SSIM`**. The predeclared joint gate passed.

T026-A preserved the information boundary: the sole 100-image run was low-light-only, all outputs/decisions/trajectories froze before task-specific normal references were deployed, target mutation/withholding preserved inference artifact hashes, no T025 oracle quantity entered adaptation/selection, and the official LOL-v2 Real test set remained untouched. Real per-image regressions remain, so T026-A is validation-only rather than an official-test/SOTA claim.

A notable T026-A diagnostic is that `88/100` images still select step 40, but `0/392` active gamma coordinates hit the new lower `0.5` floor at selected/final states. Thus the improvement validates access to the wider gamma interior; it is not merely endpoint saturation. Because T022-D's negative 80-step result applied to the old gamma-0.8 trajectory, one matched 80-step probe on the newly promoted gamma-0.5 trajectory is now active before returning to benchmark/SOTA convergence.

## Best current methods

### Broad fresh-qualified Ours-Core

**T014 Sobolev Region2 TTT** — broad deployable method under the accepted controlled/fresh protocol.

### Current LOL-v2 validation candidate

**T026-A = T014 + dark-winner EV upper `+2.0` + active gamma lower `0.5`** — best current leakage-safe validation candidate: `11.1208764 dB / 0.3737918 SSIM` on the fixed 100-pair LOL-v2 Real validation split. Not official-test qualified.

### Heterogeneous-only geometry extension

**T019 = T014 + frozen utility-aware hard-boundary selector** — fresh-qualified only for the prescribed heterogeneous spatial protocol.

## Strongest controlled findings

- T001–T003: spatial ISP capacity helps heterogeneous shifts; simple priors/capacity alone are insufficient.
- T004–T007: zero-shot exposure signals are content-confounded; frozen source-trained nuisance readout + clean abstention is usable.
- T008–T013: semantic TTT provides useful directions but scalar objective fitting/checkpointing alone do not produce a reliable restoration field.
- **T014:** Sobolev derivative supervision establishes the optimization-field principle: restoration-useful derivatives matter more than scalar value fit.
- T018–T019: utility-aware hard geometry is viable on heterogeneous shifts.
- T020: adaptive geometry is not universally safe; simple direction-factorization repairs fail.
- **T021-A:** Sobolev beats the matched value-only control in RGB-SSIM on frozen fresh outputs with cluster-bootstrap CI wholly above zero.
- **T022-A:** untuned T014 transfers positively to real LOL-v2 validation but with a large absolute-quality gap.
- **T022-B:** saved-trajectory oracle headroom is too small for selector tuning to be the main solution.
- **T022-C:** widening dark positive exposure to `+2 EV` gives `+0.9567 dB / +0.05523 SSIM`.
- **T022-D:** doubling the old T022-C trajectory budget to 80 updates does not help.
- **T023-A:** tiny real-domain Sobolev recalibration fits its source bank but fails the joint validation gate.
- **T024-A:** fair target-free baseline protocol is frozen, but recent matched SOTA coverage is incomplete.
- **T025-A:** reference oracle reveals `+3.3164 dB` reachable PSNR headroom inside the old T022-C family and strong bound pressure.
- **T026-A:** widening active gamma lower `0.8→0.5` is materially positive on leakage-safe LOL-v2 validation (`+0.8913 dB / +0.04556 SSIM`).

## Information-boundary rules

- Test-time adaptation/selection must never consume test labels, clean/normal-light targets, degradation masks/gain maps, condition IDs, annotations, semantic image IDs, reference gradients/Jacobians, oracle values, or evaluation metrics.
- Source/development/validation references may be used only in explicitly declared training, calibration, tuning, or diagnostic stages; they may not enter the per-image test-time decision path.
- Validation/test enhanced outputs and decisions must be finalized and persisted before references or metrics are attached, except for explicitly labeled non-deployable reference-oracle diagnostics whose outputs must remain isolated from deployable TTT.
- Oracle diagnostics may motivate a **global validation-tuned hyperparameter choice**, but no per-image oracle state, target statistic, oracle step, reference gradient, or oracle score may be consumed by deployable inference/training unless a later task explicitly redefines a source-training split and preserves a separate holdout.
- Final benchmark test sets must remain isolated from hyperparameter/model selection; tuning belongs only on predeclared train/validation data.
- The official 100 LOL-v2 Real test pairs remain untouched through T026-A and must stay untouched until a final configuration is frozen.
- External baselines admitted to the main comparison must also be target-free at inference; target/reference-based brightness matching or selection is inadmissible.
- Fresh/test runs must fail closed on source/provenance/preparation binding mismatches.

## Interpretation

The paper remains image-enhancement-first. T014 supplies the central scientific contribution: learning a reference-free test-time **optimization field** through source-side derivative supervision. T021-A shows this is not MSE-specific. T022–T026 are convergence work: diagnose real-domain bottlenecks, make evidence-driven validation changes, then freeze a final method for strong target-free comparison.

T026-A materially narrows the real-benchmark gap and validates the strongest action-bound signal from T025-A, but absolute quality is still not sufficient for a final SOTA claim and recent matched-baseline coverage remains incomplete. One matched budget probe is justified because the promoted trajectory changed and still terminates at step 40 for 88/100 images. After that result, priority should return to strong recent baselines / benchmark convergence unless a clearly material new gain is established.

Downstream detection is not required. Remaining paper-level gaps are competitive real-benchmark performance, recent matched target-free SOTA comparison, perceptual metrics, and efficiency/quality tradeoffs.

## Milestones

- T001–T013: completed mechanism/diagnostic sequence.
- **T014: COMPLETED — broad controlled/fresh Sobolev Ours-Core.**
- **T019-D: COMPLETED — heterogeneous adaptive-geometry fresh positive.**
- **T020-A–E: COMPLETED — universal geometry safety boundary diagnosed; repair route paused.**
- **T021-A: COMPLETED — frozen fresh RGB-SSIM transfer positive.**
- **T022-A: COMPLETED — leakage-safe untuned LOL-v2 validation anchor.**
- **T022-B: COMPLETED — selector headroom limited.**
- **T022-C: COMPLETED — dark-EV action-range probe materially positive.**
- **T022-D: COMPLETED — old gamma-0.8 80-step budget negative/insufficient.**
- **T023-A: COMPLETED — 16-pair real-domain Sobolev pilot negative/insufficient.**
- **T024-A: COMPLETED — target-free baseline protocol frozen; recent matched coverage insufficient.**
- **T025-A: COMPLETED — reference-oracle reachability + bound-pressure audit.**
- **T026-A: COMPLETED — active gamma lower `0.8→0.5` materially positive (`+0.8913 dB / +0.04556 SSIM`).**
- **T026-B: ACTIVE — matched 80-step budget probe on promoted gamma-0.5 candidate.**

## Current open task

`T026-B — 80-step budget probe on promoted gamma-0.5 candidate` in `coordination/CHATGPT_TO_CODEX.md`.

On the exact frozen 100-image LOL-v2 Real validation split, run merged T026-A with only `max_steps: 40→80`; keep every other method setting fixed. Adapt using low-light images only, freeze all outputs/decisions/trajectories before deploying normal-light references, compare only against T026-A, and stop after the single predeclared run. Do not use T025 oracle states or any reference-derived per-image information, and do not touch the official LOL-v2 Real test set.