# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Enhancement**

## Core scientific question

Can a vision system adapt a compact spatial image-processing state per test image, without test labels or clean/normal-light targets, so that unknown and spatially heterogeneous degradation is corrected by a reference-free test-time optimization field?

## Current scientific state

**T014 Sobolev Region2 TTT remains the broad fresh-qualified Ours-Core.** It combines the frozen nuisance readout + clean-abstention gate, source-supervised Sobolev restoration energy, hard Region2 EV+gamma adaptation, projected label-free updates, and minimum predicted-energy checkpoint selection. T021-A showed the matched Sobolev-over-value-only advantage also transfers to frozen RGB-SSIM.

**T019 remains a heterogeneous-only fresh-qualified adaptive-geometry extension.** T020 showed that the geometry selector is not universally safe on non-spatial inputs; universal geometry repair remains paused.

### LOL-v2 Real convergence state

- **T022-A:** leakage-safe 100-pair validation anchor carved from the 689 official training pairs. Untuned T014 improves raw `8.1097227→9.2728689 dB` and RGB-SSIM `0.1600228→0.2730060`. Official 100-pair LOL-v2 Real test remains untouched.
- **T022-C:** widening only active dark-winner EV upper `[0,+0.5]→[0,+2.0]` improves validation to `10.2295540 dB / 0.3282315`.
- **T023-A:** 16-pair real Sobolev recalibration pilot is negative/insufficient (`10.5166353 / 0.3144893`).
- **T026-A:** best deployable validation candidate. Active gamma lower `0.8→0.5` yields **`11.1208764 dB / 0.3737918`**. All low-only outputs/decisions/trajectories froze before normal-reference evaluation.
- **T026-B:** 40→80 updates adds only `+0.1210491 dB / +0.0042526` while roughly doubling runtime; 40-step T026-A remains promoted.

### Reachability and optimization-field diagnosis

- **T028-A:** exact T026-A action-family reference oracle reaches **`17.4599918 dB / 0.4317158`**, paired `+6.3391154 dB` mean / `+5.7710854 dB` median PSNR versus deployable T026-A. This proves large within-family reachable headroom.
- **T029-A:** learned-energy/reference-MSE gradient alignment is strong early but collapses late. Step 10 median cosine `0.7266`, positive-dot `97%`; step 30 `-0.2206`, `22%`; step 40 `-0.2453`, `24%`; selected states `-0.2785`, `24%`. Reference gradients are diagnostic-only.
- **T030-A:** fixed learned-gradient self-reversal guard is negative on a fresh cohort: `10.1223728 / 0.3150107` versus baseline `10.3286568 / 0.3228188`.
- **T031-A:** nearest T014 source-support distance predicts reference-gradient invalidity (AUROC `0.720469831`, Spearman to cosine `-0.482299254`), but is diagnostic association rather than causal proof.
- **T032-A:** a source-only global 95th-percentile support radius fails badly on a new cohort: `8.444270861 / 0.206756358` versus T026-A `10.203176520 / 0.322824726`. Do not rescue this rule by threshold sweep.

### Strong development anchor and action-capacity diagnosis

**T033-A:** target-free Retinexformer (`GT_mean=false`, self-ensemble off) scores **`21.478786404 dB / 0.790061209`** on the frozen 100-image development validation split, paired `+10.357909987 dB / +0.416269384` versus T026-A. All outputs froze before normal decode. **Critical limitation:** these validation images are from the official LOL-v2 training set used to train the released supervised Retinexformer checkpoint, so this is a descriptive capacity anchor, not held-out SOTA evidence.

**T034-A:** adding per-region RGB gains to the reference-only oracle raises the ceiling to **`19.907949481 dB / 0.408435396`**, paired `+2.447957703 dB` mean / `+1.909172977 dB` median PSNR versus T028, with mean SSIM `-0.023280434`. The winning RGB gains are strongly common-mode, so T034 alone does not establish that chromatic correction is the main cause.

**T035-A resolves the T034 attribution: common post-gamma intensity explains most of the WB-family PSNR gain.** A matched oracle with exactly one RGB-shared gain per Region2 reaches **`19.553022889 dB / 0.408466942`**. Common-minus-T028 is **`+2.093031111 dB` mean / `+1.562056081 dB` median PSNR**, satisfying the frozen 75%-recovery thresholds and accounting for **85.50% of T034's aggregate mean PSNR gain and 81.82% of its median gain**. Full channel-specific WB adds only **`+0.354926591 dB` mean / `+0.182969702 dB` median PSNR** beyond common gain and does not recover the SSIM tradeoff (`-0.000031546` mean SSIM versus common). Therefore the dominant extra capacity in T034 is a **post-gamma common intensity degree**, with only a modest chromatic residual.

T035 is still `REFERENCE_ORACLE_ONLY`: validation normals are allowed only inside the isolated capacity audit; no oracle state, best step, gradient, target statistic, or metric may enter deployable TTT. Identity common gain reproduces T028 exactly, tied RGB gain reproduces T034 exactly, 100 outputs / 200 histories froze before quality scoring, independent metrics replay, and official test access is false.

### Scientific consequence through T035-A

The strongest current mechanism interpretation is:

1. T014's learned Sobolev optimization field is genuinely useful in controlled/fresh settings and often directionally useful early on real trajectories.
2. T026-A has both a **field/trajectory problem** and an **action-capacity problem**: its exact family has large oracle headroom, but the frozen field drifts into anti-restorative directions late.
3. Source-support distance tracks this failure but has not yielded a safe deployable controller.
4. A strong supervised development anchor exposes a large absolute-quality gap, though its training exposure prevents fair held-out ranking.
5. Expanding the action family with post-gamma gain materially raises PSNR capacity. T035 shows that most of T034's gain comes from a shared intensity coordinate rather than independent chromatic WB.
6. The next scientifically useful question is therefore deployable and narrow: **does adding exactly this common gain to T026-A help under the unchanged frozen label-free T014 energy on a genuinely new reference-unused cohort?** If yes, action-space expansion is immediately useful without field retraining; if no, oracle capacity is not being converted by the present optimization field.

## Benchmark readiness

- **T024-A:** target-free comparison protocol frozen. Retinexformer `GT_mean` is inadmissible; Retinexformer without `GT_mean` and SNR-Aware have admissible target-free paths.
- **T027-A:** Retinexformer exporter-ready and checkpoint/source-bound; target-disabled official adapter and TTIE exporter are float-identical on eight non-validation lows.
- **T027-B:** SNR-Aware exporter-ready and source/checkpoint-bound; native-pad16 adapter and TTIE exporter are float-identical on eight non-evaluation lows. This is a predeclared protocol adaptation, not the official resize-based `test4` reproduction.
- **T033-A:** Retinexformer development anchor completed with explicit training-exposure limitation.
- **SNR-Aware quality benchmark remains not yet run.**
- **Official LOL-v2 Real test remains sealed** until final Ours and baseline protocols are frozen.

## Best current methods / ceilings

- **Broad fresh-qualified Ours-Core:** T014 Sobolev Region2 TTT.
- **Current deployable LOL-v2 validation candidate:** T026-A, **`11.1208764 dB / 0.3737918`**.
- **Exact-family non-deployable ceiling:** T028-A, **`17.4599918 / 0.4317158`**.
- **Common-gain non-deployable ceiling:** T035-A, **`19.5530229 / 0.4084669`**.
- **Full-WB non-deployable ceiling:** T034-A, **`19.9079495 / 0.4084354`**.
- **Training-exposed supervised development anchor:** T033-A Retinexformer, **`21.4787864 / 0.7900612`**.
- **Heterogeneous-only geometry extension:** T019.

## Information-boundary rules

- Test-time adaptation/selection must never consume test labels, clean/normal-light targets, degradation masks/gain maps, condition IDs, annotations, semantic image IDs, reference gradients/Jacobians, oracle values, or evaluation metrics.
- Validation/test enhanced outputs and decisions must be finalized and persisted before references or metrics are attached, except explicitly isolated non-deployable reference diagnostics.
- Oracle/reference diagnostics may motivate only global research choices; no per-image oracle quantity may enter deployable inference.
- External baselines admitted to the main comparison must be target-free at inference; reference-based brightness matching or selection is inadmissible.
- Final benchmark test sets must remain isolated from model/hyperparameter selection. The official 100 LOL-v2 Real test pairs remain untouched through T035-A.
- Fresh/test runs must fail closed on source/checkpoint/cohort/provenance mismatches.

## Milestones

T001–T013 completed mechanism/diagnostic sequence. T014 broad controlled/fresh Sobolev Ours-Core completed. T019 heterogeneous adaptive geometry fresh positive. T020 universal-geometry repair route paused. T021 RGB-SSIM transfer positive. T022 real validation/action-range sequence completed. T023 tiny real recalibration negative. T024 baseline protocol frozen. T025/T028 oracle reachability diagnosed. T026-A promoted deployable validation candidate; T026-B longer budget negative. T027-A/B baseline exporters ready. T029 late-field mismatch diagnosed. T030 self-reversal guard negative. T031 support-distance diagnostic positive association. T032 support-radius controller negative. T033 Retinexformer development anchor completed. T034 WB-family capacity positive with SSIM tradeoff. **T035-A completed — common post-gamma intensity explains most of T034 PSNR gain.**

## Current open task

**T036-A — fresh target-free qualification of common-gain action expansion** in `coordination/CHATGPT_TO_CODEX.md`.

Run exact T026-A and exactly one common-gain-expanded variant on one new deterministic 100-pair training-development cohort whose normals have never previously been accessed. Keep the frozen T014 energy/gate/Region2/optimizer/40-step predicted-energy selector unchanged; the sole new fast coordinate is one `[0.5,2.0]` post-gamma RGB-shared gain per Region2. Freeze both methods' low-only outputs/decisions/trajectories before deploying normals. Promotion requires paired mean `ΔPSNR >= +0.30 dB` and `ΔRGB-SSIM >= 0`; otherwise stop negative/insufficient. No tuning, second cohort, per-channel WB, retraining, or official-test access.
