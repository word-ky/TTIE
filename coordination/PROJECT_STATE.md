# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Enhancement**

## Core scientific question

Can a vision system adapt a compact spatial image-processing state per test image, without test labels or clean/normal-light targets, so that unknown and spatially heterogeneous degradation is corrected by a reference-free test-time optimization field?

## Current methods and real-domain status

**T014 Sobolev Region2 TTT remains the broad fresh-qualified Ours-Core.** It uses the frozen nuisance readout + clean-abstention gate, source-supervised Sobolev restoration energy, hard Region2 EV+gamma adaptation, projected label-free updates, and minimum predicted-energy checkpoint selection.

**T019 remains a heterogeneous-only adaptive-geometry extension.** T020 showed that the geometry selector is not universally safe on non-spatial inputs, so universal geometry repair remains paused.

On LOL-v2 Real development data, the official 100-pair test remains sealed. The accepted fixed-validation deployable base is **T026-A**, `11.1208764 dB / 0.3737918 RGB-SSIM`, with gamma lower bound `0.5` and a 40-step budget. T026-B showed that 80 steps add only `+0.1210491 dB / +0.0042526`, so 40 steps remain fixed.

**T036-A is the first fresh deployable action-space expansion that passes the joint aggregate gate.** On one deterministic reference-unused 100-pair training-development cohort, exact T026-A scores `10.2907831 / 0.3379463`; adding exactly one RGB-shared post-gamma common gain per Region2 under the unchanged T014 energy gives `11.2300401 / 0.3463023`. Paired mean gains are `+0.9392571 dB / +0.0083560 RGB-SSIM`, medians `+0.6439382 / +0.0053379`. This promotes common gain as a **fresh-qualified deployable aggregate-quality extension**, but not yet a per-image-safe or final method: PSNR declines on 29/100 images, SSIM on 40/100, and the worst PSNR loss is `5.61447 dB`.

## Capacity and optimization-field diagnosis

- **T028-A:** the exact T026-A EV+gamma family has a reference-only ceiling of `17.4599918 dB / 0.4317158`, proving substantial reachable headroom inside the original action family.
- **T029-A:** the frozen learned-energy gradient is strong early on real trajectories and collapses late. Step-10 median cosine is `0.7266` with `97%` positive-dot; step 30 is `-0.2206 / 22%`; step 40 is `-0.2453 / 24%`; selected states are `-0.2785 / 24%`. Reference gradients are diagnostic only.
- **T030-A:** a fixed learned-gradient self-reversal guard is negative on a fresh cohort.
- **T031-A:** source-support distance predicts gradient invalidity (AUROC `0.72047`, Spearman `-0.48230`) but is only an association.
- **T032-A:** turning support distance into a global trust-radius controller fails badly on a fresh cohort. Do not rescue it by threshold sweep.
- **T033-A:** target-free Retinexformer (`GT_mean=false`, self-ensemble off) reaches `21.4787864 / 0.7900612` on the frozen development validation split. Because the split comes from official LOL-v2 training data used by the released supervised checkpoint recipe, this is a training-exposed capacity anchor, not independent held-out SOTA evidence.
- **T034-A:** adding per-region RGB gains to the reference oracle raises the finite-budget ceiling to `19.9079495 / 0.4084354`, with an SSIM decline.
- **T035-A:** a matched RGB-shared common-gain reference oracle reaches `19.5530229 / 0.4084669`. It explains `85.50%` of T034's mean PSNR gain and `81.82%` of its median gain, showing that the dominant added capacity is a post-gamma common-intensity degree rather than independent chromatic WB. This is a finite-budget reachability result, not a certified global action-family ceiling; importantly, 85/100 T035 winning trajectories ended at the step-500 boundary.
- **T045-A:** the unchanged target-free-at-inference SNR-Aware exporter reaches `23.3963299 / 0.8237644` on the exact frozen development validation split (median `23.6347420 / 0.8485793`). This closes the missing strong-baseline row. Like T033, it is a **training-exposed development anchor**, not independent held-out SOTA evidence, because the released supervised checkpoint used LOL-v2 Real training data and this split was carved from that training set.

T045 makes the absolute quality gap impossible to ignore: current deployable Ours is far below the strong supervised-at-training anchors, and the existing finite-budget reference oracles also trail them on the same split. However, because T035 is visibly boundary-limited, the oracle gap cannot yet be cleanly attributed to renderer/action-family capacity. T046 therefore tests convergence before any new operator or field retraining is authorized.

## Unsafe-tail diagnosis through T044-A

**T037-A — limited/mixed late-selection headroom.** On the exact frozen T036 trajectories, reference-best minus selected common-gain PSNR is `+0.6832337 dB` mean but only `+0.0917852 dB` median. `18/29` prior PSNR-loss cases can be rescued to at least the T026 baseline by an earlier common state, but `11/29` cannot, and `28/100` images are genuinely reference-best at step 40. Severe overshoot exists in a subset, but a universal early-stop explanation is not supported.

**T038-A — gain-specific mismatch on the real loss subset.** At the exact 29 T036 PSNR-loss selected states, common-gain median cosine is `-0.3751342` with `5/29 = 17.24%` positive-dot; legacy EV+gamma is positive on `16/29 = 55.17%`, a `37.93 pp` gap. This is a real association on already-reference-used states, not evidence that gain is globally harmful: over all 100 selected states gain positive-dot is `54%`, and in the 71 non-loss cases it is `69.01%`.

**T039-A — no intrinsic near-identity source gain-tangent failure.** Across all `7,346` accepted T014 source states and gains `{0.75,1.00,1.25}`, legacy is `93.3545% / 0.955998` while gain is `85.2274% / 0.802533`; the `8.1271 pp / 0.153465` deficits are below the frozen `20 pp / 0.25` gates. At gain `1.25`, gain remains `90.7837% / 0.862192`.

**T040-A — no gain-specific source failure at high gain.** On the same 7,346 source states at gain `1.75`, legacy is `79.6496% / 0.770274` and gain is `73.0546% / 0.612039`; the same-gain gaps `6.5949 pp / 0.158235` remain below the gate. High gain degrades both source coordinate groups relative to gain `1.25`, so range extrapolation is real but shared rather than gain-specific.

**T041-A — strong matched-gain real selected-state deficit.** At fixed common gain `1.75`, source total-group alignment is `79.6634%` positive-dot with median cosine `0.763459`, while all 100 T036 real selected states give only `37% / -0.185848`, deficits of `42.6634 pp / 0.949307`. The all-100 collapse is dominated by legacy EV+gamma (`37% / -0.220698`) rather than an all-image gain failure (`71% / 0.309086`). This proves a real selected-state deficit beyond gain value alone, but does not by itself distinguish real-content shift from late state extrapolation.

**T042-A — late real legacy/feature-state extrapolation strongly supported as a local-field mechanism.** On those same 100 real images, same frozen Region2 gates, and the same fixed common gain `1.75`, replacing only the selected legacy EV+gamma coordinates with that image's accepted T036 step-10 legacy coordinates changes total alignment from `37% / -0.185848` to **`94% / 0.685277`**, improvements of `+57 pp / +0.871125`. The change is concentrated in legacy alignment: `37% / -0.220698` becomes `94% / 0.686961`; gain changes only from `71% / 0.309086` to `78% / 0.535115`. This strongly supports late legacy/feature-state extrapolation as a cause of local-field collapse and weakens a pure real-content explanation.

**T043-A — the matched-gain early-state quality bridge is negative.** Using only the already-frozen T041 `selected legacy + gain1.75` and T042 `step10 legacy + gain1.75` outputs, the step-10 substitution is substantially worse in actual reference quality: T041 baseline `12.0441547 dB / 0.3494575 RGB-SSIM` versus T042 candidate `10.1791756 / 0.3131188`. Paired candidate-minus-baseline means are **`-1.8649791 dB / -0.0363387`**, medians `-1.1739221 / -0.0323263`; only `15/100` images gain PSNR and `36/100` gain SSIM. The predeclared `+0.50 dB` and non-negative SSIM bridge gate fails decisively.

T043 is the key correction to T042: **directional validity and absolute state quality are not the same thing.** The early legacy state has a much more trustworthy local learned-energy direction, yet is farther from the restoration target in absolute quality. The late trajectory can therefore make useful net progress before the local field becomes unreliable. T042 does not justify a universal step-10 checkpoint, freeze, or early-stopping policy; T043 actively argues against that naive intervention.

**T044-A — raw low-only legacy excursion is not an unsafe-tail risk signal.** On the same accepted T036 100-image cohort, with `D_legacy` frozen before any metric/loss label was opened, treating larger step10-to-selected physical EV+gamma displacement as greater risk gives ROC-AUC **`0.2491501`** and Spearman(`D_legacy`, paired T036-minus-T026 `ΔPSNR`) **`+0.4838044`**. Both predeclared risk gates fail in the opposite direction. Loss cases have median `D_legacy = 0.120254`, versus `0.155492` for non-loss cases. Thus greater legacy movement is associated with greater net PSNR improvement on this cohort, not greater regression risk. Do not invert this score, sweep thresholds, or build a controller from it on the same reference-used data.

The T044 information boundary is accepted: all 100 scores and state/gate/code bindings were frozen before the accepted paired metric artifact was attached; Stage A opened no images, metrics, normals, reference gradients, or loss identities; Stage B opened no normals. Independent replay recomputed all scores, 2059 AUC pairs, Spearman, summaries and verdict with 210 scalar checks and max error `8.33e-17`. There were zero optimizer/state/selection changes, no fresh cohort, and no official-test access.

## Strongest current mechanism interpretation

1. The source-trained Sobolev field is useful in controlled/fresh settings and is often directionally correct early on real trajectories.
2. The original EV+gamma action family is too restrictive; common post-gamma intensity is a real missing degree of freedom.
3. T036 proves that common gain can improve fresh target-free aggregate quality without retraining the field.
4. Per-image safety and late real-domain field reliability remain major deployable limitations.
5. T037 rules out a universal checkpoint-timing explanation from reference-best quality alone.
6. T038 shows a gain-coordinate mismatch specifically in the 29 real PSNR-loss selected states.
7. T039/T040 reject the simplest source-side explanation: the gain tangent is not intrinsically broken near identity and does not become uniquely broken at source gain `1.75`.
8. T041 shows a dramatic all-100 real selected-state deficit at matched gain, dominated by legacy EV+gamma / feature-state alignment.
9. T042 shows that on the exact same real images, substituting the fixed early step-10 legacy state restores most directional validity. Late legacy/feature-state extrapolation is therefore a major causal contributor to **field reliability failure**.
10. T043 shows that the same early-state substitution is much worse in absolute image quality. Therefore late state extrapolation should not be equated with late-state quality degradation: the trajectory may move closer to the target before its local direction becomes untrustworthy.
11. T044 shows that raw legacy excursion magnitude does not resolve this decoupling. More parameter movement is actually associated with larger net quality improvement on the fixed cohort, so simple distance-to-step10 is not a safety proxy.
12. T045 quantifies a very large absolute gap to two strong supervised-at-training / target-free-at-inference development anchors: Retinexformer `21.4788 / 0.7901` and SNR-Aware `23.3963 / 0.8238`, versus T026-A `11.1209 / 0.3738`.
13. The finite-budget T034/T035 reference oracles also trail those anchors, so action-family capacity is now a serious concern alongside field reliability. But T035 cannot yet be treated as a hard ceiling because 85/100 winners terminate at its step-500 budget boundary.
14. The immediate method decision is therefore **convergence before redesign**: T046 tests whether the common-gain oracle gap is materially finite-budget-limited. Only after that result should we choose between field retraining and another action-space operator.

## Benchmark readiness

- T024-A froze the target-free comparison protocol. Reference-based brightness matching/selection is inadmissible.
- T027-A Retinexformer exporter is source/checkpoint bound and target-disabled.
- T027-B SNR-Aware exporter is source/checkpoint bound with the accepted native-pad16 protocol adaptation.
- T033-A Retinexformer development anchor is complete: `21.4787864 / 0.7900612`, with the training-exposure limitation above.
- **T045-A SNR-Aware development anchor is complete: `23.3963299 / 0.8237644`, under the exact same frozen split/metric convention and the same training-exposure caveat.**
- The first strong-baseline development table is therefore closed enough to guide method investment; do not add another baseline before resolving the Ours capacity/optimization question.
- **Official LOL-v2 Real test remains sealed** until Ours and all baseline protocols are frozen.

## Best current methods / ceilings

- **Broad fresh-qualified Ours-Core:** T014 Sobolev Region2 TTT.
- **Best fixed-validation deployable candidate:** T026-A, `11.1208764 / 0.3737918`.
- **Fresh-qualified real-domain action extension:** T036-A common-gain TTT, fresh paired `+0.9392571 dB / +0.0083560 RGB-SSIM` over exact T026-A on its new 100-pair cohort; unsafe tail unresolved.
- **Exact EV+gamma non-deployable finite-budget reachability:** T028-A, `17.4599918 / 0.4317158`.
- **Common-gain non-deployable finite-budget reachability:** T035-A, `19.5530229 / 0.4084669`; 85/100 winners at step 500, so convergence is unresolved.
- **Full-WB non-deployable finite-budget reachability:** T034-A, `19.9079495 / 0.4084354`.
- **Training-exposed supervised development anchors:** T033-A Retinexformer `21.4787864 / 0.7900612`; T045-A SNR-Aware `23.3963299 / 0.8237644`.
- **Heterogeneous-only geometry extension:** T019.

## Information-boundary rules

- Test-time adaptation and checkpoint selection must never consume test labels, clean/normal-light targets, reference gradients/Jacobians, oracle values, PSNR/SSIM, degradation masks/gain maps, condition IDs, annotations, or semantic image IDs.
- Validation/test outputs and decisions must be finalized and persisted before references or evaluation metrics are attached, except in explicitly isolated non-deployable reference diagnostics.
- Reference diagnostics may motivate only global research choices; no per-image oracle quantity may enter deployable inference.
- Source-training clean/reference targets may be used only for source-supervised training or isolated source-domain diagnostics; they are never admissible test-time inputs.
- External baselines admitted to the main comparison must be target-free at inference.
- Fresh/final benchmark sets must remain isolated from model/hyperparameter selection. The official LOL-v2 Real test is still untouched through T045-A.
- Fresh/test runs must fail closed on source/checkpoint/cohort/provenance mismatches.

## Milestones

T001–T013: controlled mechanism/diagnostic sequence. T014: broad controlled/fresh Sobolev Ours-Core. T019: heterogeneous adaptive geometry positive. T020: universal geometry route paused. T021: RGB-SSIM transfer positive. T022/T023: real validation/action-range and tiny recalibration sequence. T024: baseline protocol frozen. T025/T028: reachability diagnosed. T026-A: fixed-validation deployable base. T027-A/B: baseline exporters ready. T029: late-field mismatch. T030: self-reversal guard negative. T031: support-distance association positive. T032: support-radius controller negative. T033: Retinexformer development anchor. T034/T035: RGB-WB capacity resolved mainly to common intensity, but finite-budget convergence remains unresolved. T036: common gain fresh target-free aggregate positive with unsafe tail. T037: late-selection headroom limited/mixed. T038: real loss-subset gain mismatch. T039: near-range source gain-tangent deficit not supported. T040: high-range gain-specific source deficit not supported. T041: matched-gain all-100 real selected-state deficit strongly supported. T042: fixed step-10 legacy substitution restores directional field validity. T043: the same substitution is substantially worse in absolute quality, separating local field reliability from state quality. T044: fixed low-only legacy excursion magnitude fails as a regression-risk signal and points toward productive movement instead. **T045: SNR-Aware closes the missing strong-baseline development row at `23.3963 / 0.8238`, making the absolute Ours gap explicit while preserving the training-exposure caveat.**

## Current open task

**T046-A — common-gain oracle convergence extension** in `coordination/CHATGPT_TO_CODEX.md`.

On the same frozen 100-image validation split, reconstruct each accepted T035 winning common-gain state from low-only inputs, then run exactly one `REFERENCE_ORACLE_ONLY` continuation probe: fresh Adam at that frozen state, unchanged full RGB-MSE objective/lr `0.05`, exactly 1000 additional updates, one start per image. The sole verdict asks whether paired T046-minus-T035 PSNR improves by at least `+1.00 dB` mean **and** `+0.75 dB` median. No new operator, no field retraining, no hyperparameter sweep, no deployable change, and no official test.
