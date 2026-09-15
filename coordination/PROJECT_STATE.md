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
- **T034-A:** adding per-region RGB gains to the reference oracle raises the ceiling to `19.9079495 / 0.4084354`, with an SSIM decline.
- **T035-A:** a matched RGB-shared common-gain oracle reaches `19.5530229 / 0.4084669`. It explains `85.50%` of T034's mean PSNR gain and `81.82%` of its median gain, showing that the dominant added capacity is a post-gamma common-intensity degree rather than independent chromatic WB.

## Unsafe-tail diagnosis through T042-A

**T037-A — limited/mixed late-selection headroom.** On the exact frozen T036 trajectories, reference-best minus selected common-gain PSNR is `+0.6832337 dB` mean but only `+0.0917852 dB` median. `18/29` prior PSNR-loss cases can be rescued to at least the T026 baseline by an earlier common state, but `11/29` cannot, and `28/100` images are genuinely reference-best at step 40. Severe overshoot exists in a subset, but a universal early-stop explanation is not supported.

**T038-A — gain-specific mismatch on the real loss subset.** At the exact 29 T036 PSNR-loss selected states, common-gain median cosine is `-0.3751342` with `5/29 = 17.24%` positive-dot; legacy EV+gamma is positive on `16/29 = 55.17%`, a `37.93 pp` gap. This is a real association on already-reference-used states, not evidence that gain is globally harmful: over all 100 selected states gain positive-dot is `54%`, and in the 71 non-loss cases it is `69.01%`.

**T039-A — no intrinsic near-identity source gain-tangent failure.** Across all `7,346` accepted T014 source states and gains `{0.75,1.00,1.25}`, legacy is `93.3545% / 0.955998` while gain is `85.2274% / 0.802533`; the `8.1271 pp / 0.153465` deficits are below the frozen `20 pp / 0.25` gates. At gain `1.25`, gain remains `90.7837% / 0.862192`.

**T040-A — no gain-specific source failure at high gain.** On the same 7,346 source states at gain `1.75`, legacy is `79.6496% / 0.770274` and gain is `73.0546% / 0.612039`; the same-gain gaps `6.5949 pp / 0.158235` remain below the gate. High gain degrades both source coordinate groups relative to gain `1.25`, so range extrapolation is real but shared rather than gain-specific.

**T041-A — strong matched-gain real selected-state deficit.** At fixed common gain `1.75`, source total-group alignment is `79.6634%` positive-dot with median cosine `0.763459`, while all 100 T036 real selected states give only `37% / -0.185848`, deficits of `42.6634 pp / 0.949307`. The all-100 collapse is dominated by legacy EV+gamma (`37% / -0.220698`) rather than an all-image gain failure (`71% / 0.309086`). This proves a real selected-state deficit beyond gain value alone, but does not by itself distinguish real-content shift from late state extrapolation.

**T042-A — late real legacy/feature-state extrapolation strongly supported.** On those same 100 real images, same frozen Region2 gates, and the same fixed common gain `1.75`, replacing only the selected legacy EV+gamma coordinates with that image's accepted T036 step-10 legacy coordinates changes total alignment from `37% / -0.185848` to **`94% / 0.685277`**, improvements of `+57 pp / +0.871125`. The predeclared `+20 pp / +0.25` gate passes decisively. The change is concentrated in legacy alignment: `37% / -0.220698` becomes `94% / 0.686961`; gain changes only from `71% / 0.309086` to `78% / 0.535115`.

T042 materially weakens a pure real-image-content explanation for T041. At matched gain and on the exact same real images, the early legacy state largely restores the source-like directional field. The strongest supported mechanism is therefore that **late legacy EV+gamma / feature-state extrapolation is a major contributor to the real selected-state collapse**, with an additional loss-subset-specific gain interaction from T038.

T042 still does **not** qualify a universal step-10 checkpoint or early-stopping policy. It is a `REFERENCE_GRADIENT_DIAGNOSTIC_ONLY` local-gradient comparison at fixed gain, not a PSNR/SSIM quality result or a deployable intervention. Positive local gradient alignment is not itself proof that the substituted state has better image quality.

The information boundary is accepted: 100 low-only probes were frozen before any normal/reference or prior reference-derived baseline file was opened; step-10 state slices/gates were hash-bound; all Stage-B outputs were bit-exact; there were zero optimizer updates, selection changes, fresh cohorts, deployable edits, or official-test accesses. Independent replay checked 2,251 scalars with max error `4.44e-16`.

## Strongest current mechanism interpretation

1. The source-trained Sobolev field is useful in controlled/fresh settings and is often directionally correct early on real trajectories.
2. The original EV+gamma action family is too restrictive; common post-gamma intensity is a real missing degree of freedom.
3. T036 proves that common gain can improve fresh target-free aggregate quality without retraining the field.
4. The remaining problem is **per-image safety and late real-domain field reliability**, not lack of aggregate capacity.
5. T037 rules out a universal checkpoint-timing explanation from reference-best quality alone.
6. T038 shows a gain-coordinate mismatch specifically in the 29 real PSNR-loss selected states.
7. T039/T040 reject the simplest source-side explanation: the gain tangent is not intrinsically broken near identity and does not become uniquely broken at source gain `1.75`.
8. T041 shows a dramatic all-100 real selected-state deficit at matched gain, dominated by legacy EV+gamma / feature-state alignment.
9. T042 shows that on the exact same real images, substituting the fixed early step-10 legacy state restores most directional validity. **Late legacy/feature-state extrapolation is therefore a major causal contributor to the field failure, not merely correlated real-image content.**
10. What remains unproven is the bridge from restored local gradient alignment to actual restoration quality at the matched frozen states. T043 tests exactly that before any controller or deployable repair is designed.

## Benchmark readiness

- T024-A froze the target-free comparison protocol. Reference-based brightness matching/selection is inadmissible.
- T027-A Retinexformer exporter is source/checkpoint bound and target-disabled.
- T027-B SNR-Aware exporter is source/checkpoint bound with the accepted native-pad16 protocol adaptation.
- T033-A Retinexformer development anchor is complete with the training-exposure limitation above.
- SNR-Aware quality benchmarking has not yet been run.
- **Official LOL-v2 Real test remains sealed** until Ours and all baseline protocols are frozen.

## Best current methods / ceilings

- **Broad fresh-qualified Ours-Core:** T014 Sobolev Region2 TTT.
- **Best fixed-validation deployable candidate:** T026-A, `11.1208764 / 0.3737918`.
- **Fresh-qualified real-domain action extension:** T036-A common-gain TTT, fresh paired `+0.9392571 dB / +0.0083560 RGB-SSIM` over exact T026-A on its new 100-pair cohort; unsafe tail unresolved.
- **Exact EV+gamma non-deployable ceiling:** T028-A, `17.4599918 / 0.4317158`.
- **Common-gain non-deployable ceiling:** T035-A, `19.5530229 / 0.4084669`.
- **Full-WB non-deployable ceiling:** T034-A, `19.9079495 / 0.4084354`.
- **Training-exposed supervised development anchor:** T033-A Retinexformer, `21.4787864 / 0.7900612`.
- **Heterogeneous-only geometry extension:** T019.

## Information-boundary rules

- Test-time adaptation and checkpoint selection must never consume test labels, clean/normal-light targets, reference gradients/Jacobians, oracle values, PSNR/SSIM, degradation masks/gain maps, condition IDs, annotations, or semantic image IDs.
- Validation/test outputs and decisions must be finalized and persisted before references or evaluation metrics are attached, except in explicitly isolated non-deployable reference diagnostics.
- Reference diagnostics may motivate only global research choices; no per-image oracle quantity may enter deployable inference.
- Source-training clean/reference targets may be used only for source-supervised training or isolated source-domain diagnostics; they are never admissible test-time inputs.
- External baselines admitted to the main comparison must be target-free at inference.
- Fresh/final benchmark sets must remain isolated from model/hyperparameter selection. The official LOL-v2 Real test is still untouched through T042-A.
- Fresh/test runs must fail closed on source/checkpoint/cohort/provenance mismatches.

## Milestones

T001–T013: controlled mechanism/diagnostic sequence. T014: broad controlled/fresh Sobolev Ours-Core. T019: heterogeneous adaptive geometry positive. T020: universal geometry route paused. T021: RGB-SSIM transfer positive. T022/T023: real validation/action-range and tiny recalibration sequence. T024: baseline protocol frozen. T025/T028: reachability diagnosed. T026-A: fixed-validation deployable base. T027-A/B: baseline exporters ready. T029: late-field mismatch. T030: self-reversal guard negative. T031: support-distance association positive. T032: support-radius controller negative. T033: Retinexformer development anchor. T034/T035: RGB-WB capacity resolved mainly to common intensity. T036: common gain fresh target-free aggregate positive with unsafe tail. T037: late-selection headroom limited/mixed. T038: real loss-subset gain mismatch. T039: near-range source gain-tangent deficit not supported. T040: high-range gain-specific source deficit not supported. T041: matched-gain all-100 real selected-state deficit strongly supported. **T042: same-image fixed-gain step-10 legacy substitution restores directional field validity, strongly supporting late legacy/feature-state extrapolation.**

## Current open task

**T043-A — matched-gain frozen-output quality bridge** in `coordination/CHATGPT_TO_CODEX.md`.

Use only the already-frozen T041 `selected legacy + gain1.75` and T042 `step10 legacy + gain1.75` outputs for the same 100 already-reference-used images. Bind both output sets before any normal opens, then evaluate with the accepted T026/T036 PSNR and RGB-SSIM convention. The single gate is mean paired `ΔPSNR >= +0.50 dB` and mean paired `ΔRGB-SSIM >= 0`. This is evaluation-only: no optimizer, no selection, no controller, no fresh cohort, and no official-test access.
