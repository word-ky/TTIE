# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Enhancement**

## Core scientific question

Can a vision system adapt a compact spatial image-processing state per test image, without test labels or clean/normal-light targets, so that unknown and spatially heterogeneous degradation is corrected by a reference-free test-time optimization field?

## Current methods and real-domain status

**T014 Sobolev Region2 TTT remains the broad fresh-qualified Ours-Core.** It uses the frozen nuisance readout + clean-abstention gate, source-supervised Sobolev restoration energy, hard Region2 EV+gamma adaptation, projected label-free updates, and minimum predicted-energy checkpoint selection.

**T019 remains a heterogeneous-only adaptive-geometry extension.** T020 showed that the geometry selector is not universally safe on non-spatial inputs, so universal geometry repair remains paused.

On LOL-v2 Real development data, the official 100-pair test remains sealed. The accepted fixed-validation deployable base is **T026-A**, which reaches `11.1208764 dB / 0.3737918 RGB-SSIM` with active gamma lower bound `0.5` and a 40-step budget. T026-B showed that doubling the budget to 80 steps adds only `+0.1210491 dB / +0.0042526`, so 40 steps remain fixed.

**T036-A is the first fresh deployable action-space expansion that passes the joint aggregate gate.** On one deterministic reference-unused 100-pair training-development cohort, exact T026-A scores `10.2907831 / 0.3379463`; adding exactly one RGB-shared post-gamma common gain per Region2 under the unchanged T014 energy gives `11.2300401 / 0.3463023`. Paired mean gains are `+0.9392571 dB / +0.0083560 RGB-SSIM`, medians `+0.6439382 / +0.0053379`. This promotes the common-gain variant as a **fresh-qualified deployable aggregate-quality extension**, but not yet a per-image-safe or final method: PSNR declines on 29/100 images, SSIM on 40/100, and the worst PSNR loss is `5.61447 dB`.

## Capacity and optimization-field diagnosis

- **T028-A:** the exact T026-A EV+gamma family has a reference-only ceiling of `17.4599918 dB / 0.4317158`, proving substantial reachable headroom inside the original action family.
- **T029-A:** the frozen learned-energy gradient is strong early on real trajectories and collapses late. Step-10 median cosine is `0.7266` with `97%` positive-dot; step 30 is `-0.2206 / 22%`; step 40 is `-0.2453 / 24%`; selected states are `-0.2785 / 24%`. Reference gradients are diagnostic only.
- **T030-A:** a fixed learned-gradient self-reversal guard is negative on a fresh cohort.
- **T031-A:** source-support distance predicts gradient invalidity (AUROC `0.72047`, Spearman `-0.48230`) but is only an association.
- **T032-A:** turning that support distance into a global trust-radius controller fails badly on a fresh cohort. Do not rescue it by threshold sweep.
- **T033-A:** target-free Retinexformer (`GT_mean=false`, self-ensemble off) reaches `21.4787864 / 0.7900612` on the frozen development validation split. Because that split comes from the official LOL-v2 training data used by the released supervised checkpoint recipe, this is a training-exposed capacity anchor, not independent held-out SOTA evidence.
- **T034-A:** adding per-region RGB gains to the reference oracle raises the ceiling to `19.9079495 / 0.4084354`, but with an SSIM decline.
- **T035-A:** a matched RGB-shared common-gain oracle reaches `19.5530229 / 0.4084669`. It explains `85.50%` of T034's mean PSNR gain and `81.82%` of its median gain, showing that the dominant added capacity is a post-gamma common-intensity degree rather than independent chromatic WB.

## Unsafe-tail diagnosis through T040-A

**T037-A — limited/mixed late-selection headroom.** On the exact frozen T036 trajectories, reference-best minus selected common-gain PSNR is `+0.6832337 dB` mean but only `+0.0917852 dB` median. `18/29` prior PSNR-loss cases can be rescued to at least the T026 baseline by an earlier common state, but `11/29` cannot, and `28/100` images are genuinely reference-best at step 40. Severe overshoot exists in a subset, but a universal early-stop explanation is not supported.

**T038-A — gain-specific mismatch on the real loss subset.** At the exact 29 T036 PSNR-loss selected states, the common-gain coordinate has median cosine `-0.3751342` and positive-dot `5/29 = 17.24%`; legacy EV+gamma is positive on `16/29 = 55.17%`, a `37.93` percentage-point gap. This is a real association at those already-reference-used states, not evidence that gain is globally harmful: over all 100 selected states, gain positive-dot is `54%`, and in the 71 non-loss cases it is `69.01%`. T038 is `REFERENCE_GRADIENT_DIAGNOSTIC_ONLY`.

**T039-A — no intrinsic near-identity source gain-tangent failure.** Across all `7,346` accepted T014 source states and gains `{0.75,1.00,1.25}`, legacy positive-dot is `93.3545%` with median cosine `0.955998`, while gain is `85.2274% / 0.802533`. The legacy-minus-gain gaps (`8.1271 pp`, `0.153465`) are below the frozen `20 pp / 0.25` deficit gates. At gain `1.25`, gain remains `90.7837%` positive-dot with median cosine `0.862192`.

**T040-A — no gain-specific source failure even at the T036 high-gain range.** T040 reuses all 7,346 accepted source states and probes only common gain `1.50` and `1.75`. At gain `1.75`, legacy EV+gamma is `79.6496%` positive-dot with median cosine `0.770274`; gain is `73.0546% / 0.612039`. The same-gain legacy-minus-gain gaps are only `6.5949 pp / 0.158235`, so the frozen `20 pp / 0.25` gate is not met: **source high-gain tangent deficit is not supported**.

T040 also reveals a real but shared high-gain degradation. Relative to the frozen source gain=`1.25` baseline, the gain coordinate drops `17.729 pp` in positive-dot and `0.250152` in median cosine by gain=`1.75`, and legacy alignment declines as well. Therefore high gain is not harmless, but it does not uniquely break the gain coordinate on source. This materially weakens the explanation that T038 is merely coordinate-range extrapolation.

T040's information boundary is accepted: all `14,692` learned gradients were frozen before any of the 80 authorized source clean JPGs were opened; Stage B reproduced all output hashes, made zero optimizer/selection changes, and accessed no LOL-v2 data; independent replay checked `322,753` scalars with max error `1.776e-15`. T040 is `SOURCE_REFERENCE_GRADIENT_DIAGNOSTIC_ONLY` and changes no deployable inference.

## Strongest current mechanism interpretation

1. The source-trained Sobolev field is genuinely useful in controlled/fresh settings and is often directionally correct early on real trajectories.
2. The original EV+gamma action family is too restrictive; common post-gamma intensity is a real missing degree of freedom.
3. T036 proves that this extra common-gain capacity can improve fresh target-free aggregate quality without retraining the field.
4. The remaining problem is **per-image safety and late real-domain field reliability**, not lack of aggregate capacity.
5. T037 rules out a universal checkpoint-timing explanation.
6. T038 shows a strong gain-coordinate mismatch specifically in the 29 real PSNR-loss selected states.
7. T039 and T040 jointly reject the simplest source-side explanation: the gain tangent is not intrinsically broken near identity and does not become uniquely broken on source at gain `1.75`.
8. High gain still degrades alignment for both source coordinate groups, so range extrapolation contributes globally, but it is insufficient to explain the real loss-subset pattern.
9. The strongest remaining hypothesis is a **real target/selected-state distribution effect beyond gain value alone**. This must be phrased carefully: current evidence does not yet separate image-domain content shift from the real trajectory's legacy EV/gamma and learned-feature state.

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
- Fresh/final benchmark sets must remain isolated from model/hyperparameter selection. The official LOL-v2 Real test is still untouched through T040-A.
- Fresh/test runs must fail closed on source/checkpoint/cohort/provenance mismatches.

## Milestones

T001–T013: controlled mechanism/diagnostic sequence. T014: broad controlled/fresh Sobolev Ours-Core. T019: heterogeneous adaptive geometry positive. T020: universal geometry route paused. T021: RGB-SSIM transfer positive. T022/T023: real validation/action-range and tiny recalibration sequence. T024: baseline protocol frozen. T025/T028: reachability diagnosed. T026-A: fixed-validation deployable base. T027-A/B: baseline exporters ready. T029: late-field mismatch. T030: self-reversal guard negative. T031: support-distance association positive. T032: support-radius controller negative. T033: Retinexformer development anchor. T034/T035: RGB-WB capacity resolved mainly to common intensity. T036: common gain fresh target-free aggregate positive with unsafe tail. T037: late-selection headroom limited/mixed. T038: real loss-subset gain mismatch. T039: near-range source gain-tangent deficit not supported. **T040: high-range gain-specific source deficit also not supported; shared high-gain degradation exists, leaving real selected-state distribution as the stronger remaining mechanism.**

## Current open task

**T041-A — fixed-gain real selected-state alignment audit** in `coordination/CHATGPT_TO_CODEX.md`.

Reuse exactly the 100 already-reference-used T036 common-gain selected states. With each image's selected legacy EV/gamma and gate frozen, create only fixed common-gain probes `{1.25,1.75}`. Freeze all 200 low-only learned gradients and output hashes before opening any normal/reference or prior metric/loss-case file; then compute isolated RGB-MSE reference gradients on the same already-used normals. Compare the real all-100 total-group alignment at gain `1.75` against the accepted T040 source gain=`1.75` baseline under one predeclared `20 pp / 0.25` gate. This is diagnostic only: no optimizer, controller, fresh cohort, deployable change, or official-test access.
