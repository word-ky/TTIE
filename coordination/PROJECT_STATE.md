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
- **T034-A:** adding per-region RGB gains to the reference oracle raises finite-budget reachability to `19.9079495 / 0.4084354`, with an SSIM decline.
- **T035-A:** a matched RGB-shared common-gain reference oracle reaches `19.5530229 / 0.4084669`. It explains `85.50%` of T034's mean PSNR gain and `81.82%` of its median gain, showing that the dominant added capacity is a post-gamma common-intensity degree rather than independent chromatic WB.
- **T045-A:** the unchanged target-free-at-inference SNR-Aware exporter reaches `23.3963299 / 0.8237644` on the exact frozen development validation split (median `23.6347420 / 0.8485793`). Like T033, this is a **training-exposed development anchor**, not independent held-out SOTA evidence.
- **T046-A:** material T035 underconvergence is **not supported** under the predeclared fixed extension. Starting from exact T035 winners and adding +1000 fresh-Adam reference-MSE updates raises mean PSNR only from `19.5530228894` to `19.5732278418` and median only from `18.8982131190` to `18.9220694331`: paired `+0.0202049524 dB` mean / `+0.0123440643 dB` median, far below the frozen `+1.00 / +0.75 dB` gate. Mean RGB-SSIM improves only `+0.0009001422`; 49/100 winners remain at the new boundary, so global optimality is not certified.
- **T047-A:** a bounded RGB-shared additive lift after common gain is broadly helpful but **not materially sufficient under the frozen marginal-capacity gate**. With every T046 EV/gamma/common-gain coordinate frozen, mean PSNR rises from `19.5732278418` to `20.0559720402 dB` and median from `18.9220694331` to `19.2949802533`, paired `+0.4827441984 / +0.1331822579 dB`, missing the predeclared `+0.50 / +0.25 dB` gate. Mean RGB-SSIM rises `+0.0049064412`; PSNR improves on `100/100` and SSIM on `81/100`.
- **T048-A:** post-gamma affine coupling is **materially supported**. Starting from exact T047 states with EV/gamma frozen, jointly re-optimizing only accepted common gain + additive lift raises mean PSNR from `20.0559720402` to **`21.0649799860 dB`** and median from `19.2949802533` to `21.6436267060`. Paired T048-minus-T047 PSNR is **`+1.0090079459 dB` mean / `+0.5583504734 dB` median**, passing the frozen `+0.50 / +0.25 dB` gate. Mean RGB-SSIM rises to **`0.4566638129`**; PSNR improves on `100/100`, SSIM on `97/100`. This proves scale/offset interaction matters. It is not a certified affine ceiling: 87/392 selected lift coordinates hit `+0.20`, and 12/100 winners are at step 500.
- **T049-A:** an RGB-shared 8-segment monotonic regional tone LUT is **useful but does not pass the frozen multi-dB gate at 500 steps**. With all T048 EV/gamma/gain/lift coordinates frozen, reference-only mean PSNR/RGB-SSIM rises from `21.0649799860 / 0.4566638129` to **`21.8722552498 / 0.5047578705`**. Paired PSNR is **`+0.8072752637 dB` mean / `+0.5988926101 dB` median**; mean SSIM is `+0.0480940576`; PSNR improves on `100/100`. The predeclared `+2.00/+1.00 dB` SOTA-scale gate fails. `91/100` winners are exactly at step 500, motivating one fixed convergence extension rather than a parameter sweep.
- **T050-A:** material monotonic-tone underconvergence is **not supported**. Continuing only the accepted T049 LUT `q` for an additional fixed 1000 fresh-Adam updates raises mean PSNR only from `21.8722552498` to **`21.8742627274 dB`** and median from `22.6465425426` to `22.6476070829`: paired **`+0.0020074776 dB` mean / `+0.0006543315 dB` median**, far below the frozen `+1.00/+0.50 dB` gate. Mean RGB-SSIM rises only `+0.0004028470` to **`0.5051607175`**. PSNR is 99 win / 1 equal / 0 loss, but the total quality increment is negligible. `50/100` winners remain at step 1000, so this is not a global-optimum certificate; nevertheless another pure tone LR/step/budget extension is not justified.

T046–T050 sharpen the capacity story. The original common-gain family is not rescued by more steps; affine scale/offset coupling adds a real `+1.0 dB`; nonlinear monotonic tone adds another broad `+0.81 dB` mean and a large SSIM increment; and T050 shows that simply extending the same tone optimization another 1000 steps adds essentially nothing. The accepted compact affine+tone reference reachability is therefore about `21.8743 / 0.5052`: already above the Retinexformer training-exposed PSNR anchor by `0.3955 dB`, but still `1.5221 dB` below SNR-Aware in PSNR and roughly `0.3186` below it in RGB-SSIM. The next capacity investment must introduce a genuinely new intra-region spatial illumination degree of freedom rather than another affine/tone reparameterization or optimizer-budget rescue.

## Unsafe-tail diagnosis through T044-A

**T037-A — limited/mixed late-selection headroom.** On exact frozen T036 trajectories, reference-best minus selected common-gain PSNR is `+0.6832337 dB` mean but only `+0.0917852 dB` median. `18/29` prior PSNR-loss cases can be rescued to at least the T026 baseline by an earlier common state, but `11/29` cannot, and `28/100` images are genuinely reference-best at step 40.

**T038-A — gain-specific mismatch on the real loss subset.** At the exact 29 T036 PSNR-loss selected states, common-gain median cosine is `-0.3751342` with `5/29 = 17.24%` positive-dot; legacy EV+gamma is positive on `16/29 = 55.17%`, a `37.93 pp` gap. Over all 100 selected states gain positive-dot is `54%`, and in the 71 non-loss cases it is `69.01%`.

**T039-A — no intrinsic near-identity source gain-tangent failure.** Across all `7,346` accepted T014 source states and gains `{0.75,1.00,1.25}`, legacy is `93.3545% / 0.955998` while gain is `85.2274% / 0.802533`; the `8.1271 pp / 0.153465` deficits are below the frozen `20 pp / 0.25` gates.

**T040-A — no gain-specific source failure at high gain.** At source gain `1.75`, legacy is `79.6496% / 0.770274` and gain is `73.0546% / 0.612039`; same-gain gaps `6.5949 pp / 0.158235` remain below gate. High gain degrades both groups rather than uniquely breaking gain.

**T041-A — strong matched-gain real selected-state deficit.** At fixed gain `1.75`, source total-group alignment is `79.6634% / 0.763459`, while all 100 T036 real selected states give only `37% / -0.185848`; the collapse is dominated by legacy EV+gamma (`37% / -0.220698`) rather than an all-image gain failure (`71% / 0.309086`).

**T042-A — late real legacy/feature-state extrapolation strongly supported as a local-field mechanism.** On the same real images and fixed gain `1.75`, replacing only selected legacy EV+gamma with each image's fixed step-10 legacy coordinates changes total alignment from `37% / -0.185848` to `94% / 0.685277`. Legacy alignment changes from `37% / -0.220698` to `94% / 0.686961`.

**T043-A — the matched-gain early-state quality bridge is negative.** The step-10 substitution that restores directional validity is substantially worse in actual reference quality: T041 baseline `12.0441547 / 0.3494575` versus T042 candidate `10.1791756 / 0.3131188`; paired `-1.8649791 dB / -0.0363387`. Directional validity and absolute state quality are therefore not the same thing.

**T044-A — raw low-only legacy excursion is not an unsafe-tail risk signal.** On the accepted T036 cohort, larger step10-to-selected EV+gamma displacement gives ROC-AUC `0.2491501` for the fixed 29/71 regression split and Spearman with paired T036-minus-T026 `ΔPSNR` of `+0.4838044`, both opposite the predeclared risk direction. Do not invert this score or fit thresholds on the same reference-used data.

## Strongest current mechanism interpretation

1. The source-trained Sobolev field is useful in controlled/fresh settings and is often directionally correct early on real trajectories.
2. The original EV+gamma action family is too restrictive; common post-gamma intensity is a real missing degree of freedom.
3. T036 proves common gain can improve fresh target-free aggregate quality without retraining the field, but per-image safety is unresolved.
4. T041/T042 show that late real legacy/feature-state extrapolation is a major causal contributor to **local field-reliability failure**.
5. T043 shows that local field validity and absolute restoration quality decouple: useful net progress can occur before the late local direction becomes unreliable.
6. T044 rejects simple distance-to-step10 as a target-free safety proxy.
7. T045 quantifies a large absolute development gap to training-exposed target-free-at-inference anchors: Retinexformer `21.4788 / 0.7901` and SNR-Aware `23.3963 / 0.8238`, versus T026-A `11.1209 / 0.3738`.
8. T046 rejects material fixed-budget underconvergence as the dominant explanation for the common-gain oracle.
9. T047 shows a post-gamma offset direction is consistently useful but too small when gain is frozen.
10. T048 proves **affine coupling is real and important**, adding `+1.009 dB` mean over T047.
11. T049 proves a compact nonlinear monotonic tone coordinate adds broad, nontrivial capacity and substantial SSIM.
12. T050 rejects material pure-tone underconvergence under a fixed +1000-update extension: the extra gain is only `+0.0020 dB`. The next SOTA-capacity question is no longer optimizer budget or another curve tweak; it is whether a compact smooth **intra-region spatial illumination field** can supply a multi-dB jump.

## Benchmark readiness and sprint target

- T024-A froze the target-free comparison protocol. Reference-based brightness matching/selection is inadmissible.
- T027-A Retinexformer exporter is source/checkpoint bound and target-disabled.
- T027-B SNR-Aware exporter is source/checkpoint bound with the accepted native-pad16 protocol adaptation.
- T033-A Retinexformer development anchor is complete: `21.4787864 / 0.7900612`, with the training-exposure limitation above.
- T045-A SNR-Aware development anchor is complete: `23.3963299 / 0.8237644`, under the exact same frozen split/metric convention and the same training-exposure caveat.
- The first strong-baseline development table is closed enough to guide method investment; do not add another baseline before resolving Ours capacity/optimization.
- **Sprint objective:** the final Ours should target a clear `+2–3 dB` PSNR advantage over the strongest fair target-free baseline on the same held-out protocol, without sacrificing the no-test-target rule. This is an objective, not a current claim.
- **Official LOL-v2 Real test remains sealed** until Ours and all baseline protocols are frozen.

## Best current methods / ceilings

- **Broad fresh-qualified Ours-Core:** T014 Sobolev Region2 TTT.
- **Best fixed-validation deployable candidate:** T026-A, `11.1208764 / 0.3737918`.
- **Fresh-qualified real-domain action extension:** T036-A common-gain TTT, fresh paired `+0.9392571 dB / +0.0083560 RGB-SSIM` over exact T026-A on its new 100-pair cohort; unsafe tail unresolved.
- **Exact EV+gamma non-deployable finite-budget reachability:** T028-A, `17.4599918 / 0.4317158`.
- **Common-gain non-deployable reference reachability after fixed convergence probe:** T046-A, `19.5732278 / 0.4093671`; not a certified global ceiling.
- **Additive-lift-on-frozen-common-gain non-deployable reachability:** T047-A, `20.0559720 / 0.4142735`.
- **Joint affine non-deployable reference reachability:** T048-A, `21.0649800 / 0.4566638`.
- **Monotonic-tone-on-frozen-affine non-deployable reachability:** T049-A, `21.8722552 / 0.5047579`.
- **Fixed tone-convergence extension non-deployable reachability:** T050-A, **`21.8742627 / 0.5051607`**; +1000 extra tone-only updates add only `+0.0020 dB`, so pure budget extension is closed, without claiming a global ceiling.
- **Full-WB non-deployable finite-budget reachability:** T034-A, `19.9079495 / 0.4084354`.
- **Training-exposed supervised development anchors:** T033-A Retinexformer `21.4787864 / 0.7900612`; T045-A SNR-Aware `23.3963299 / 0.8237644`.
- **Heterogeneous-only geometry extension:** T019.

## Information-boundary rules

- Test-time adaptation and checkpoint selection must never consume test labels, clean/normal-light targets, reference gradients/Jacobians, oracle values, PSNR/SSIM, degradation masks/gain maps, condition IDs, annotations, or semantic image IDs.
- Validation/test outputs and decisions must be finalized and persisted before references or evaluation metrics are attached, except in explicitly isolated non-deployable reference diagnostics.
- Reference diagnostics may motivate only global research choices; no per-image oracle quantity may enter deployable inference.
- Source-training clean/reference targets may be used only for source-supervised training or isolated source-domain diagnostics; they are never admissible test-time inputs.
- External baselines admitted to the main comparison must be target-free at inference.
- Fresh/final benchmark sets must remain isolated from model/hyperparameter selection. The official LOL-v2 Real test is still untouched through T050-A.
- Fresh/test runs must fail closed on source/checkpoint/cohort/provenance mismatches.

## Milestones

T001–T013: controlled mechanism/diagnostic sequence. T014: broad controlled/fresh Sobolev Ours-Core. T019: heterogeneous adaptive geometry positive. T020: universal geometry route paused. T021: RGB-SSIM transfer positive. T022/T023: real validation/action-range and tiny recalibration sequence. T024: baseline protocol frozen. T025/T028: reachability diagnosed. T026-A: fixed-validation deployable base. T027-A/B: baseline exporters ready. T029: late-field mismatch. T030: self-reversal guard negative. T031: support-distance association positive. T032: support-radius controller negative. T033: Retinexformer development anchor. T034/T035: RGB-WB capacity resolved mainly to common intensity. T036: common gain fresh target-free aggregate positive with unsafe tail. T037: late-selection headroom limited/mixed. T038: real loss-subset gain mismatch. T039: near-range source gain-tangent deficit not supported. T040: high-range gain-specific source deficit not supported. T041: matched-gain all-100 real selected-state deficit strongly supported. T042: fixed step-10 legacy substitution restores directional field validity. T043: the same substitution is substantially worse in absolute quality, separating local field reliability from state quality. T044: fixed low-only legacy excursion magnitude fails as a regression-risk signal and points toward productive movement instead. T045: SNR-Aware closes the missing strong-baseline development row at `23.3963 / 0.8238`. T046: +1000 fixed common-gain oracle updates add only `+0.0202 dB`. T047: additive lift is broad-positive but sub-gate when gain is frozen. T048: joint gain+lift coupling adds `+1.0090 dB` mean. T049: fixed 8-segment monotonic tone adds `+0.8073 dB` mean / `+0.5989 dB` median PSNR and `+0.0481` mean SSIM. **T050: +1000 additional tone-only updates add only `+0.0020 dB` mean / `+0.00065 dB` median, rejecting material pure-tone underconvergence and closing further tone-budget extension.**

## Current open task

**T051-A — smooth spatial exposure-field marginal-capacity oracle** in `coordination/CHATGPT_TO_CODEX.md`.

Start from each exact accepted T050 selected state; keep EV/gamma/common gain/lift/tone bit-exact frozen; add only one zero-initialized RGB-shared `8×8` smooth log-exposure residual field, bounded to `[-2,+2] EV` by `2*tanh(u)`, bilinearly upsampled (`align_corners=False`) and inserted after affine clamp / before the frozen tone LUT inside the existing hard active mask. Optimize only `u` with one fresh Adam at fixed `lr=0.05` for exactly 500 updates. The sole promotion gate is paired T051-minus-T050 PSNR mean `>= +1.50 dB`, median `>= +0.75 dB`, and non-negative mean RGB-SSIM. This remains one isolated `REFERENCE_ORACLE_ONLY` capacity probe; no sweep, old-coordinate update, retraining, fresh cohort, baseline rerun or official test.