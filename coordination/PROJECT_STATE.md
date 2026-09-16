# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Enhancement**

## Core scientific question

Can a vision system adapt a compact spatial image-processing state per test image, without test labels or clean/normal-light targets, so that unknown and spatially heterogeneous degradation is corrected by a reference-free test-time optimization field?

## Current deployable state

**T014 Sobolev Region2 TTT remains the broad fresh-qualified Ours-Core.** It uses the frozen nuisance readout + clean-abstention gate, source-supervised Sobolev restoration energy, hard Region2 EV+gamma adaptation, projected label-free updates, and minimum predicted-energy checkpoint selection.

On LOL-v2 Real development data, the official 100-pair test remains sealed. The accepted fixed-validation deployable base is **T026-A: `11.1208764 dB / 0.3737918 RGB-SSIM`**. T026-B showed that extending the same trajectory from 40 to 80 steps adds only `+0.1210491 dB / +0.0042526`, so the 40-step base remains fixed.

**T036-A common gain is the strongest fresh-qualified deployable action expansion so far.** On its deterministic reference-unused 100-pair training-development cohort it improves exact T026-A by `+0.9392571 dB / +0.0083560 RGB-SSIM` mean, but it is not per-image safe: PSNR declines on 29/100 images, SSIM on 40/100, and the worst PSNR loss is `5.61447 dB`. It is therefore a mechanism result, not the final deployable Ours.

## Optimization-field diagnosis

- **T028-A:** the exact EV+gamma family has reference-only reachability `17.4599918 / 0.4317158`, proving large reachable headroom beyond the learned path.
- **T029-A:** the learned field is directionally strong early on real trajectories and collapses late: step-10 median cosine `0.7266` with `97%` positive-dot, versus selected-state median cosine `-0.2785` with `24%` positive-dot.
- **T031-A:** source-support distance is associated with gradient invalidity (AUROC `0.72047`, Spearman `-0.48230`), but T032 shows the obvious global trust-radius controller fails.
- **T041/T042:** late real legacy/feature-state extrapolation is a major local field-reliability failure mechanism; substituting fixed step-10 legacy coordinates restores strong directional alignment.
- **T043:** the same early-state substitution is substantially worse in reference quality, proving local gradient validity and absolute restoration quality are not the same thing.
- **T044:** raw step10-to-selected excursion is not a useful target-free regression-risk signal.

The deployable bottleneck remains two-sided: the action family must be expressive enough, and the learned reference-free optimization field must remain valid over the states it visits.

## Capacity sequence through T053

All results in this section are **non-deployable `REFERENCE_ORACLE_ONLY` diagnostics** on the frozen development cohort. Their clean targets, gradients, oracle states, PSNR/SSIM, or per-image oracle quantities are forbidden from deployable test-time adaptation.

- **T046-A common-gain continuation:** `19.5732278 / 0.4093671`; +1000 extra updates add only `+0.0202 dB`, so common-gain underconvergence is not the main explanation.
- **T047-A additive lift with gain frozen:** `20.0559720 / 0.4142735`; broad-positive but sub-gate.
- **T048-A joint regional affine coupling:** `21.0649800 / 0.4566638`; joint gain+lift adds `+1.0090 dB` mean / `+0.5584 dB` median over T047, establishing that scale/offset coupling matters.
- **T049-A regional monotonic tone LUT:** `21.8722552 / 0.5047579`; adds `+0.8073 dB` mean / `+0.5989 dB` median and `+0.0481` mean SSIM.
- **T050-A fixed tone convergence extension:** `21.8742627 / 0.5051607`; another +1000 tone-only updates add only `+0.0020 dB` mean / `+0.00065 dB` median, closing pure tone LR/step/budget extension without claiming a global optimum.
- **T051-A smooth RGB-shared 8×8 exposure field:** `22.5332679 / 0.5219925`. Relative to T050, paired PSNR is `+0.6590052 dB` mean / `+0.4525454 dB` median and mean RGB-SSIM is `+0.0168318`; PSNR improves on 100/100, SSIM on 96/100. It misses the frozen `+1.50/+0.75 dB` promotion gate, so multiplicative spatial exposure alone is not supported as a SOTA-scale capacity jump.
- **T052-A joint smooth exposure + additive field:** **`23.1071511 / 0.5700402`** (median PSNR `23.8982378`). Relative to T050, paired PSNR is **`+1.2328884 dB` mean / `+1.0713880 dB` median** and mean RGB-SSIM is **`+0.0648795`**. The frozen total gate is formally negative because the mean-PSNR requirement was `+1.50 dB`, while median and SSIM conditions pass. Relative to T051, T052 adds `+0.5738832 / +0.5183313 dB` mean/median PSNR and `+0.0480477` mean SSIM; PSNR improves on 100/100 and SSIM on 99/100. Because T052 continued `u` while adding `b`, this supports the joint spatial-affine family broadly but does not isolate scale/offset interaction from continued exposure optimization.
- **T053-A fixed additive-range closure:** `23.1137228 / 0.5693885`. Widening only `b` from `[-0.20,+0.20]` to `[-0.40,+0.40]` with accepted T052 exposure and every older coordinate frozen adds only **`+0.0065717 dB` mean / `+0.0012222 dB` median PSNR** and changes mean RGB-SSIM by **`-0.0006517`**. The predeclared `+0.50/+0.25/nonnegative-SSIM` gate fails. PSNR is microscopically positive on 100/100 images but SSIM declines on 63/100; no winner is at step 500; selected `b` has no `+0.40` hit and only one `-0.40` hit. Therefore the T052 `+0.20` saturation pattern is **not** a material capacity bottleneck. Additive range/LR/budget rescue is closed.

The T052 verifier-only coordinate-rounding repair remains accepted as an evidence correction: scientific source/trajectory/metrics/tolerances are unchanged. T053 uses the accepted verifier arithmetic directly and independently replays 50,100 states / 200 metrics / 728 scalar checks with max renderer/interpolation errors below `1e-6`.

The best accepted compact reference reachability therefore remains **T052: `23.1072 / 0.5700`**, not T053, because T053 fails its gate and slightly reduces mean SSIM. The capacity diagnosis has shifted again: illumination/tone/low-frequency affine freedom is no longer the most plausible next bottleneck. The current renderer is still fundamentally radiometric and lacks an explicit local-frequency/detail operator, while the gap to SNR-Aware is small in PSNR but very large in RGB-SSIM. The next isolated capacity probe tests one compact local-detail field rather than another illumination reparameterization.

## Strong baseline development anchors

- **Retinexformer T033-A:** `21.4787864 / 0.7900612`.
- **SNR-Aware T045-A:** `23.3963299 / 0.8237644`.

Both are target-free at inference under the frozen comparison protocol, but their released supervised recipes/checkpoints are exposed to LOL-v2 Real training data containing this development split. They are therefore **training-exposed development anchors**, not independent held-out SOTA evidence.

The first strong-baseline development table is sufficient for method investment; do not add another baseline before resolving Ours capacity and optimization. The final sprint objective remains a clear **`+2–3 dB` PSNR advantage over the strongest fair target-free baseline on the same held-out protocol**, without sacrificing the no-test-target rule. This is an objective, not a current claim.

## Information-boundary rules

- Test-time adaptation and checkpoint selection must never consume test labels, clean/normal-light targets, reference gradients/Jacobians, oracle values, PSNR/SSIM, degradation masks/gain maps, condition IDs, annotations, or semantic image IDs.
- Validation/test outputs and decisions must be finalized and persisted before references or evaluation metrics are attached, except in explicitly isolated non-deployable reference diagnostics.
- Reference diagnostics may motivate only global research choices; no per-image oracle quantity may enter deployable inference.
- Source-training clean/reference targets may be used only for source-supervised training or isolated source-domain diagnostics; they are never admissible test-time inputs.
- External baselines admitted to the main comparison must be target-free at inference.
- Fresh/final benchmark sets must remain isolated from model/hyperparameter selection. **The official LOL-v2 Real test remains untouched through T053-A.**
- Fresh/test runs must fail closed on source/checkpoint/cohort/provenance mismatches.

## Best current methods / ceilings

- **Broad fresh-qualified Ours-Core:** T014 Sobolev Region2 TTT.
- **Best fixed-validation deployable candidate:** T026-A, `11.1208764 / 0.3737918`.
- **Fresh-qualified deployable action extension:** T036-A common gain, `+0.9392571 dB` mean on its fresh cohort, unsafe tail unresolved.
- **Best accepted non-deployable reference reachability:** T052-A, **`23.1071511 / 0.5700402`**.
- **Training-exposed development anchors:** Retinexformer `21.4787864 / 0.7900612`; SNR-Aware `23.3963299 / 0.8237644`.
- **Heterogeneous-only adaptive geometry extension:** T019; universal geometry repair remains paused after T020.

## Milestones

T001–T013: controlled mechanism/diagnostic sequence. T014: broad controlled/fresh Sobolev Ours-Core. T019: heterogeneous adaptive geometry positive. T020: universal geometry route paused. T021: RGB-SSIM transfer positive. T024: baseline protocol frozen. T026-A: fixed-validation deployable base. T027-A/B: baseline exporters ready. T029: late-field mismatch. T031/T032: support-distance association positive but controller negative. T033: Retinexformer development anchor. T036: common-gain fresh target-free aggregate positive with unsafe tail. T041/T042: late real field-reliability failure and early-state directional rescue. T043: directional validity versus absolute quality separated. T044: simple excursion risk signal rejected. T045: SNR-Aware anchor `23.3963 / 0.8238`. T046: common-gain budget extension negative. T047: lift broad-positive/sub-gate. T048: regional affine coupling positive. T049: nonlinear monotonic tone positive. T050: tone underconvergence rejected. T051: smooth spatial exposure broad-positive but below promotion gate. T052: joint smooth spatial affine family reaches `23.1072 / 0.5700`. **T053: widening the additive spatial range gives only `+0.0066 dB` and slightly worse mean SSIM, rejecting additive-range saturation as a material bottleneck and closing additive range/budget rescue.**

PR #75/T050 is merged as `276c0b1fa5c9e6548bef90048ec6fcc43da439c3`. PR #76/T051, PR #77/T052, and PR #78/T053 are scientifically adjudicated but inherit non-mergeable evidence history; preserve their exact accepted scientific/evidence states rather than rewriting history during experiment cycles.

## Current open task

**T054-A — fixed local-detail-field marginal-capacity oracle** in `coordination/CHATGPT_TO_CODEX.md`.

This is one isolated `REFERENCE_ORACLE_ONLY` probe starting from exact accepted T052 outputs. All existing T052 coordinates are frozen. The only new trainable state is one RGB-shared 8×8 coefficient field multiplying a fixed image-derived local-detail basis `D = y0 - B(y0)` using a deterministic 5×5 binomial blur. No blur/kernel/range/LR/grid sweep, RGB-specific residual, free pixel field, joint reoptimization, retraining, baseline rerun, fresh cohort, deployable use of references, or official-test access is authorized.
