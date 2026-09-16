# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Enhancement**

## Core scientific question

Can a vision system adapt a compact spatial image-processing state per test image, without test labels or clean/normal-light targets, so that unknown and spatially heterogeneous degradation is corrected by a reference-free test-time optimization field?

## Current deployable state

**T014 Sobolev Region2 TTT remains the broad fresh-qualified Ours-Core.** It uses the frozen nuisance readout + clean-abstention gate, source-supervised Sobolev restoration energy, hard Region2 EV+gamma adaptation, projected label-free updates, and minimum predicted-energy checkpoint selection.

On LOL-v2 Real development data, the official 100-pair test remains sealed. The accepted fixed-validation deployable base is **T026-A: `11.1208764 dB / 0.3737918 RGB-SSIM`**. T026-B showed that extending the same trajectory from 40 to 80 steps adds only `+0.1210491 dB / +0.0042526`, so the 40-step base remains fixed.

**T036-A common gain is the strongest fresh-qualified deployable action expansion so far.** On its deterministic reference-unused 100-pair training-development cohort it improves exact T026-A by `+0.9392571 dB / +0.0083560 RGB-SSIM` mean, but it is not per-image safe: PSNR declines on 29/100 images, SSIM on 40/100, and the worst PSNR loss is `5.61447 dB`. It is a mechanism result, not the final deployable Ours.

## Optimization-field diagnosis

- **T028-A:** exact EV+gamma reference-only reachability `17.4599918 / 0.4317158`, proving large reachable headroom beyond the learned path.
- **T029-A:** the learned field is directionally strong early and collapses late: step-10 median cosine `0.7266` with `97%` positive-dot, versus selected-state median cosine `-0.2785` with `24%` positive-dot.
- **T031/T032:** source-support distance predicts gradient invalidity moderately well, but the obvious global trust-radius controller fails.
- **T041/T042:** late real legacy/feature-state extrapolation is a major local field-reliability failure; substituting fixed step-10 legacy coordinates restores strong directional alignment.
- **T043:** that early-state substitution is worse in reference quality, proving local gradient validity and absolute restoration quality are not the same thing.
- **T044:** raw step10-to-selected excursion is not a useful target-free regression-risk signal.

The deployable bottleneck remains two-sided: the action family must be expressive enough, and the learned reference-free optimization field must remain valid over the states it visits.

## Capacity sequence through T055

All results in this section are **non-deployable `REFERENCE_ORACLE_ONLY` diagnostics** on the frozen development cohort. Clean targets, reference gradients/states, PSNR/SSIM, or per-image oracle quantities from these diagnostics are forbidden from deployable test-time adaptation.

- **T046-A common-gain continuation:** `19.5732278 / 0.4093671`; +1000 extra updates add only `+0.0202 dB`, so common-gain underconvergence is not the main explanation.
- **T047-A additive lift with gain frozen:** `20.0559720 / 0.4142735`; broad-positive but sub-gate.
- **T048-A joint regional affine coupling:** `21.0649800 / 0.4566638`; joint gain+lift adds `+1.0090 dB` mean / `+0.5584 dB` median over T047.
- **T049-A regional monotonic tone LUT:** `21.8722552 / 0.5047579`; adds `+0.8073 dB` mean / `+0.5989 dB` median and `+0.0481` mean SSIM.
- **T050-A fixed tone convergence extension:** `21.8742627 / 0.5051607`; another +1000 tone-only updates add only `+0.0020 dB` mean / `+0.00065 dB` median, closing pure tone LR/step/budget rescue.
- **T051-A smooth RGB-shared 8×8 exposure field:** `22.5332679 / 0.5219925`; relative to T050, `+0.6590052 dB` mean / `+0.4525454 dB` median PSNR and `+0.0168318` mean SSIM. Broad-positive but below the frozen SOTA-scale promotion gate.
- **T052-A joint smooth exposure + additive field:** `23.1071511 / 0.5700402` (median PSNR `23.8982378`). Relative to T050, `+1.2328884 dB` mean / `+1.0713880 dB` median PSNR and `+0.0648795` mean SSIM. The frozen total gate misses only the `+1.50 dB` mean requirement; relative to T051 it adds `+0.5738832 / +0.5183313 dB` mean/median PSNR and `+0.0480477` mean SSIM.
- **T053-A fixed additive-range closure:** `23.1137228 / 0.5693885`. Widening only `b` from `[-0.20,+0.20]` to `[-0.40,+0.40]` adds only `+0.0065717 dB` mean / `+0.0012222 dB` median and reduces mean SSIM by `0.0006517`; additive range/LR/budget rescue is closed.
- **T054-A fixed local-detail field:** `24.3454867 dB / 0.7577572 RGB-SSIM` (median PSNR `24.8107619`). Starting from exact accepted T052 and freezing every old coordinate, one RGB-shared 8×8 coefficient field multiplies the fixed image-derived basis `D=y0-B5(y0)` using the deterministic separable 5×5 binomial blur. Relative to T052, T054 adds **`+1.2383356 dB` mean / `+1.2353360 dB` median PSNR and `+0.1877170` mean RGB-SSIM**, with both metrics improving on 100/100 images. The selected field is strongly negative, establishing spatially varying detail attenuation / denoising as a major missing renderer capability.
- **T055-A fixed one-scale detail convergence extension:** now independently accepted after T055-V verifier adjudication. Continuing only the accepted T054 detail `v` for fixed +1000 fresh-Adam updates at the same `lr=0.05` reaches **`24.3497922 / 0.7591279`**. Relative to T054 this is only **`+0.0043055 dB` mean / `+0.0029189 dB` median PSNR and `+0.0013707` mean RGB-SSIM**, far below the frozen `+0.25/+0.10 dB/+0.010` gate. Material underconvergence of the fixed one-scale 5×5 family is therefore **not supported**; pure step/LR-budget rescue is closed.
- **T055-V verifier adjudication:** accepted as verifier-only. The original `1.0952353e-6` interpolation failure was caused by separate NumPy rounding versus CUDA contracted weighted sums. A global FMA-order emulation, with no scientific/tolerance change, reduces max interpolation error to `1.1920929e-7`. The frozen 299-file scientific manifest is unchanged, optimizer calls are zero, metrics were not regenerated, and replay passes 100 images / 100,100 history states / 200 metrics / 724 scalar checks with max basis/interpolation/renderer/scalar errors `1.7719e-7 / 1.1921e-7 / 0 / 3.55e-15`.

The capacity diagnosis is now sharper. The one-scale local-detail operator was genuinely valuable, but more optimization of that same family is essentially exhausted. T055 also drives many controls toward the negative asymptote without material quality gain. Because `c=-1` already moves the active output from `y0` to the fixed 5×5 low-pass `B5(y0)`, the next isolated capacity question is not a larger step budget but whether a **coarser local-frequency band** provides residual capacity.

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
- Fresh/final benchmark sets must remain isolated from model/hyperparameter selection. **The official LOL-v2 Real test remains untouched through T055-A/T055-V.**
- Fresh/test runs must fail closed on source/checkpoint/cohort/provenance mismatches.

## Best current methods / ceilings

- **Broad fresh-qualified Ours-Core:** T014 Sobolev Region2 TTT.
- **Best fixed-validation deployable candidate:** T026-A, `11.1208764 / 0.3737918`.
- **Fresh-qualified deployable action extension:** T036-A common gain, `+0.9392571 dB` mean on its fresh cohort, unsafe tail unresolved.
- **Best accepted non-deployable reference reachability:** T055-A, **`24.3497922 / 0.7591279`**; its gain over T054 is negligible, so it is a convergence closure rather than a new capacity mechanism.
- **Training-exposed development anchors:** Retinexformer `21.4787864 / 0.7900612`; SNR-Aware `23.3963299 / 0.8237644`.
- **Heterogeneous-only adaptive geometry extension:** T019; universal geometry repair remains paused after T020.

## Milestones

T001–T013: controlled mechanism/diagnostic sequence. T014: broad controlled/fresh Sobolev Ours-Core. T019: heterogeneous adaptive geometry positive. T020: universal geometry route paused. T021: RGB-SSIM transfer positive. T024: baseline protocol frozen. T026-A: fixed-validation deployable base. T027-A/B: baseline exporters ready. T029: late-field mismatch. T031/T032: support-distance association positive but controller negative. T033: Retinexformer development anchor. T036: common-gain fresh target-free aggregate positive with unsafe tail. T041/T042: late real field-reliability failure and early-state directional rescue. T043: directional validity versus absolute quality separated. T044: simple excursion risk signal rejected. T045: SNR-Aware anchor `23.3963 / 0.8238`. T046: common-gain budget extension negative. T047: lift broad-positive/sub-gate. T048: regional affine coupling positive. T049: nonlinear monotonic tone positive. T050: tone underconvergence rejected. T051: smooth spatial exposure broad-positive but below promotion gate. T052: joint smooth spatial affine family reaches `23.1072 / 0.5700`. T053: additive-range rescue rejected. T054: compact one-scale local-detail control gives a universal large gain to `24.3455 / 0.7578`. **T055/T055-V: another +1000 updates add essentially nothing and independent replay verifies the negative convergence result, closing pure one-scale detail budget rescue.**

PR #75/T050 is merged as `276c0b1fa5c9e6548bef90048ec6fcc43da439c3`. PR #76/T051, #77/T052, #78/T053, #79/T054, #80/T055, and #81/T055-V inherit evidence-history/integration complications; preserve exact accepted scientific/evidence states rather than rewriting history during experiment cycles.

## Current open task

**T056-A — fixed second-scale detail-band marginal-capacity oracle** in `coordination/CHATGPT_TO_CODEX.md`.

Start from exact accepted T055 selected states. Freeze every old coordinate, including T054/T055 `v`. Add only one new RGB-shared 8×8 coefficient field on the fixed mid-frequency basis `D2=B5(y0)-B9(y0)`, where `B5` is the existing 5×5 binomial blur and `B9` is the fixed 9×9 binomial blur. Zero start, fresh Adam `lr=0.05`, exactly 500 updates, single start, same active mask, `REFERENCE_ORACLE_ONLY`. No scale/kernel/range/LR/budget sweep, joint reoptimization, retraining, baseline rerun, fresh cohort, official-test access, or T057 is authorized.
