# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Enhancement**

This file is the live scientific state. Detailed task-by-task history remains in Git history and `research_log/`.

## Core scientific question

Can a vision system adapt a compact spatial image-processing state per test image, without test labels or clean/normal-light targets, so that unknown and spatially heterogeneous degradation is corrected by a reference-free test-time optimization field?

## Current deployable state

**T014 Sobolev Region2 TTT remains the broad fresh-qualified Ours-Core.** It uses the frozen nuisance readout + clean-abstention gate, source-supervised Sobolev restoration energy, hard Region2 EV+gamma adaptation, projected label-free updates, and minimum predicted-energy checkpoint selection.

On LOL-v2 Real development data, the official 100-pair test remains sealed. The accepted fixed-validation deployable base remains **T026-A: `11.1208764 dB / 0.3737918 RGB-SSIM`**. T036-A common gain remains the strongest fresh-qualified deployable action expansion found so far, improving exact T026-A by `+0.9392571 dB / +0.0083560 RGB-SSIM` mean on its deterministic reference-unused development cohort, but with an unresolved unsafe tail.

No T059 matched-detail result is deployable. **No real-domain detail rollout is authorized.**

## Renderer / action-family diagnosis

- **T054-A local-detail field** is the strongest missing renderer capability identified: `24.3454867 / 0.7577572`, adding `+1.2383356 dB` mean / `+1.2353360 dB` median PSNR and `+0.1877170` mean SSIM over T052 on the reference-only mechanism cohort, with both metrics improving on all 100 images.
- **T055-A/T055-V** closes simple one-scale optimization-budget rescue: `24.3497922 / 0.7591279`; another fixed +1000 updates add only `+0.0043055 dB` mean.
- **T056-A** second-scale detail and **T057-A** chroma-detail extensions are scientifically valid but fail their frozen materiality gates; neither supersedes T054/T055 as the main mechanism.

These renderer results are non-deployable `REFERENCE_ORACLE_ONLY` diagnostics. Clean targets, reference gradients/states, PSNR/SSIM, and per-image oracle quantities are forbidden from deployable inference.

## Matched-detail optimization-field diagnosis

The matched-detail program has established a strong asymmetry: **detail-direction information transfers much better than calibrated scalar energy/value**.

- **T058-AF:** frozen T014 energy is not ready for direct T054-detail integration. On all 7,346 canonical source states, local-detail alignment is `68.4567%` positive-dot / `0.230516` median cosine, below `>=75% / >=0.50` readiness gates.
- **T059-BF:** unchanged 28-D T014 `EnergyHead` can fit value/legacy/detail jointly in-sample; this establishes source capacity only.
- **T059-C2/E:** unique-image source holdouts are negative for scalar transfer. T059-E bank-relative training reaches inner-train Huber `0.056903` but inner-held Huber `0.221076`; detail direction remains much healthier (`0.868874` positive-dot / `0.491553` median cosine).
- **T059-F2:** positive bankwise scale correction does not fix value Huber (`0.2147133`), but the original T059-E predictions have **median within-bank Spearman `0.9356522` and median argmin regret `0`** on the 80-bank inner-held source diagnostic. The tail is unsafe: mean regret `0.2402099`, p90 `0.0864220`, maximum `6.3640804`, with several strongly reversed banks. This means scalar calibration failure does not automatically imply useless ordering, but ranking safety is unresolved.
- **T059-G/H/I/J/K:** fixed 28-D cross-image locality separates detail from scalar transfer. Train-LOO scalar can pass while unseen-image scalar fails; marginal nearest-distance shift and five-image smoothing do not explain/rescue it; even a five-donor oracle fails on held rows. A global oracle over the fixed 48-image/4,357-row source pool is nearly perfect (`0.000511841` inner-held), so the scalar **values exist** in the source pool but current local geometry does not locate them.
- **T059-L:** simple bank-relative feature centering `x-x_state0` fails even the source control.
- **T059-M:** scalar-only training nearly fits source train (`0.00525188`) yet still fails inner-held (`0.2274732`), ruling out joint Sobolev-objective interference as the main cause.
- **T059-N:** source-only early stopping does not rescue transfer; selected epoch 13 is `0.0407366` fit / `0.149374` selector.
- **T059-O (corrected pinned metrics):** exact bank-context 56-D `z=concat(u-u0,u0)` passes fit-LOO at **`0.0480293073`** but fails the reused 8-image selector at **`0.1556149274`**. A prior lead review transcribed different numbers; PR #114/evidence are authoritative. The classification is unchanged: explicit state-0 anchor context is insufficient for unseen-source-image scalar localization.
- **T059-P:** full frozen T014 CLIP latent does not rescue locality under the fixed bank-context Euclidean 1-NN test. Using all five ordered 512-D normalized views, `z_clip=concat(e-e0,e0)` gives fit-LOO Huber **`0.1124138981`**, already above the unchanged `0.0765084978` gate; selector is **`0.1962943375`**. Thus prompt-score compression is **not established** as the bottleneck, and naive raw-CLIP metric/PCA/layer sweeps are not justified by this result.
- **T059-Q:** exact T059-E/T059-M argmin consensus is **not** a sufficient target-free safety gate. Non-singleton agreement coverage is `41/63 = 0.6507937`; the consensus/abstention policy lowers mean regret from E's `0.2402099` to `0.2136564` and harmed banks from `3/80` to `1/80`, but p90 regret worsens from `0.0864220` to `0.5538099` and maximum regret remains `6.3640804`. Bank 280 is the decisive shared-bias counterexample: both heads choose state 14 and preserve `+6.3640804` harm. Simple argmin disagreement therefore cannot be promoted to C2 or deployment.

### Current scientific interpretation after T059-Q

The scalar marginal support exists globally (T059-K), and the head has ample in-sample capacity (T059-M), but calibrated unseen-image scalar prediction remains unsupported. Absolute 28-D locality, distance-shift explanations, smoothing, local donor oracle, simple displacement/anchor coordinate fixes, source-only early stopping, and the raw frozen-CLIP Euclidean geometry have all failed as sufficient rescues.

T059-F2 still shows that **within-bank ordering is often much better than scalar calibration**: median Spearman is high and median argmin regret is zero, while a small unsafe tail dominates the risk. T059-Q adds an important negative: two heads trained under different objectives can share the same catastrophic argmin, so simple inter-head argmin disagreement does not reliably expose the tail. This is evidence for a **systematic shared bias**, not just independent head noise.

Before abandoning the multi-head uncertainty idea entirely, the next isolated diagnostic asks whether the shared wrong argmin nevertheless comes with low **full-curve rank concordance** between E and M. This is a source-only mechanism check, not a new gate. If full-curve disagreement also fails to concentrate unsafe banks, the dual-head disagreement line should stop rather than be rescued with post-hoc margins/top-k thresholds on the already-opened cohort.

The matched-detail line therefore remains **non-deployable**. Detail-direction evidence is encouraging, but safe target-free scalar/ranking control for the detail-expanded state space is not established.

## Strong baseline development anchors

- **Retinexformer T033-A:** `21.4787864 / 0.7900612`.
- **SNR-Aware T045-A:** `23.3963299 / 0.8237644`.

Both are target-free at inference under the frozen comparison protocol, but their released supervised checkpoints are exposed to LOL-v2 Real training data containing this development split. They are training-exposed development anchors, not independent held-out SOTA evidence.

The final sprint objective remains a clear **`+2–3 dB` PSNR advantage over the strongest fair target-free baseline on the same held-out protocol**, without violating the no-test-target rule. This is an objective, not a current claim.

## Information-boundary rules

- Test-time adaptation and checkpoint selection must never consume test labels, clean/normal-light targets, reference gradients/Jacobians, oracle values, PSNR/SSIM, degradation masks/gain maps, condition IDs, annotations, or semantic image IDs.
- Validation/test outputs and decisions must be finalized and persisted before references or evaluation metrics are attached, except in explicitly isolated non-deployable source/reference diagnostics.
- Reference diagnostics may motivate only global research choices; no per-image oracle quantity may enter deployable inference.
- Source-training clean/reference targets may be used only for source-supervised training or isolated source-domain diagnostics; they are never admissible test-time inputs.
- External baselines admitted to the main comparison must be target-free at inference.
- Fresh/final benchmark sets must remain isolated from model/hyperparameter selection. **The official LOL-v2 Real test remains untouched through T059-Q.**
- Fresh/test runs must fail closed on source/checkpoint/cohort/provenance mismatches.

## Best current methods / ceilings

- **Broad fresh-qualified Ours-Core:** T014 Sobolev Region2 TTT.
- **Best fixed-validation deployable candidate:** T026-A, `11.1208764 / 0.3737918`.
- **Fresh-qualified deployable action extension:** T036-A common gain, `+0.9392571 dB` mean on its fresh cohort, unsafe tail unresolved.
- **Best promoted non-deployable mechanism state:** T055-A, `24.3497922 / 0.7591279`; T056/T057 are valid but failed materiality gates.
- **Matched-detail candidate:** source detail-direction transfer remains promising, but calibrated scalar transfer repeatedly fails and exact dual-head argmin consensus does not safely remove the tail. Current work performs one full-curve disagreement diagnostic before deciding whether to stop this uncertainty family.
- **Training-exposed development anchors:** Retinexformer `21.4787864 / 0.7900612`; SNR-Aware `23.3963299 / 0.8237644`.

## Integration note

Historical PRs may contain inherited history/integration complications. Preserve accepted scientific/evidence states rather than repairing history inside experiment cycles. T059-P is PR #115 and T059-Q is PR #116; both are scientifically reviewable through their pinned source/evidence even while the PRs remain open.

## Current open task

**T059-R — frozen full-curve cross-head disagreement diagnostic** in `coordination/CHATGPT_TO_CODEX.md`.

Using only the already-persisted T059-E and T059-M predictions on the same already-opened 16-image / 1,529-row / 80-bank inner-held source diagnostic, freeze one target-free uncertainty statistic per non-singleton bank: `u = 1 - Spearman(p_E, p_M)` over the complete within-bank score curves. Only after all descriptors and uncertainty ranks are persisted may source `delta_t` be reopened to test whether unsafe E banks are concentrated at high disagreement. No new gate, training, model/feature forward, C2 outer access, target-domain data, LOL-v2, official-test access, or real-domain rollout is authorized in this cycle.
