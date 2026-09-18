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

The matched-detail program has established a strong asymmetry: **detail-direction information transfers much better than calibrated scalar energy/value, and a fixed one-step action has aggregate source benefit, but neither a safe finite-step guarantee nor a transferable target-free abstention rule has been established.**

- **T058-AF:** frozen T014 energy is not ready for direct T054-detail integration. On all 7,346 canonical source states, local-detail alignment is `68.4567%` positive-dot / `0.230516` median cosine, below `>=75% / >=0.50` readiness gates.
- **T059-BF:** unchanged 28-D T014 `EnergyHead` can fit value/legacy/detail jointly in-sample; this establishes source capacity only.
- **T059-C2/E:** unique-image source holdouts are negative for scalar transfer. T059-E bank-relative training reaches inner-train Huber `0.056903` but inner-held Huber `0.221076`; detail direction remains much healthier (`0.868874` positive-dot / `0.491553` median cosine).
- **T059-F2:** positive bankwise scale correction does not fix value Huber (`0.2147133`), but the original T059-E predictions have **median within-bank Spearman `0.9356522` and median argmin regret `0`** on the 80-bank inner-held source diagnostic. The tail is unsafe: mean regret `0.2402099`, p90 `0.0864220`, maximum `6.3640804`, with several strongly reversed banks.
- **T059-G/H/I/J/K:** fixed 28-D cross-image locality separates detail from scalar transfer. Train-LOO scalar can pass while unseen-image scalar fails; marginal nearest-distance shift and five-image smoothing do not explain/rescue it; even a five-donor oracle fails on held rows. A global oracle over the fixed 48-image/4,357-row source pool is nearly perfect (`0.000511841` inner-held), so the scalar **values exist** in the source pool but current local geometry does not locate them.
- **T059-L:** simple bank-relative feature centering `x-x_state0` fails even the source control.
- **T059-M:** scalar-only training nearly fits source train (`0.00525188`) yet still fails inner-held (`0.2274732`), ruling out joint Sobolev-objective interference as the main cause.
- **T059-N:** source-only early stopping does not rescue transfer; selected epoch 13 is `0.0407366` fit / `0.149374` selector.
- **T059-O (corrected pinned metrics):** exact bank-context 56-D `z=concat(u-u0,u0)` passes fit-LOO at **`0.0480293073`** but fails the reused 8-image selector at **`0.1556149274`**. A prior lead review transcribed different numbers; PR #114/evidence are authoritative. The classification is unchanged: explicit state-0 anchor context is insufficient for unseen-source-image scalar localization.
- **T059-P:** full frozen T014 CLIP latent does not rescue locality under the fixed bank-context Euclidean 1-NN test. Using all five ordered 512-D normalized views, `z_clip=concat(e-e0,e0)` gives fit-LOO Huber **`0.1124138981`**, already above the unchanged `0.0765084978` gate; selector is **`0.1962943375`**. Thus prompt-score compression is **not established** as the bottleneck, and naive raw-CLIP metric/PCA/layer sweeps are not justified by this result.
- **T059-Q:** exact T059-E/T059-M argmin consensus is **not** a sufficient target-free safety gate. Non-singleton agreement coverage is `41/63 = 0.6507937`; the consensus/abstention policy lowers mean regret from E's `0.2402099` to `0.2136564` and harmed banks from `3/80` to `1/80`, but p90 regret worsens from `0.0864220` to `0.5538099` and maximum regret remains `6.3640804`. Bank 280 is the decisive shared-bias counterexample: both heads choose state 14 and preserve `+6.3640804` harm.
- **T059-R:** full-curve E/M disagreement also fails as a tail-uncertainty mechanism. On the 63 non-singleton inner-held banks there are 3 unsafe E-alone banks and `u=1-Spearman(p_E,p_M)` achieves only **AUROC `0.8222222`**. Bank 280 / image 45229, the maximum-harm case (`+6.3640804`), has very high E/M curve concordance (`rho=0.9294118`) and only uncertainty rank `31/63`, outside both the preregistered top quartile and top decile. Therefore the dual-head disagreement line is closed.
- **T059-S:** the first finite-step bridge is **negative under its preregistered gate**. Exactly one inherited T054-A Adam step (`lr=0.05`) is applied from each of the 80 source inner-held state-0 anchors after all target-free actions are frozen. Among 61 inherited eligible anchors, **57/61 improve**, median relative MSE change is `-0.0007094`, and anchor-only direction alignment is strong (`0.9180328` positive-dot / `0.6585569` median cosine), but the mean relative-MSE gate fails because a near-zero-MSE anchor (bank 305: `1.27e-16`) is moved to `2.60e-6`. The accepted classification remains `one-step detail-direction transfer is not supported; do not extend to multi-step detail TTT`. This does not authorize retroactive denominator changes or a multi-step rescue.
- **T059-T:** the frozen first-order/finite-step decomposition rejects the proposed Adam-scale/curvature explanation. `57/61 = 0.9344262` eligible anchors have `L=<g_R,Δv><0`, but **0/4** actual harmful anchors are `L<0,A>0` overshoot flips; all four harmful steps already have the wrong first-order sign along the persisted action. Median Adam saturation is `1.0`, so coordinatewise magnitude-normalization is present but does not explain these harms. The direct optimizer-scale rescue is therefore closed.
- **T059-U:** the fixed low-gradient-norm abstention hypothesis fails on the separate corrected T059-C2 outer source cohort. With immutable `tau=0.031453661388567547`, gated coverage is only **`55/80 = 0.6875`**, below the preregistered `0.75` gate. More decisively, only **1/5** ungated harmful anchors is abstained (`0.20` harmful recall), while four harmful actions survive at substantially different norms. The low-norm safety route is therefore closed; no threshold/percentile/norm retuning on this opened cohort is justified. The ungated fixed one-step action itself is aggregate-favorable on the same independent source cohort: **55 improve / 5 harm / 20 tie**, mean absolute MSE change `-3.7988555e-5`, median `-2.2112635e-5`, maximum harm `7.3056247e-6`. This is source evidence for a useful direction field, not a deployable safety claim.
- **T059-V:** the image-only target-free online bridge reproduces the prescribed source replay numerically. All 16 fixed anchors satisfy the preregistered feature/Jacobian/gradient/render bounds, and cached degraded-image tensors are opened only after online tensors freeze. However, **15/16 anchors have a closed gate with exactly zero `J/g` and identity action; only bank 300 / image 45728 exercises a nonzero active path.** Thus the cached-J dependency is removed in principle, but broad active-gradient reproducibility is not yet established and no target-domain rollout is authorized.

### Current scientific interpretation after T059-V

The scalar marginal support exists globally (T059-K), and the head has ample in-sample capacity (T059-M), but calibrated unseen-image scalar prediction remains unsupported. Absolute 28-D locality, distance-shift explanations, smoothing, local donor oracle, simple displacement/anchor coordinate fixes, source-only early stopping, and the raw frozen-CLIP Euclidean geometry have all failed as sufficient scalar rescues. T059-Q/R additionally close the simple multi-head disagreement safety route because the worst scalar-ranking error is a shared systematic bias.

The detail branch now has a more precise status. T059-S/T show that the transferred field is directionally meaningful on many source anchors, but its harmful cases already have the wrong first-order sign; optimizer-scale or curvature rescue is not the explanation. T059-U rejects the predeclared low-gradient-norm abstention clue on an independent source outer cohort. At the same time, the completely ungated fixed one-step action remains aggregate-favorable there, with only 5/80 harmful anchors and negative mean/median absolute MSE change. Therefore the field is **potentially useful but not safety-calibrated**.

T059-V resolves one engineering prerequisite only narrowly: the degraded-image feature/action Jacobian and one-step action can be reconstructed online from the current image and frozen model without reference information, and the sole nonzero prescribed case matches the accepted cached path. Because the fixed lowest-bank selector produced 15 identity cases, active-path coverage is inadequate for a target-domain rollout. T059-W therefore stress-tests exactly one target-free nonzero-gradient anchor per fixed outer source image before any target-domain evidence is spent.

The matched-detail line remains **non-deployable**. No multi-step detail TTT, target-domain detail action, LOL-v2 detail action, official-test detail action, or real-domain detail rollout is currently authorized.

## Strong baseline development anchors

- **Retinexformer T033-A:** `21.4787864 / 0.7900612`.
- **SNR-Aware T045-A:** `23.3963299 / 0.8237644`.

Both are target-free at inference under the frozen comparison protocol, but their released supervised checkpoints are exposed to LOL-v2 Real training data containing this development split. They are training-exposed development anchors, not independent held-out SOTA evidence.

The final sprint objective remains a clear **`+2–3 dB` PSNR advantage over the strongest fair target-free baseline on the same held-out protocol**, without violating the no-test-target rule. This is an objective, not a current claim.

## Information-boundary rules

- Test-time adaptation and checkpoint selection must never consume test labels, clean/normal-light targets, reference gradients/Jacobians, oracle values, PSNR/SSIM, degradation masks/gain maps, condition IDs, annotations, or semantic image IDs.
- A degraded-image feature/action Jacobian such as `dx/dv`, computed entirely from the current degraded image and frozen model, is permissible; a Jacobian/gradient that depends on a clean/reference target is not.
- Validation/test outputs and decisions must be finalized and persisted before references or evaluation metrics are attached, except in explicitly isolated non-deployable source/reference diagnostics.
- Reference diagnostics may motivate only global research choices; no per-image oracle quantity may enter deployable inference.
- Source-training clean/reference targets may be used only for source-supervised training or isolated source-domain diagnostics; they are never admissible test-time inputs.
- External baselines admitted to the main comparison must be target-free at inference.
- Fresh/final benchmark sets must remain isolated from model/hyperparameter selection. **The official LOL-v2 Real test remains untouched through T059-V.**
- Fresh/test runs must fail closed on source/checkpoint/cohort/provenance mismatches.

## Best current methods / ceilings

- **Broad fresh-qualified Ours-Core:** T014 Sobolev Region2 TTT.
- **Best fixed-validation deployable candidate:** T026-A, `11.1208764 / 0.3737918`.
- **Fresh-qualified deployable action extension:** T036-A common gain, `+0.9392571 dB` mean on its fresh cohort, unsafe tail unresolved.
- **Best promoted non-deployable mechanism state:** T055-A, `24.3497922 / 0.7591279`; T056/T057 are valid but failed materiality gates.
- **Matched-detail candidate:** scalar calibration/localization, simple dual-head uncertainty, optimizer-scale rescue, and low-gradient-norm abstention are all unsupported as sufficient solutions. The ungated fixed one-step detail field is aggregate-favorable on the independent source outer cohort but lacks a transferable safety gate. T059-V establishes only a narrow source-side online-Jacobian/action replay because 15/16 prescribed anchors are identity cases. The only active task is T059-W active-path source replay; no target-domain detail rollout is authorized yet.
- **Training-exposed development anchors:** Retinexformer `21.4787864 / 0.7900612`; SNR-Aware `23.3963299 / 0.8237644`.

## Integration note

Historical PRs may contain inherited history/integration complications. Preserve accepted scientific/evidence states rather than repairing history inside experiment cycles. T059-P is PR #115, T059-Q is PR #116, T059-R is PR #117, T059-S is PR #118, T059-T is PR #119, T059-U is PR #120, and T059-V is PR #121; they are scientifically reviewable through their pinned source/evidence even while PRs remain open.

## Current open task

**T059-W — active-path online Jacobian/action replay stress test** in `coordination/CHATGPT_TO_CODEX.md`.

Using only pre-reference T059-U target-free action metadata, select exactly one state-0 anchor per fixed outer source image by maximum persisted predicted-gradient norm (smallest-bank tie break), require every chosen anchor to have positive norm and active regions, bind the 16 IDs before execution, then rerun the T059-V image-only online path. Cached degraded-image tensors may be opened only after all online tensors freeze. No clean/reference image or metric, target-domain data, LOL-v2, or official test may be accessed. This task tests active numerical reproducibility only; even a pass does not authorize a target-domain rollout until the next research-lead review.
