# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Enhancement**

This file is the live scientific state. Detailed task history remains in Git history and `research_log/`.

## Core scientific question

Can a vision system adapt a compact spatial image-processing state per test image, without test labels or clean/normal-light targets, so that unknown and spatially heterogeneous degradation is corrected by a reference-free test-time optimization field/objective?

## Current deployable / qualification state

- **Broad fresh-qualified Ours-Core:** T014 Sobolev Region2 TTT.
- **Fixed-validation deployable base:** T026-A, `11.1208764 dB / 0.3737918 RGB-SSIM` on its original validation cohort.
- **Fresh-qualified action extension:** T036-A CommonRegion2/CommonBox 12-D gain path, `+0.9392571 dB` mean over exact T026-A on its fresh cohort, with the fixed unresolved tail gate (`29/100` regressions; worst `-5.614 dB`).
- T062/T063 establish a substantially stronger zero-reference trajectory, roughly `+3.5` to `+3.9 dB` mean PSNR over T036 replicates across independent 100-image cohorts, but rare worst-tail checkpoint selection remains unresolved.
- T063-A/T064-A show the severe failures are **selection-limited, not trajectory-limited**: safe prefix checkpoints exist for every diagnosed image.
- Official LOL-v2 Real test and all cross-dataset held-out sets remain sealed.

The accepted T036/T062/T063 action family starts from each raw low image with identity state and uses the fixed 12-D EV/gamma/gain CommonRegion2/CommonBox renderer. T062/T063 keep this action space and Adam `lr=0.03`, using the fixed low-only objective `L_spa + 10 L_exp + 5 L_col`.

## Current scientific state

### Strong trajectory utility, unresolved rare-tail checkpoint safety

T063-C development selects global `rho=0.9857470621423519`. On the later T063-D/T064-A 100-image cohort, the frozen normalized-progress rule gives absolute `15.4718452 dB / 0.3978969`, mean/median PSNR delta vs exact T036 `+3.8619303/+3.8144067 dB`, `12/100` regressions, and mean RGB-SSIM delta `+0.0184218`, but worst paired delta is `-10.3649447 dB`. T064-A re-rendering proves all `100/100` images have safety-reachable prefix checkpoints, so the dominant problem is checkpoint selection/safety rather than reachable-state capacity.

### Dynamics identify safety entry but not late-stage failure

T066-A adds target-free trajectory-dynamics features and obtains development LOIO unsafe-state recall `73/81 = 0.9012346`. T066-B transfer still identifies many unsafe states (`77/98` overall; `77/89` inside the normalized-progress prefix) but detects `0/2` unsafe selected base checkpoints. T066-C shows those correctly detected unsafe states occur before the first predicted-safe crossing; after crossing, the two catastrophic images remain predicted safe through their selected late checkpoints.

**Implication:** the frozen dynamics model is useful as a lower safety-entry signal, not a reliable late-stage quality monitor.

### First-safe and normalized progress are complementary, but one global interpolation is not sufficient

T067-A exact first-safe stopping protects the known tails but fires much too early globally: `64/100` images stop at step `0` or `1`; mean/median PSNR delta vs T036 is `-2.3669107/-2.8148540 dB`, with `75/100` regressions.

T067-B development-only interpolation between first-safe and normalized progress selects a robustness-first global `lambda=0.875` and passes all five development gates:

- mean / median PSNR delta vs T036 `+2.6360346 / +2.3057191 dB`;
- `1/100` regressions vs T026;
- worst paired delta `-2.4273992 dB`;
- mean RGB-SSIM delta `+0.0204407`.

T067-C applies that exact frozen rule once to the exposed transfer cohort. It preserves strong utility and passes four of five gates: absolute `14.4947881 dB / 0.4086309`, mean/median PSNR delta vs T036 `+2.8848732/+2.5733133 dB`, `5/100` regressions, mean RGB-SSIM delta `+0.0291559`, but worst paired delta remains `-6.9954830 dB`, below the unchanged `-5.614 dB` floor. Only one selected endpoint remains unsafe under the fixed gate: index 86 at `k_FS=9`, `k_lambda=21`.

T067-D shows unsafe behavior inside `[k_FS,k_rho]` is sparse and can recover: only `12` unsafe states across `5/100` images, with three unsafe intervals later returning safe. The two non-recovering diagnosed tails are safe through absolute step 19 and first become unsafe at step 20, but their normalized boundary positions differ, so neither a universal normalized-q cutoff nor an absolute-step rule is justified from the exposed cohort.

### Closed global/trajectory-statistic families

The following exact constructions have been tested and are insufficient for qualification or tail detection under their predeclared contracts:

- **T068-A global absolute-step cap:** full development family `K=0..27` selects `K=27`, the no-cap control. The exposed step-20 coincidence is not portable evidence for a global budget.
- **T068-B cumulative objective-motion knee:** changes `98/100` development choices and loses about `2.64 dB` mean PSNR relative to the frozen `lambda=0.875` control; it fires too early on ordinary trajectories.
- **T068-C three-transition tail motion/objective inefficiency:** development `T99=0.6324473435`; the sole unsafe transfer endpoint scores `0.4650582340`, rank `15/100`, so unsafe-above-threshold is `0/1` and safe false positives are `2/99`.
- **T068-D aggregate objective-component regret:** development `T99_comp=0.1910869733`; the sole unsafe transfer endpoint scores `R_comp=0.0419923923`, rank `52/100`; unsafe-above-threshold `0/1`, safe false positives `4/99`. At index 86 the weighted exposure term is at its interval minimum while spatial/color regrets are at their interval maxima, but this does not authorize post-hoc reweighting or component dropping.
- **T069-A endpoint CommonBox projection pressure:** development `T99_proj=0.7311988023`; the sole unsafe transfer endpoint has nonzero final proposed displacement `0.0756094836` but `p == s_end` exactly, hence zero clipping and `R_proj=0`. Unsafe-above-threshold is `0/1`; safe false positives `1/99`. Independent verification reconstructs the exact CommonBox transition and confirms the result with `optimizer_runs=0`, `model_fits=0`.

**Scientific implication after T069-A:** the known residual catastrophic endpoint is not explained by final-step action-bound saturation or clipping. Scalar objective progress, rendered-image motion, endpoint component-value regret, and final projection pressure have all failed as exact late-tail diagnostics. The next justified probe is a mechanistically distinct, symmetric test of **weighted component-gradient cancellation** at the frozen endpoint. This must not be turned into component reweighting or an exposed-cohort-tuned selector.

## Retained mechanism conclusions

- The 12-D renderer/trajectory has substantial usable capacity; the dominant unresolved issue is **target-free checkpoint selection/safety**, not reachable-state capacity.
- Large mean improvement of the zero-reference trajectory has replicated across independent cohorts.
- `first_safe` and normalized progress are complementary: one provides a lower target-free entry point, the other preserves utility. Their global interpolation materially reduces tail harm but does not completely remove rare transfer failures.
- Unsafe interval behavior is sparse and can be non-monotone/recovering. Neither a universal normalized-q cutoff nor a universal absolute-step cap is supported.
- The exact cumulative objective-motion knee, tail-local motion/objective ratio, aggregate objective-component regret, and final-transition projection-pressure statistic are closed. Do not tune variants of these exact failed mechanisms on the exposed transfer cohort.
- The next diagnostic tests whether the three fixed weighted low-only objective gradients substantially cancel in raw 12-D ISP parameter space at the frozen endpoint. It remains diagnosis-only until a target-free frozen signal is demonstrated.
- Reference-oracle diagnostics may motivate only global research choices. Per-image oracle values, safe ranges, PSNR/SSIM, baseline outcomes and clean targets are forbidden from deployable inference.

## Fixed evaluation gates used in the current development line

From the accepted T063B contract:

- mean PSNR delta vs exact T036 `>= 2 dB`;
- median PSNR delta vs exact T036 `> 0`;
- regressions vs exact T026 `<= 29/100`;
- worst paired PSNR delta vs exact T026 `>= -5.614 dB`;
- mean RGB-SSIM delta vs exact T036 `>= -0.001`.

These thresholds must not be changed in response to exposed-cohort outcomes.

## Development versus final-evaluation protocol

The original 100-image LOL-v2 Real Train-derived cohort is a **development set**. It may be used for method design, global training/hyperparameter selection, ablations and failure analysis. Exposed transfer cohorts may support diagnostics/method development only after explicit target-free freezing; they are not fresh qualification once references have been inspected. None of these cohorts may support final Ours-vs-baseline gap claims.

Final comparison rules:

- Freeze Final Ours, model assets, action space, optimizer/stopping/selection rule and all hyperparameters before held-out evaluation.
- Standard in-domain comparison must use the **complete official LOL-v2 Real test split** under the frozen inference protocol.
- A **cross-dataset/domain-shift held-out evaluation is required** for the unknown-degradation motivation, e.g. complete LSRW and UHD-LL test splits or equivalent fixed sets.
- No target-specific retraining/tuning is allowed on held-out sets; only per-image target-free test-time adaptation is permitted.
- Development baseline anchors are diagnostic only and must not be reported as the final Ours-vs-baseline gap.

Current development baseline anchors: Retinexformer T033-A `21.4787864 / 0.7900612`; SNR-Aware T045-A `23.3963299 / 0.8237644`.

## Information-boundary rules

- Test-time adaptation and checkpoint/state selection must never consume test labels, clean/normal-light targets, reference gradients/Jacobians, oracle values, PSNR/SSIM, degradation masks/gain maps, condition IDs, annotations, semantic image IDs, or per-image baseline outcome/harm labels.
- Quantities computed entirely from the current degraded image/current target-free intermediate image, frozen state, and frozen globally trained model or development-only resource are permissible.
- Development clean/reference targets may be used only offline for global method development/training/hyperparameter selection; they may never become per-image inference inputs.
- Exposed transfer references may be used only for explicitly post-freeze diagnosis/evaluation; they must never feed the same task's target-free feature/event construction, fitting, thresholding or selection.
- Fresh/final sets must remain isolated from method/hyperparameter selection until the corresponding rule is frozen.
- Official LOL-v2 Real test and cross-dataset held-out sets remain sealed until Final Ours is frozen.
- Fresh/test runs must fail closed on source/checkpoint/cohort/provenance mismatches.

## Current open task

**T069-B — frozen endpoint component-gradient cancellation diagnosis** in `coordination/CHATGPT_TO_CODEX.md`.

Keep the T066-A/T067-B/T067-C machinery, `lambda=0.875`, accepted trajectories, objective weights `[1,10,5]`, renderer, optimizer/action-box settings and all cohorts fixed. At each frozen endpoint, use only the degraded image and frozen raw state to compute the three fixed weighted low-only objective gradients over all 12 raw ISP parameters and the single symmetric cancellation score specified in the task. Derive only a target-free development nearest-rank T99 threshold; freeze the exposed-transfer score table before joining already-exposed endpoint safety labels; and diagnose whether the residual unsafe endpoint is an extreme gradient-cancellation outlier with low false-positive count. Do not implement a selector/rollback or tune component weights/statistics in this cycle. Fresh cohort, official LOL-v2 Real test, LSRW and UHD-LL remain sealed.
