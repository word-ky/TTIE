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
- T067-B is the current simplest development-selected candidate: frozen interpolation `lambda=0.875` between first predicted-safe and normalized-progress endpoints.
- Official LOL-v2 Real test and all cross-dataset held-out sets remain sealed.

The accepted T036/T062/T063 action family starts from each raw low image with identity state and uses the fixed 12-D EV/gamma/gain CommonRegion2/CommonBox renderer. T062/T063 keep this action space and Adam `lr=0.03`, using the fixed low-only objective `L_spa + 10 L_exp + 5 L_col`.

## Current scientific state

### Strong trajectory utility, unresolved rare-tail checkpoint safety

T063-C development selected global `rho=0.9857470621423519`. On the later T063-D/T064-A 100-image cohort, the frozen normalized-progress rule gave absolute `15.4718452 dB / 0.3978969`, mean/median PSNR delta vs exact T036 `+3.8619303/+3.8144067 dB`, `12/100` regressions, and mean RGB-SSIM delta `+0.0184218`, but worst paired delta was `-10.3649447 dB`. T064-A re-rendering proved all `100/100` images had safety-reachable prefix checkpoints, so the dominant problem is checkpoint selection/safety rather than reachable-state capacity.

T066-A target-free dynamics obtained development LOIO unsafe-state recall `73/81 = 0.9012346`, but T066-B/C showed the model is useful mainly as a lower safety-entry signal and does not reliably recognize late-stage catastrophic endpoints.

T067-B development-only interpolation between first-safe and normalized progress selected `lambda=0.875` and passed all five development gates:

- mean / median PSNR delta vs T036 `+2.6360346 / +2.3057191 dB`;
- `1/100` regressions vs T026;
- worst paired delta `-2.4273992 dB`;
- mean RGB-SSIM delta `+0.0204407`.

T067-C applied that exact frozen rule once to the exposed transfer cohort. It preserved strong utility and passed four of five gates: absolute `14.4947881 dB / 0.4086309`, mean/median PSNR delta vs T036 `+2.8848732/+2.5733133 dB`, `5/100` regressions, and mean RGB-SSIM delta `+0.0291559`, but worst paired delta remained `-6.9954830 dB`, below the unchanged `-5.614 dB` floor. Only one selected endpoint remained unsafe under the fixed gate: index 86 at `k_FS=9`, `k_lambda=21`.

T067-D showed unsafe behavior inside `[k_FS,k_rho]` is sparse and can recover: only `12` unsafe states across `5/100` images, with three unsafe intervals later returning safe. The two non-recovering diagnosed tails were safe through absolute step 19 and first became unsafe at step 20, but their normalized boundary positions differed, so neither a universal normalized-q cutoff nor an absolute-step rule was justified.

### Closed late-tail diagnostic families

The following exact constructions have been tested under predeclared contracts and are insufficient for qualification or tail detection:

- **T068-A global absolute-step cap:** development `K=0..27` selects `K=27`, the no-cap control.
- **T068-B cumulative objective-motion knee:** changes `98/100` development choices and loses about `2.64 dB` mean PSNR relative to the frozen `lambda=0.875` control.
- **T068-C three-transition tail motion/objective inefficiency:** development `T99=0.6324473435`; the unsafe transfer endpoint scores `0.4650582340`, rank `15/100`, with `2/99` safe false positives.
- **T068-D aggregate objective-component regret:** development `T99_comp=0.1910869733`; the unsafe transfer endpoint scores `0.0419923923`, rank `52/100`, with `4/99` safe false positives.
- **T069-A endpoint CommonBox projection pressure:** development `T99_proj=0.7311988023`; the unsafe endpoint has zero final clipping pressure (`R_proj=0`) and is not detected.
- **T069-BR symmetric endpoint weighted component-gradient cancellation:** after T069-BN established a numerically sound float64 diagnostic path, development `T99_cancel=0.5881467784474902`; the unsafe endpoint scores `R_cancel=0.1564035401013536`, rank `46/100`, so unsafe-above-threshold is `0/1` while safe false positives are `4/99`. All 200 float32 direct gradients exactly reproduce stored traces, all 200 float64 component-sum checks pass the frozen criterion at machine precision, and the independent verifier reproduces the negative classification. This exact gradient-cancellation statistic is closed.

### Scientific decision after T069-BR

The residual exposed tail has now motivated and falsified several mechanistically distinct scalar diagnostics. Continuing to invent statistics against the same single exposed unsafe endpoint would create increasing post-hoc overfitting risk. Therefore the research line now **stops exposed-tail fitting**.

The next step is not another heuristic. Freeze the simplest development-selected T067-B rule (`lambda=0.875`, `rho=0.9857470621423519`, T066-A model/threshold, T062/T063 trajectory/objective/action space) into an immutable degraded-image-only Final-Ours candidate, replay-audit it against existing target-free anchors, and only after a successful freeze consider opening genuinely held-out official/cross-dataset evaluation.

This freeze is a methodological decision, not a claim that the known exposed rare-tail failure disappeared. Final held-out evaluation will determine whether that failure mode materially limits generalization.

## Retained mechanism conclusions

- The 12-D renderer/trajectory has substantial usable capacity; the dominant unresolved issue is target-free checkpoint selection/safety, not reachable-state capacity.
- Large mean improvement of the zero-reference trajectory has replicated across independent cohorts.
- `first_safe` and normalized progress are complementary: one provides a lower target-free entry point, the other preserves utility. Their global interpolation materially reduces tail harm but does not completely remove rare transfer failures.
- Unsafe interval behavior can be sparse, non-monotone, and recovering. Neither a universal normalized-q cutoff nor a universal absolute-step cap is supported.
- Scalar objective progress, rendered-image motion, aggregate component-value regret, endpoint projection pressure, and the exact symmetric endpoint component-gradient cancellation score do not provide a validated rare-tail guard.
- Do not tune variants of these failed mechanisms on the already-exposed transfer cohort.
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

The original 100-image LOL-v2 Real Train-derived cohort is a **development set**. Exposed transfer cohorts may support diagnostics/method development only after explicit target-free freezing; they are not fresh qualification once references have been inspected. None of these cohorts may support final Ours-vs-baseline gap claims.

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

**T070-A — freeze and replay-audit the Final-Ours candidate** in `coordination/CHATGPT_TO_CODEX.md`.

Compose the exact existing T067-B scientific method into one immutable degraded-image-only inference package. Freeze all T062/T066/T067 code/model/resource/config hashes and exact constants, then replay-audit the 100 development and 100 already-exposed transfer images using only their degraded inputs and existing target-free anchors. Acceptance requires exact selected-step/state/output identity, deterministic repeat evidence, a hash-stable Final-Ours manifest, `reference_reads=0`, no model fit/tuning, and independent verification. No new PSNR/SSIM, clean/reference access, fresh cohort, official LOL-v2 Real test, LSRW, UHD-LL, baseline comparison, or new selector/guard is authorized in this cycle.