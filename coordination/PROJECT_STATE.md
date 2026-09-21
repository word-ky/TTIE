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
- Fixed stopping, cumulative loss balance, antithetic sensitivity, snapshot-based safety readouts, the current dynamics rollback guard, exact first-safe stopping, exact global `lambda=0.875` interpolation, a global absolute-step cap, the exact cumulative objective-motion knee, and the exact three-transition endpoint motion/objective inefficiency statistic have all been tested and are insufficient for qualification.
- Official LOL-v2 Real test and all cross-dataset held-out sets remain sealed.

The accepted T036/T062/T063 action family starts from each raw low image with identity state and uses the fixed 12-D EV/gamma/gain CommonRegion2/CommonBox renderer. T062/T063 keep this action space and Adam `lr=0.03`, using the fixed low-only objective `L_spa + 10 L_exp + 5 L_col`.

## Current scientific state

### Normalized progress is strong on average but has rare catastrophic tails

T063-C development selects global `rho=0.9857470621423519`. On an already reference-exposed transfer cohort it passes all five fixed gates with absolute `15.8475933 dB / 0.4171932`, mean/median PSNR delta vs exact T036 `+3.7100596/+3.2033990 dB`, `10/100` regressions vs T026, worst paired delta `-4.3235641 dB`, and mean RGB-SSIM delta `+0.0145986`. This is exposed-cohort development evidence only, not qualification.

On the later T063-D/T064-A 100-image cohort, the same normalized-progress logic gives absolute `15.4718452 dB / 0.3978969`, mean/median PSNR delta vs exact T036 `+3.8619303/+3.8144067 dB`, `12/100` regressions, and mean RGB-SSIM delta `+0.0184218`, but worst paired delta is `-10.3649447 dB`. T064-A re-rendering proves all `100/100` images have safety-reachable prefix checkpoints; the problem is therefore checkpoint selection/safety, not reachable-state capacity.

### Dynamics identify broad safety entry but do not monitor late failure

T066-A adds trajectory-dynamics features and obtains development LOIO unsafe-state recall `73/81 = 0.9012346`. On transfer, T066-B still achieves `77/98` unsafe recall overall and `77/89` within the normalized-progress prefix, but detects `0/2` unsafe selected base checkpoints. T066-C shows all `77` correctly detected unsafe prefix states occur before first predicted-safe crossing; after that crossing the two catastrophic images remain predicted safe through their selected late checkpoints and do not exhibit safe→unsafe→safe re-entry.

**Implication:** the frozen dynamics classifier is useful as a lower safety-entry signal, not as a reliable late-stage quality monitor.

### First-safe and progress are complementary, but the global interval rule is still insufficient

T067-A exact first-safe stopping protects the catastrophic tails but fires much too early globally: `64/100` images stop at step `0` or `1`, with `9.2430042 dB / 0.2483034`, mean/median PSNR delta vs T036 `-2.3669107/-2.8148540 dB`, `75/100` regressions, worst `-6.1126334 dB`, and SSIM delta `-0.1311716`. First-safe is not a quality-optimal stop.

T067-B development-only interpolation between first-safe and normalized progress selects a robustness-first global `lambda=0.875` and passes all five development gates:

- mean / median PSNR delta vs T036 `+2.6360346 / +2.3057191 dB`;
- `1/100` regressions vs T026;
- worst paired delta `-2.4273992 dB`;
- mean RGB-SSIM delta `+0.0204407`.

T067-C applies that exact frozen rule once to the exposed transfer cohort. It preserves strong utility and passes four of five gates: absolute `14.4947881 dB / 0.4086309`, mean/median PSNR delta vs T036 `+2.8848732/+2.5733133 dB`, `5/100` regressions, mean RGB-SSIM delta `+0.0291559`, but worst paired delta `-6.9954830 dB`, below the unchanged `-5.614 dB` floor. Relative to T063-D, interpolation improves the worst tail by about `3.37 dB` but leaves one severe residual failure.

T067-D diagnoses the frozen interval geometry. Across `2465` inclusive `[k_FS,k_rho]` states there are only `12` reference-unsafe states across `5/100` images; three unsafe intervals later recover. The two non-recovering known tails are safe through absolute step 19 and first become unsafe at step 20, but their normalized boundary locations differ (`q≈0.9779` vs `0.8726`).

### Global absolute-step budget is not portable

T068-A tests the complete development-only cap family `K=0..27` under `k_K=max(k_FS,min(k_lambda,K))`. The predeclared ranking selects `K=27`, the no-cap control; K=25/26/27 are identical. K=20..27 all pass the fixed gates, but no finite cap improves the ranked robustness objective over uncapped `lambda=0.875`.

**Implication:** the exposed step-20 coincidence is diagnostic, not evidence for a portable global optimizer-step budget. The absolute-step-cap family is closed.

### Scalar objective-versus-motion geometry is not a reliable late-tail signal

T068-B tests exactly one parameter-free per-image cumulative knee inside `[k_FS,k_lambda]`: normalized cumulative low-only objective progress `u_k` versus normalized cumulative rendered-image RMS motion `v_k`, selecting `argmax(u_k-v_k)`.

The exact rule changes `98/100` development choices and moves almost all of them earlier. It fails three fixed development gates:

- mean PSNR delta vs T036 `-0.0056982 dB` — fail;
- median PSNR delta vs T036 `-0.1198571 dB` — fail;
- regressions vs T026 `25/100` — pass;
- worst paired delta vs T026 `-2.4273992 dB` — pass;
- mean RGB-SSIM delta vs T036 `-0.0116460` — fail.

Versus the exact frozen `lambda=0.875` control, the knee loses `-2.6417328 dB` mean PSNR, `-2.3047391 dB` median PSNR, and `-0.0320868` mean RGB-SSIM. The verifier rerenders `2,041` interval states, independently recomputes all `2,800` development quality states, and matches the primary curves/metrics to numerical precision; `optimizer_runs=0`, `model_fits=0`.

T068-C then tests one strictly local endpoint statistic with a fixed 3-transition window: trailing rendered-image motion share divided by trailing scalar low-only objective-progress share. The development-only nearest-rank threshold is `T99=0.6324473435`. On the exposed transfer cohort, the sole unsafe T067-C selected endpoint (index 86, margin vs T026 `-6.9954830 dB`) has score `R=0.4650582340`, rank `15/100`, so it is not flagged; `0/1` unsafe endpoints and `2/99` safe endpoints exceed T99. Independent verification reproduces the scores, threshold and diagnosis with `optimizer_runs=0`, `model_fits=0`.

**Scientific implication:** both tested scalar-total-objective/image-motion constructions fail for complementary reasons: the cumulative rule fires too early on normal trajectories, while the fixed tail-local ratio does not make the transferred late failure exceptional. These exact constructions are closed; do not tune them on the exposed cohort. The remaining mechanistically distinct question is whether the scalar objective hides **conflict among its own low-only components** as the trajectory approaches the late endpoint.

## Retained mechanism conclusions

- The 12-D renderer/trajectory has substantial usable capacity; the dominant unresolved issue is **target-free checkpoint selection/safety**, not reachable-state capacity.
- Large mean improvement of the zero-reference trajectory has replicated across independent cohorts.
- `first_safe` and normalized progress are complementary: one gives a lower target-free entry endpoint, the other retains utility. Their global interpolation materially reduces tail harm but still misses a rare transferred tail.
- Unsafe interval behavior is sparse and can be non-monotone/recovering. Neither a universal normalized-q cutoff nor a universal absolute-step cap is supported.
- The exact cumulative objective-motion knee and exact three-transition motion/objective endpoint ratio are both closed. Their failures do not prove that every conceivable motion statistic is useless, but there is no justification for further exposed-cohort tuning of this family without a new mechanism.
- The next justified probe is a **diagnostic of low-only objective-component Pareto regret** using the already-existing target-free `L_spa`, `L_exp` and `L_col` histories. It must remain diagnostic until a frozen signal is demonstrated.
- Reference-oracle diagnostics may motivate only global research choices. Per-image oracle values, safe ranges, PSNR/SSIM, baseline outcomes and clean targets are forbidden from deployable inference.

## Fixed evaluation gates used in the current development line

From the accepted `research_log/T063B/core.py` contract:

- mean PSNR delta vs exact T036 `>= 2 dB`;
- median PSNR delta vs exact T036 `> 0`;
- regressions vs exact T026 `<= 29/100`;
- worst paired PSNR delta vs exact T026 `>= -5.614 dB`;
- mean RGB-SSIM delta vs exact T036 `>= -0.001`.

These thresholds must not be changed in response to exposed-cohort outcomes.

## Development versus final-evaluation protocol

The original 100-image LOL-v2 Real Train-derived cohort is a **development set**. It may be used for method design, global training/hyperparameter selection, ablations and failure analysis. Exposed transfer cohorts may support diagnostics/method development only after explicit freezing; they are not fresh qualification once references have been inspected. None of these cohorts may support final Ours-vs-baseline gap claims.

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

**T068-D — frozen low-only objective-component Pareto-regret diagnosis** in `coordination/CHATGPT_TO_CODEX.md`.

Keep T066-A/T067-B/T067-C machinery and the exact low-only objective weights fixed. Over each frozen `[k_FS,k_lambda]` interval, compute exactly one target-free endpoint component-regret score from weighted `L_spa`, `L_exp`, and `L_col`; derive only a target-free development nearest-rank T99 threshold; freeze the exposed-transfer score table before joining already-exposed endpoint safety labels; and diagnose whether the residual unsafe endpoint is an extreme component-regret outlier with low false-positive count. Do not implement a selector/rollback in this cycle. Fresh cohort, official LOL-v2 Real test, LSRW and UHD-LL remain sealed.
