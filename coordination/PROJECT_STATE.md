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
- T062/T063 establish a substantially stronger zero-reference trajectory, roughly `+3.5` to `+3.9 dB` mean PSNR over T036 replicates across independent 100-image cohorts.
- T063-A/T064-A show severe failures are **selection-limited, not trajectory-limited**: safe prefix checkpoints exist for every diagnosed image.
- **Final-Ours candidate is now frozen by T070-A** as the unchanged T067-B rule: CommonRegion2/CommonBox 12-D trajectory, T066-A first-safe signal, `rho=0.9857470621423519`, and `lambda=0.875` safety–utility interpolation.
- T070-A immutable manifest SHA256: `e7f129d931d27531e5f6a14cd72e8c94734c474c40c3c403959cc8f5764e5ab9`; scientific source: `aa4d920dff4b5b76751c24266e95ac9696d55d90`.
- The complete official LOL-v2 Real test is now authorized only under T071-A's frozen held-out protocol. LSRW, UHD-LL, and other cross-dataset held-out sets remain sealed.

The frozen action family starts from each raw low image with identity state and uses the fixed 12-D EV/gamma/gain CommonRegion2/CommonBox renderer. Adaptation uses Adam `lr=0.03`, 27 updates, and the fixed low-only objective `L_spa + 10 L_exp + 5 L_col`.

## Current scientific state

### Strong trajectory utility; rare-tail behavior remains known but no longer drives fitting

T063-C development selected global `rho=0.9857470621423519`. On the later T063-D/T064-A 100-image cohort, normalized progress gave `15.4718452 dB / 0.3978969`, mean/median PSNR delta vs exact T036 `+3.8619303/+3.8144067 dB`, but worst paired delta `-10.3649447 dB`. T064-A proved all `100/100` images had safety-reachable prefix checkpoints.

T066-A target-free dynamics achieved development LOIO unsafe-state recall `73/81 = 0.9012346`, but T066-B/C established that it is mainly a lower safety-entry signal and not a reliable late-stage catastrophic-quality monitor.

T067-B development-only interpolation between first-safe and normalized progress selected `lambda=0.875` and passed all five development gates:

- mean / median PSNR delta vs T036 `+2.6360346 / +2.3057191 dB`;
- `1/100` regressions vs T026;
- worst paired delta `-2.4273992 dB`;
- mean RGB-SSIM delta vs T036 `+0.0204407`.

T067-C applied that exact frozen rule once to the exposed transfer cohort and preserved strong average utility: absolute `14.4947881 dB / 0.4086309`, mean/median PSNR delta vs T036 `+2.8848732/+2.5733133 dB`, `5/100` regressions, mean RGB-SSIM delta `+0.0291559`. One selected endpoint remained below the unchanged tail floor, with worst paired delta `-6.9954830 dB`.

That exposed tail motivated several predeclared diagnostics, all now closed as insufficient: global absolute-step cap, cumulative objective-motion knee, tail-local motion/objective inefficiency, aggregate objective-component regret, final CommonBox projection pressure, and symmetric endpoint weighted component-gradient cancellation. Continuing to invent statistics against the same exposed failure would be post-hoc overfitting.

### T070-A freezes Final Ours and ends the method-design phase

T070-A packages the exact T067-B candidate as a degraded-image-only inference API. The per-image scientific input is only the low RGB image; the manifest binds frozen global code/assets/configuration. No clean/reference image, label, PSNR/SSIM, baseline outcome, oracle range, condition ID, or per-image safety annotation enters inference.

The freeze/replay audit is exact:

- 100 development + 100 already-exposed transfer full inference replays;
- zero selected-step mismatches;
- zero selected-state mismatches;
- zero output mismatches;
- zero complete 27-update trajectory-prefix mismatches;
- 10/10 predeclared deterministic repeats exact;
- independent verifier PASS after 5,600 GPU renders;
- `reference_reads=0`, `model_fits=0`;
- primary `optimizer_runs=210`, `optimizer_updates=5670`.

This changes the scientific phase: **Final Ours is frozen for held-out evaluation.** The known exposed rare-tail issue is not claimed solved, but it is no longer permissible to tune the method against that exposed cohort or against any upcoming official/cross-dataset test outcome.

## Retained mechanism conclusions

- The 12-D renderer/trajectory has substantial usable capacity; the main historical bottleneck was target-free checkpoint selection/safety, not reachable-state capacity.
- Large mean improvement of the zero-reference trajectory replicated across independent cohorts.
- `first_safe` and normalized progress are complementary: one gives a lower safety-entry point; the other preserves utility. Their frozen global interpolation materially reduces tail harm without eliminating every exposed rare failure.
- Unsafe behavior can be sparse, non-monotone, and recovering. Neither a universal normalized-q cutoff nor a universal absolute-step cap is supported.
- Scalar objective progress, rendered-image motion, aggregate component-value regret, endpoint projection pressure, and the exact symmetric component-gradient cancellation score are not validated late-tail guards.
- Do not tune variants of these failed mechanisms on exposed or held-out evaluation outcomes.
- Reference-oracle diagnostics may motivate only global development decisions before freezing. Per-image oracle values, safe ranges, PSNR/SSIM, baseline outcomes and clean targets are forbidden from deployable inference.

## Fixed development gates retained for historical interpretation

From the accepted T063B contract:

- mean PSNR delta vs exact T036 `>= 2 dB`;
- median PSNR delta vs exact T036 `> 0`;
- regressions vs exact T026 `<= 29/100`;
- worst paired PSNR delta vs exact T026 `>= -5.614 dB`;
- mean RGB-SSIM delta vs exact T036 `>= -0.001`.

These were development gates only. They must not be retrofitted into held-out-test tuning criteria.

## Development versus final-evaluation protocol

The original 100-image LOL-v2 Real Train-derived cohort is a **development set**. Exposed transfer cohorts are not fresh qualification once references have been inspected. None of these cohorts may support final Ours-vs-baseline gap claims.

Final comparison rules:

- Final Ours, model assets, action space, optimizer/stopping/selection rule and all hyperparameters are now frozen by T070-A.
- Standard in-domain comparison must use the **complete official LOL-v2 Real test split** under the frozen inference protocol.
- A **cross-dataset/domain-shift held-out evaluation remains required** for the unknown-degradation motivation, e.g. complete LSRW and UHD-LL test splits or equivalent fixed sets.
- No target-specific retraining/tuning is allowed on held-out sets; only per-image target-free test-time adaptation is permitted.
- Development baseline anchors are diagnostic only and must not be reported as the final Ours-vs-baseline gap.
- Held-out outputs/decisions must be frozen before references are read. Reference metrics may be computed only post hoc and may never feed reruns or method changes.

Current development baseline anchors: Retinexformer T033-A `21.4787864 / 0.7900612`; SNR-Aware T045-A `23.3963299 / 0.8237644`.

## Information-boundary rules

- Test-time adaptation and checkpoint/state selection must never consume test labels, clean/normal-light targets, reference gradients/Jacobians, oracle values, PSNR/SSIM, degradation masks/gain maps, condition IDs, annotations, semantic image IDs, or per-image baseline outcome/harm labels.
- Quantities computed entirely from the current degraded image/current target-free intermediate image, frozen state, and frozen globally trained/development-only assets are permissible.
- Held-out clean/reference targets may be read only after the corresponding target-free output/decision table is irrevocably frozen and hashed.
- Held-out results cannot authorize method tuning, threshold changes, sample exclusion, or reruns with altered scientific settings.
- Fresh/test runs must fail closed on source/checkpoint/cohort/provenance mismatches.

## Current open task

**T071-A — complete official LOL-v2 Real test held-out evaluation of frozen Final Ours** in `coordination/CHATGPT_TO_CODEX.md`.

Run the exact T070-A manifest/source once on the complete official LOL-v2 Real test low images with strict inference-stage `reference_reads=0`. Freeze and hash every target-free output/decision before any reference read, then compute post-hoc PSNR/RGB-SSIM and selected-step statistics over the complete official split. No baseline run, cross-dataset run, method change, or outcome-driven rerun is authorized in this cycle. LSRW and UHD-LL remain sealed.