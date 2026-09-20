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
- T062/T063 establish a much stronger zero-reference trajectory: roughly `+3.5` to `+3.9 dB` mean PSNR over T036 replicates across independent 100-image cohorts, but rare worst-tail checkpoint selection remains unresolved.
- T063-A and T064-A show the catastrophic tails are **selection-limited, not trajectory-limited**: safe prefix checkpoints exist for every diagnosed image.
- T063-B cumulative loss-balance, T064-B antithetic sensitivity, T065-A ridge quality regression, T065-B class-balanced snapshot logistic, and T065-C snapshot 5-NN are closed.
- T066-A shows trajectory dynamics contain substantial development safety signal (`73/81` unsafe-state LOIO recall), but its frozen transfer guard misses the selected catastrophic tails.
- T066-B is accepted as **`SELECTED_TAIL_SPECIFIC_FAILURE`**: transfer unsafe recall is substantial overall (`77/98`) and inside the normalized-progress prefix (`77/89`), but `0/2` unsafe selected base checkpoints are detected.
- T066-C is accepted as **`PREFIX_REENTRY_SIGNAL_ABSENT`**: after the first predicted-safe crossing, the two catastrophic images remain predicted safe through the selected base checkpoint; same-sequence hysteresis/re-entry rollback is unsupported.
- T067-A is accepted as **`FIRST_SAFE_TRANSFER_NEGATIVE`**: earliest predicted-safe stopping protects the catastrophic tails but fires much too early globally (`64/100` at step 0 or 1) and fails all five gates. `first_safe` is a lower safety-entry signal, not a quality-optimal stopping event.
- T067-B is accepted as **`INTERIOR_PROGRESS_DEV_CANDIDATE_FROZEN`**: development-only robustness-first interpolation between `first_safe` and `k_rho` selects `lambda=0.875`; it passes all five development gates with mean/median PSNR delta vs T036 `+2.6360/+2.3057 dB`, `1/100` regressions vs T026, worst paired delta `-2.4274 dB`, and mean RGB-SSIM delta `+0.02044`.
- T067-C is accepted as **`INTERIOR_PROGRESS_TRANSFER_NEGATIVE`**: the exact frozen `lambda=0.875` rule preserves strong exposed-transfer utility and passes four of five gates, but worst paired PSNR delta vs T026 is `-6.9954830 dB`, below the unchanged `-5.614 dB` floor. Mean/median PSNR delta vs T036 is `+2.8848732/+2.5733133 dB`, regressions are `5/100`, and mean RGB-SSIM delta is `+0.0291559`. Index 16 is brought inside the safety floor (`-4.8984 dB`), while index 86 remains unsafe (`-6.9955 dB`).
- The current bounded question is no longer whether the fixed interior rule transfers—it does not satisfy the safety criterion. The next question is **where post-first-safe safety is lost inside the frozen interval**, before considering any new selector family.
- Official LOL-v2 Real test and all cross-dataset held-out sets remain sealed.

The accepted T036/T062/T063 action family starts from each raw low image with identity state and uses the fixed 12-D EV/gamma/gain CommonRegion2/CommonBox renderer. T062/T063 keep this action space and Adam `lr=0.03`, using the fixed low-only objective `L_spa + 10 L_exp + 5 L_col`.

## Key accepted evidence

### T063-C — normalized objective progress: exposed transfer PASS

Development selects global `rho=0.9857470621423519`. On its already reference-exposed transfer cohort:

- absolute `15.8475933 dB / 0.4171932`;
- mean / median PSNR delta vs exact T036 `+3.7100596 / +3.2033990 dB`;
- `10/100` regressions vs T026;
- worst paired delta `-4.3235641 dB`;
- mean RGB-SSIM delta `+0.0145986`;
- all five gates pass.

This is exposed-cohort development evidence only, not qualification.

### T063-D / T064-A — fresh normalized-progress failure but oracle headroom

The exact frozen normalized-progress selector on a fresh 100-image cohort gives:

- absolute `15.4718452 dB / 0.3978969`;
- mean / median PSNR delta vs T036 `+3.8619303 / +3.8144067 dB`;
- `12/100` regressions vs T026;
- worst paired delta `-10.3649447 dB` — fail;
- mean RGB-SSIM delta `+0.0184218`.

T064-A re-rendering of the frozen `k=0..27` states proves all `100/100` images have safety-reachable prefix checkpoints, with oracle worst paired delta `+0.4279520 dB`. Oracle information is diagnostic only and forbidden at inference.

### T066-A/B/C — dynamics signal exists, but late selected-tail monitoring fails

- T066-A development LOIO unsafe recall: `73/81 = 0.9012346`.
- T066-B transfer unsafe recall: `77/98` overall, `77/89` inside `k<=k_rho`, but `0/2` on unsafe selected base checkpoints.
- The two catastrophic base states are index/step `16/21` and `86/25`; both receive `p_safe≈1` while their true margins are `-7.1311 dB` and `-10.3649 dB`.
- T066-C shows all `77` correctly detected unsafe prefix states occur before first-safe. After first-safe there are `12` reference-unsafe prefix states with no later warning; neither catastrophic image exhibits safe→unsafe→safe re-entry.

**Scientific implication:** the frozen dynamics classifier is useful for identifying entry into a broadly safer region, but not as a reliable late-stage quality monitor.

### T067-A — first-safe stopping: `FIRST_SAFE_TRANSFER_NEGATIVE`

Exact first-safe stopping on the exposed 100-image cohort:

- absolute `9.2430042 dB / 0.2483034`;
- mean / median PSNR delta vs T036 `-2.3669107 / -2.8148540 dB`;
- `75/100` regressions vs T026;
- worst paired delta `-6.1126334 dB`;
- mean RGB-SSIM delta `-0.1311716`;
- all five gates fail.

Yet it protects the two known catastrophic normalized-progress tails: index 16 at step 12 gives `+3.4529 dB` vs T026; index 86 at step 9 gives `-4.2099 dB`, inside the fixed safety floor. This supports using first-safe as an interval endpoint, not as the final stop.

### T067-B — development-only interval interpolation: `INTERIOR_PROGRESS_DEV_CANDIDATE_FROZEN`

A predeclared nine-value interpolation grid between `first_safe` and `k_rho` is frozen before development-quality reads. Robustness-first ranking chooses `lambda=0.875`.

For `lambda=0.875` on development:

- mean / median PSNR delta vs T036 `+2.6360346 / +2.3057191 dB`;
- `1/100` regressions vs T026;
- worst paired delta `-2.4273992 dB`;
- mean RGB-SSIM delta `+0.0204407`;
- all five gates pass.

The endpoint `lambda=1` also passes but has a weaker worst tail (`-4.0744464 dB`). This is in-sample global calibration only.

### T067-C — frozen interior interpolation transfer audit: `INTERIOR_PROGRESS_TRANSFER_NEGATIVE`

The exact development-frozen `lambda=0.875`, `rho=0.9857470621423519`, T066-A model/features, and `0.5` probability threshold are applied once to the already-exposed T063-D/T064-A cohort. All 100 choices and outputs are frozen before reference-quality reads; independent verification passes with `optimizer_runs=0`, `model_fits=0`.

Result:

- absolute `14.4947881 dB / 0.4086309`;
- mean / median PSNR delta vs T036 `+2.8848732 / +2.5733133 dB` — pass;
- `5/100` regressions vs T026 — pass;
- worst paired delta vs T026 `-6.9954830 dB` — **fail**;
- mean RGB-SSIM delta vs T036 `+0.0291559` — pass.

Relative to T063-D, the global interior rule improves the worst tail by about `3.37 dB` but does not meet the fixed floor. Index 16 moves to step 19 and becomes safe (`-4.8984203 dB` vs T026); index 86 moves to step 21 but remains unsafe (`-6.9954830 dB`). The exact `lambda=0.875` transfer candidate is closed; do not lower lambda or add an exposed-cohort rescue rule post hoc.

## Retained mechanism conclusions

- The 12-D renderer/trajectory has substantial usable capacity; the dominant unresolved issue is **target-free checkpoint selection/safety**, not reachable-state capacity.
- Large mean improvement of the zero-reference trajectory has replicated across multiple independent cohorts.
- Fixed stopping, cumulative loss balance, normalized progress alone, antithetic sensitivity, snapshot-based safety readouts, the current dynamics rollback guard, exact first-safe stopping, and the exact global `lambda=0.875` interval rule are insufficient for qualification.
- `first_safe` and normalized progress are complementary: one gives a lower safety-entry endpoint, the other retains utility. Their global interpolation materially reduces tail harm but still misses a rare transferred tail.
- The next scientifically bounded step is diagnostic localization of post-first-safe unsafe states inside the frozen interval, not exposed-cohort retuning.
- Reference-oracle diagnostics may motivate only global research choices. Per-image oracle values, safe ranges, PSNR/SSIM, baseline outcomes, and clean targets are forbidden from deployable inference.

## Fixed evaluation gates used in the current development line

From the accepted `research_log/T063B/core.py` contract:

- mean PSNR delta vs exact T036 `>= 2 dB`;
- median PSNR delta vs exact T036 `> 0`;
- regressions vs exact T026 `<= 29/100`;
- worst paired PSNR delta vs exact T026 `>= -5.614 dB`;
- mean RGB-SSIM delta vs exact T036 `>= -0.001`.

These thresholds must not be changed in response to exposed-cohort outcomes.

## Development versus final-evaluation protocol

The original 100-image LOL-v2 Real Train-derived cohort is a **development set**. It may be used for method design, global training/hyperparameter selection, ablations, and failure analysis. Exposed transfer cohorts may support diagnostics/method development only after explicit freezing; they are not fresh qualification once references have been inspected. None of these cohorts may support final Ours-vs-baseline gap claims.

Final comparison rules:

- Freeze Final Ours, model assets, action space, optimizer/stopping/selection rule, and all hyperparameters before held-out evaluation.
- Standard in-domain comparison must use the **complete official LOL-v2 Real test split** under the frozen inference protocol.
- A **cross-dataset/domain-shift held-out evaluation is required** for the unknown-degradation motivation, e.g. complete LSRW and UHD-LL test splits or equivalent fixed sets.
- No target-specific retraining/tuning is allowed on held-out sets; only per-image target-free test-time adaptation is permitted.
- Development baseline anchors are diagnostic only and must not be reported as the final Ours-vs-baseline gap.

Current development baseline anchors: Retinexformer T033-A `21.4787864 / 0.7900612`; SNR-Aware T045-A `23.3963299 / 0.8237644`.

## Information-boundary rules

- Test-time adaptation and checkpoint/state selection must never consume test labels, clean/normal-light targets, reference gradients/Jacobians, oracle values, PSNR/SSIM, degradation masks/gain maps, condition IDs, annotations, semantic image IDs, or per-image baseline outcome/harm labels.
- Quantities computed entirely from the current degraded image/current target-free intermediate image, frozen state, and frozen globally trained model or development-only resource are permissible.
- Development clean/reference targets may be used only offline for global method development/training/hyperparameter selection; they may never become per-image inference inputs.
- Exposed transfer references may be used only for explicitly post-freeze diagnosis/evaluation; they must never feed the same task's target-free feature/event construction, fitting, thresholding, or selection.
- Fresh/final sets must remain isolated from method/hyperparameter selection until the corresponding rule is frozen.
- Official LOL-v2 Real test and cross-dataset held-out sets remain sealed until Final Ours is frozen.
- Fresh/test runs must fail closed on source/checkpoint/cohort/provenance mismatches.

## Current open task

**T067-D — frozen interval safety-boundary diagnosis** in `coordination/CHATGPT_TO_CODEX.md`.

Freeze the complete target-free `first_safe→k_rho` interval table before any reference-quality read, then use only the already-exposed transfer references to diagnose where the fixed `-5.614 dB` safety boundary is crossed and whether unsafe states recover later. This cycle is diagnostic only: no alternate lambda, cutoff, rollback selector, new model, fresh cohort, official LOL-v2 Real test, LSRW, or UHD-LL access is authorized.
