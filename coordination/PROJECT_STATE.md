# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Enhancement**

This file is the live scientific state. Detailed task history remains in Git history and `research_log/`.

## Core scientific question

Can a vision system adapt a compact spatial image-processing state per test image, without test labels or clean/normal-light targets, so that unknown and spatially heterogeneous degradation is corrected by a reference-free test-time optimization field/objective?

## Current deployable / qualification state

- **Broad fresh-qualified Ours-Core:** T014 Sobolev Region2 TTT.
- **Fixed-validation deployable base:** T026-A, `11.1208764 dB / 0.3737918 RGB-SSIM` on its original validation cohort.
- **Fresh-qualified action extension:** T036-A CommonRegion2/CommonBox 12-D gain path, `+0.9392571 dB` mean over exact T026-A on its fresh cohort, with an unresolved tail (`29/100` regressions; worst `-5.614 dB`).
- T062/T063 establish a much stronger zero-reference trajectory: mean gains of about `+3.5` to `+3.9 dB` over T036 replicate across independent 100-image cohorts, but fixed/global or normalized-progress checkpoint rules fail rare worst-tail qualification.
- T063-A and T064-A independently show the relevant catastrophic tails are **selection-limited, not trajectory-limited**: every image in each diagnosed cohort has a safe prefix checkpoint.
- T063-B cumulative loss-balance and T064-B antithetic photometric-sensitivity guards are closed as non-discriminative target-free statistics.
- T065-A fixed 11-feature ridge trajectory-quality regression is also closed: it preserves strong average performance but badly overestimates the rare unsafe transfer states and fails the immutable worst-tail gate.
- T065-B is the current bounded experiment: reuse the exact T065-A features, but directly train one class-balanced linear safety detector on development labels and use it only as a rollback guard.
- Official LOL-v2 Real test and all cross-dataset held-out sets remain sealed.

The accepted T036/T062/T063 action family starts from each raw low image with identity state and uses the fixed 12-D EV/gamma/gain CommonRegion2/CommonBox renderer. T062/T063 keep this action space and Adam `lr=0.03`, using the fixed low-only objective `L_spa + 10 L_exp + 5 L_col`.

## Key accepted evidence

### T062-C-R2 — fixed step 27 fresh qualification: NEGATIVE

On a deterministic previously reference-unused 100-pair LOL-v2 Real Train cohort:

- absolute `15.6879652 dB / 0.4124338`;
- mean / median PSNR delta vs exact T036 `+3.5504315 / +3.3795780 dB`;
- `10/100` regressions vs exact T026;
- worst paired delta vs exact T026 `-7.3341753 dB` — worst-tail gate fail;
- mean RGB-SSIM delta vs T036 `+0.0098393`.

Four of five gates pass. The large mean signal is real; fixed step 27 is not qualified.

### T063-A — `SELECTION_HEADROOM_PRESENT` (`REFERENCE_ORACLE_ONLY`)

On the frozen T062-C-R2 prefix `k=0..27`:

- safety-reachable prefix state: `100/100`;
- mean / median PSNR delta vs exact T036 `+5.2221903 / +4.2352859 dB`;
- regressions vs exact T026 `0/100`;
- worst paired delta `+0.8757352 dB`;
- mean RGB-SSIM delta vs T036 `+0.0295648`;
- oracle absolute `17.3597241 dB / 0.4321593`.

This proves strong checkpoint headroom but is diagnostic only; reference/oracle information is forbidden from deployable inference.

### T063-C — normalized objective progress: exposed transfer PASS

Development selects one global fraction `rho=0.9857470621423519`. For each image, using only the low/current trajectory, define `L_best=min_{0..27} L_k`, `D=L_0-L_best`, then choose the earliest checkpoint satisfying `L_k <= L_0-rho*D`.

On an already reference-exposed 100-image transfer cohort:

- absolute `15.8475933 dB / 0.4171932`;
- mean / median PSNR delta vs exact T036 `+3.7100596 / +3.2033990 dB`;
- `10/100` regressions vs exact T026;
- worst paired delta `-4.3235641 dB`;
- mean RGB-SSIM delta `+0.0145986`;
- all five gates pass;
- `45/100` choose earlier than step 27.

This is exposed-cohort transfer evidence only, not qualification.

### T063-D — frozen normalized-progress fresh qualification: NEGATIVE

The exact frozen T063-C rule was applied without retuning to a second deterministic previously reference-unused 100-pair cohort:

- absolute `15.4718452 dB / 0.3978969`;
- mean / median PSNR delta vs exact T036 `+3.8619303 / +3.8144067 dB`;
- `12/100` regressions vs exact T026;
- worst paired delta `-10.3649447 dB` — worst-tail gate fail;
- mean RGB-SSIM delta `+0.0184218`;
- four of five gates pass.

The exact normalized-progress selector is closed as a qualification candidate.

### T064-A — `FRESH_TAIL_SELECTION_LIMITED` (`REFERENCE_ORACLE_ONLY`)

Re-rendering only the already-frozen T063-D `k=0..27` states shows:

- safety-reachable prefix state: `100/100`;
- mean / median PSNR delta vs exact T036 `+5.4523215 / +4.8916423 dB`;
- regressions vs exact T026 `0/100`;
- worst paired delta `+0.4279520 dB`;
- mean RGB-SSIM delta `+0.0370898`;
- oracle absolute `17.0622363 dB / 0.4165648`.

Both T063-D catastrophic failures contain safe early checkpoints. Their exact oracle steps/ranges are reference-derived diagnostics and must not be hard-coded or used at inference.

### T064-B — antithetic photometric-sensitivity guard: `TRANSFER_NEGATIVE`

A fixed target-free statistic measured local amplification of a deterministic `±1/255` antithetic Rademacher perturbation at frozen states. Development-only calibration selected `tau=16.09636904055864`.

Result:

- guard changes vs frozen T063-C: `0/100` development and `0/100` transfer;
- transfer absolute `15.4718452 dB / 0.3978969`;
- mean / median PSNR delta vs exact T036 `+3.8619303 / +3.8144067 dB`;
- `12/100` regressions vs exact T026;
- worst paired delta `-10.3649447 dB` — fail;
- mean RGB-SSIM delta `+0.0184218`.

The two known tail checkpoints have sensitivity values well below the selected threshold, so this statistic does not discriminate the unsafe late states. The exact antithetic-sensitivity guard is closed. The inference/freeze boundary was valid.

### T065-A — fixed linear trajectory-quality head: `TRANSFER_NEGATIVE`

T065-A reused the frozen T063 trajectory and exactly 11 target-free state/trajectory/image features. On the original development cohort, one float64 ridge regression (`lambda=1e-3`) was trained against development-only paired margins `PSNR(state)-PSNR(T026)`, then frozen before transfer selection.

On the already reference-exposed T063-D/T064-A transfer cohort:

- absolute `15.4662827 dB / 0.3977580`;
- mean / median PSNR delta vs exact T036 `+3.8563678 / +3.8144067 dB`;
- `12/100` regressions vs exact T026;
- worst paired delta `-10.3649447 dB` — fail;
- mean RGB-SSIM delta `+0.0182830`;
- only `4/100` transfer choices change relative to frozen normalized progress.

The development fit RMSE is `2.2575781 dB` over 2,800 state samples, but the two catastrophic transfer states are severely overestimated by the regressor: predicted margins `+4.3396/+5.1456 dB` versus true post-freeze margins `-7.1311/-10.3649 dB`. Thus the exact average-margin ridge head is closed. The result suggests the next controlled question is whether the fixed feature space can support **direct rare-safety classification** when class imbalance/loss mismatch is addressed; it does not justify adding test labels, oracle signals, or transfer supervision.

The T065-A inference/freeze boundary was valid: the model was trained only from development references, all transfer choices/outputs were frozen before reference-quality reads, and no transfer clean target, PSNR/SSIM, oracle step/range, official test, or cross-dataset data entered selection. Independent verification reproduced features, coefficients, choices, hashes, metrics, and classification.

## Retained mechanism conclusions

- The 12-D renderer/trajectory has substantial usable capacity; the dominant unresolved issue is **target-free checkpoint selection/safety**, not absence of reachable good states.
- Large mean improvement of the T062/T063 zero-reference trajectory has replicated on multiple independent 100-image cohorts.
- Global fixed stopping, cumulative loss-balance, frozen normalized progress alone, antithetic photometric sensitivity, and average-margin linear ridge prediction are insufficient to guarantee the rare worst tail.
- The exact T065-A failure is consistent with rare-event loss/class-imbalance mismatch, but this remains a hypothesis until T065-B; do not yet claim the 11-feature space is sufficient or insufficient in general.
- Reference-oracle diagnostics may motivate only global research choices. Per-image oracle values, safe ranges, PSNR/SSIM, baseline outcomes, and clean targets are forbidden from deployable inference.

## Development versus final-evaluation protocol

The original fixed 100-image LOL-v2 Real Train-derived cohort is a **development set**. It may be used for method design, global training/hyperparameter selection, ablations, and failure analysis. It must not support final Ours-vs-baseline gap claims.

Final comparison rules:

- Freeze Final Ours, model assets, action space, optimizer/stopping/selection rule, and all hyperparameters before held-out evaluation.
- Standard in-domain comparison must use the **complete official LOL-v2 Real test split** under the frozen inference protocol.
- A **cross-dataset/domain-shift held-out evaluation is required** for the unknown-degradation motivation, e.g. complete LSRW and UHD-LL test splits or equivalent fixed sets.
- No target-specific retraining/tuning is allowed on held-out sets; only per-image target-free test-time adaptation is permitted.
- Development baseline anchors are diagnostic only and must not be reported as the final Ours-vs-baseline gap.

Current development baseline anchors: Retinexformer T033-A `21.4787864 / 0.7900612`; SNR-Aware T045-A `23.3963299 / 0.8237644`.

## Information-boundary rules

- Test-time adaptation and checkpoint/state selection must never consume test labels, clean/normal-light targets, reference gradients/Jacobians, oracle values, PSNR/SSIM, degradation masks/gain maps, condition IDs, annotations, semantic image IDs, or per-image baseline outcome/harm labels.
- Quantities computed entirely from the current degraded image/current target-free intermediate image, frozen state, and frozen globally trained model are permissible.
- Development clean/reference targets may be used only offline for global method development/training/hyperparameter selection; they may never become per-image inference inputs.
- Fresh/final sets must remain isolated from method/hyperparameter selection until the corresponding rule is frozen.
- Official LOL-v2 Real test and cross-dataset held-out sets remain sealed until Final Ours is frozen.
- Fresh/test runs must fail closed on source/checkpoint/cohort/provenance mismatches.

## Current open task

**T065-B — class-balanced linear safety rollback transfer audit** in `coordination/CHATGPT_TO_CODEX.md`.

Reuse the exact T065-A 11 target-free features and frozen T063-C normalized-progress base checkpoint. Train exactly one development-only class-balanced linear logistic detector for whether `PSNR(state)-PSNR(T026) >= -5.614 dB`, freeze it, and use it only to roll back a base checkpoint that is predicted unsafe. No new features/model family, threshold/hyperparameter sweep, optimizer rerun, new cohort, official test, or cross-dataset access is authorized.