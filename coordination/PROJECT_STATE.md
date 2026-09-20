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
- T065-A fixed 11-feature ridge trajectory-quality regression, T065-B class-balanced linear logistic safety guard, and T065-C fixed 5-NN safety guard are all closed. The same 11-feature snapshot representation fails under global regression, direct linear classification, and local cross-image nearest-neighbor readout.
- T065-C is especially diagnostic: development leave-one-image-out 5-NN identifies only `1/81` unsafe states and triggers `0/100` development rollbacks; transfer also triggers `0/100` rollbacks, with both catastrophic states surrounded by safe-labeled nearest development states. This materially shifts the bottleneck from classifier form toward **missing transferable safety information in snapshot features**.
- **T066-A is the current bounded experiment:** keep all optimization/inference settings frozen and test one predeclared trajectory-dynamics representation (8 temporal features appended to the same 11 snapshot features) with one fixed development-only class-balanced logistic rollback guard.
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

Development-only calibration selected `tau=16.09636904055864`, but the guard changes `0/100` development and `0/100` transfer choices. Transfer remains `15.4718452 dB / 0.3978969`; the worst paired delta remains `-10.3649447 dB`. The two tail states have sensitivity well below the selected threshold. This statistic is closed.

### T065-A — fixed linear trajectory-quality head: `TRANSFER_NEGATIVE`

T065-A uses exactly 11 target-free snapshot/state/image features and one development-only ridge model (`lambda=1e-3`). Transfer:

- absolute `15.4662827 dB / 0.3977580`;
- mean / median PSNR delta vs exact T036 `+3.8563678 / +3.8144067 dB`;
- `12/100` regressions vs exact T026;
- worst paired delta `-10.3649447 dB` — fail;
- only `4/100` choices change.

The two catastrophic transfer states are severely overestimated by the regressor (`+4.3396/+5.1456 dB` predicted margin versus `-7.1311/-10.3649 dB` true post-freeze margin). The exact ridge head is closed.

### T065-B — class-balanced linear safety guard: `TRANSFER_NEGATIVE`

With the same 11 features and normalization, a development-only class-balanced linear logistic detector identifies `80/81` unsafe states on the training sample but changes `0/100` development and `0/100` transfer choices. Both catastrophic transfer states receive essentially unit safe probability. Transfer remains `15.4718452 dB / 0.3978969` with worst delta `-10.3649447 dB`. This weakens the loss/imbalance explanation and motivates cross-image evaluation of representation quality.

### T065-C — fixed 5-NN safety guard: `TRANSFER_NEGATIVE`

T065-C keeps the exact T065-A 11 features/normalization and development safety labels, replacing the linear boundary with ordinary float64 Euclidean 5-NN (`k=5`, unweighted vote, threshold `0.5`) used only as a rollback guard.

Development leave-one-image-out state confusion:

- safe→safe `2694`, safe→unsafe `25`;
- unsafe→safe `80`, unsafe→unsafe `1`;
- unsafe recall `1/81`;
- `0/100` development choices change.

On the already reference-exposed transfer cohort:

- `0/100` choices change;
- absolute `15.4718452 dB / 0.3978969`;
- mean / median PSNR delta vs exact T036 `+3.8619303 / +3.8144067 dB`;
- `12/100` regressions vs exact T026;
- worst paired delta `-10.3649447 dB` — fail;
- mean RGB-SSIM delta `+0.0184218`.

The two catastrophic states (indices 16/86, steps 21/25) each have five safe-labeled nearest development neighbors and therefore `p_safe=1.0`. This is strong evidence that the exact 11 snapshot features do not carry a simple cross-image transferable safety neighborhood. It does **not** prove that every richer target-free representation must fail.

The information boundary is valid: development labels are allowed only as frozen global training/bank information; transfer inference reads only degraded/current states plus the frozen development resource; all transfer choices/outputs freeze before reference-quality reads. Official test and cross-dataset sets remain untouched.

## Retained mechanism conclusions

- The 12-D renderer/trajectory has substantial usable capacity; the dominant unresolved issue is **target-free checkpoint selection/safety**, not absence of reachable good states.
- Large mean improvement of the T062/T063 zero-reference trajectory has replicated on multiple independent 100-image cohorts.
- Global fixed stopping, cumulative loss-balance, frozen normalized progress alone, antithetic photometric sensitivity, and three readouts of the same 11 snapshot features (ridge, class-balanced logistic, 5-NN) are insufficient to guarantee the rare worst tail.
- T065-C materially strengthens the representation diagnosis because its cross-image leave-one-out unsafe recall collapses even on development. The next controlled question is whether explicit **trajectory dynamics / overshoot features** recover transferable warning information without changing the optimizer or using test references.
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
- Quantities computed entirely from the current degraded image/current target-free intermediate image, frozen state, and frozen globally trained model or development-only reference resource are permissible.
- Development clean/reference targets may be used only offline for global method development/training/hyperparameter selection; they may never become per-image inference inputs.
- Fresh/final sets must remain isolated from method/hyperparameter selection until the corresponding rule is frozen.
- Official LOL-v2 Real test and cross-dataset held-out sets remain sealed until Final Ours is frozen.
- Fresh/test runs must fail closed on source/checkpoint/cohort/provenance mismatches.

## Current open task

**T066-A — fixed trajectory-dynamics safety guard transfer audit** in `coordination/CHATGPT_TO_CODEX.md`.

Keep the optimization trajectory frozen. Append exactly eight predeclared target-free temporal-dynamics features to the same 11 T065-A snapshot features, use development-only normalization for the new features, and fit one fixed class-balanced linear logistic rollback guard. Development evaluation must be leave-one-image-out and must reach unsafe recall `>=0.50` plus all five development gates before transfer quality evaluation is allowed. No feature/model/hyperparameter sweep, optimizer rerun, new cohort, official test, or cross-dataset access is authorized.