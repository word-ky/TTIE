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
- T066-C is accepted as **`PREFIX_REENTRY_SIGNAL_ABSENT`**: after first predicted-safe crossing, the two catastrophic images remain predicted safe through the selected base checkpoint; same-sequence hysteresis/re-entry rollback is unsupported.
- T067-A is accepted as **`FIRST_SAFE_TRANSFER_NEGATIVE`**: earliest predicted-safe stopping protects the catastrophic tails but fires much too early globally (`64/100` at step 0 or 1) and fails all five gates. `first_safe` is a lower safety-entry signal, not a quality-optimal stopping event.
- T067-B is accepted as **`INTERIOR_PROGRESS_DEV_CANDIDATE_FROZEN`**: development-only robustness-first interpolation between `first_safe` and `k_rho` selects `lambda=0.875`; it passes all five development gates with mean/median PSNR delta vs T036 `+2.6360/+2.3057 dB`, `1/100` regressions vs T026, worst paired delta `-2.4274 dB`, and mean RGB-SSIM delta `+0.02044`.
- T067-C is accepted as **`INTERIOR_PROGRESS_TRANSFER_NEGATIVE`**: the exact frozen `lambda=0.875` rule preserves strong exposed-transfer utility and passes four of five gates, but worst paired PSNR delta vs T026 is `-6.9954830 dB`, below the unchanged `-5.614 dB` floor. Mean/median PSNR delta vs T036 is `+2.8848732/+2.5733133 dB`, regressions are `5/100`, and mean RGB-SSIM delta is `+0.0291559`.
- T067-D is accepted as **`INTERVAL_BOUNDARY_DIAGNOSIS_COMPLETE`**: across the frozen `[k_FS,k_rho]` intervals there are `2465` states and only `12` reference-unsafe states across `5/100` images. Three unsafe intervals later recover. The two non-recovering residual tails, indices 16 and 86, are both safe through absolute step 19 and first become unsafe at step 20; their normalized boundary locations differ (`q≈0.9779` vs `0.8726`).
- T068-A is accepted as **`ABS_STEP_CAP_DEV_NO_GAIN`**: the complete development-only global cap family `K=0..27`, with `k_K=max(k_FS,min(k_lambda,K))`, selects **K=27**, the no-cap control. K=25/26/27 are identical; K=20..27 pass the fixed gates, but no cap improves the predeclared robustness-first ranking over uncapped `lambda=0.875`. The absolute-step-cap family is closed.
- **Current bounded question:** whether an image-adaptive, parameter-free target-free signal derived from the trajectory itself can identify useful late-stage stopping inside the fixed `[k_FS,k_lambda]` interval. The currently authorized probe is objective-progress versus cumulative rendered-image-motion curve geometry, development-only.
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
- mean / median PSNR delta vs exact T036 `+3.8619303 / +3.8144067 dB`;
- `12/100` regressions vs T026;
- worst paired delta `-10.3649447 dB` — fail;
- mean RGB-SSIM delta `+0.0184218`.

T064-A re-rendering of frozen `k=0..27` states proves all `100/100` images have safety-reachable prefix checkpoints, with oracle worst paired delta `+0.4279520 dB`. Oracle information is diagnostic only and forbidden at inference.

### T066-A/B/C — dynamics signal exists, but late selected-tail monitoring fails

- T066-A development LOIO unsafe recall: `73/81 = 0.9012346`.
- T066-B transfer unsafe recall: `77/98` overall, `77/89` inside `k<=k_rho`, but `0/2` on unsafe selected base checkpoints.
- The two catastrophic base states are index/step `16/21` and `86/25`; both receive `p_safe≈1` while their true margins are `-7.1311 dB` and `-10.3649 dB`.
- T066-C shows all `77` correctly detected unsafe prefix states occur before first-safe. After first-safe there are reference-unsafe prefix states with no later classifier warning; neither catastrophic image exhibits safe→unsafe→safe re-entry.

**Scientific implication:** the frozen dynamics classifier is useful for identifying entry into a broadly safer region, but not as a reliable late-stage quality monitor.

### T067-A/B/C/D — safety entry and utility progress are complementary, but a global interval rule remains insufficient

T067-A exact first-safe stopping is globally too early: `9.2430042 dB / 0.2483034`, mean/median PSNR delta vs T036 `-2.3669107/-2.8148540 dB`, `75/100` regressions, worst `-6.1126334 dB`, SSIM delta `-0.1311716`; all gates fail. Yet it protects the two known catastrophic normalized-progress tails.

T067-B development-only global interpolation chooses `lambda=0.875` and passes all five gates:

- mean / median PSNR delta vs T036 `+2.6360346 / +2.3057191 dB`;
- `1/100` regressions vs T026;
- worst paired delta `-2.4273992 dB`;
- mean RGB-SSIM delta `+0.0204407`.

T067-C applies that exact frozen rule once to the exposed transfer cohort:

- absolute `14.4947881 dB / 0.4086309`;
- mean / median PSNR delta vs T036 `+2.8848732 / +2.5733133 dB` — pass;
- `5/100` regressions vs T026 — pass;
- worst paired delta vs T026 `-6.9954830 dB` — **fail**;
- mean RGB-SSIM delta vs T036 `+0.0291559` — pass.

Relative to T063-D, interpolation improves the worst tail by about `3.37 dB` but does not meet the fixed floor. Index 16 moves to step 19 and becomes safe (`-4.8984203 dB` vs T026); index 86 moves to step 21 but remains unsafe (`-6.9954830 dB`).

T067-D freezes the target-free interval table before reference access and then diagnoses `2465` inclusive `[k_FS,k_rho]` states. Only `12` are reference-unsafe across `5/100` images; three unsafe intervals later recover. The two non-recovering known tails both first become unsafe at absolute step 20, but their normalized-q boundaries differ substantially. This justified testing a global absolute-step family, not choosing a cutoff from exposed references.

### T068-A — global absolute-step family closes as no-gain

The full development-only family `K=0..27` is frozen before reference-quality access and independently verified. The no-cap control K=27 wins the predeclared ranking and exactly reproduces T067-B:

- mean / median PSNR delta vs T036 `+2.6360346 / +2.3057191 dB`;
- `1/100` regressions vs T026;
- worst paired delta `-2.4273992 dB`;
- mean RGB-SSIM delta `+0.0204407`.

K=25/26/27 yield identical choices and metrics; selected K=27 changes `0/100` choices. The verifier checks 2,800 candidates, 2,041 unique GPU renders and 2,800 independently recomputed quality states with maximum discrepancy about `1.09e-12`, `optimizer_runs=0`, `model_fits=0`.

**Scientific implication:** the exposed step-20 coincidence is not a portable global budget signal. Further progress should seek an image-adaptive target-free late-stage indicator rather than another globally calibrated step/q cutoff.

## Retained mechanism conclusions

- The 12-D renderer/trajectory has substantial usable capacity; the dominant unresolved issue is **target-free checkpoint selection/safety**, not reachable-state capacity.
- Large mean improvement of the zero-reference trajectory has replicated across multiple independent cohorts.
- Fixed stopping, cumulative loss balance, normalized progress alone, antithetic sensitivity, snapshot-based safety readouts, the current dynamics rollback guard, exact first-safe stopping, exact global `lambda=0.875` interpolation, and the global absolute-step-cap family are insufficient for qualification.
- `first_safe` and normalized progress are complementary: one gives a lower target-free entry endpoint, the other retains utility. Their global interpolation materially reduces tail harm but still misses a rare transferred tail.
- Unsafe interval behavior can be sparse, non-monotone and recovering. Neither a universal normalized-q cutoff nor a universal absolute-step cap is currently supported.
- The next justified probe is an **image-adaptive trajectory-geometry signal** that uses only quantities available from the degraded image, frozen rendered states and low-only objective history.
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

**T068-B — development-only objective–motion knee selector** in `coordination/CHATGPT_TO_CODEX.md`.

Keep T066-A/T067-B/T068-A machinery fixed. Within each development image's frozen `[k_FS,k_lambda]` interval, test exactly one parameter-free target-free knee rule based on low-only objective progress versus cumulative RMS rendered-image motion. Freeze all 100 target-free curve/choice identities before any development reference-quality access, then apply only the unchanged five gates. Do not open exposed-transfer references, a fresh cohort, official LOL-v2 Real test, LSRW or UHD-LL in this cycle.
