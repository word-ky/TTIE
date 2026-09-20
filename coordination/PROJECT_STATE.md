# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Enhancement**

This file is the live scientific state. Detailed task history remains in Git history and `research_log/`.

## Core scientific question

Can a vision system adapt a compact spatial image-processing state per test image, without test labels or clean/normal-light targets, so that unknown and spatially heterogeneous degradation is corrected by a reference-free test-time optimization field/objective?

## Current deployable state

- **Broad fresh-qualified Ours-Core:** T014 Sobolev Region2 TTT.
- **Fixed-validation deployable base:** T026-A, `11.1208764 dB / 0.3737918 RGB-SSIM` on its original validation cohort.
- **Fresh-qualified deployable action extension:** T036-A CommonRegion2/CommonBox 12-D gain path, `+0.9392571 dB` mean over exact T026-A on its fresh cohort, but with an unresolved tail (`29/100` regressions; worst `-5.614 dB`).
- T059/T060 action-transfer rescue is closed. T061 source-global fixed stopping is closed.
- T062 step 27 is **not** deployable/fresh-qualified: it retains a large mean gain but failed the immutable fresh worst-tail gate.
- T063-A proves strong checkpoint-selection headroom on the frozen T062 prefix.
- T063-B's cumulative loss-balance selector failed and is closed.
- T063-C normalized objective progress passes all five gates on an exposed transfer cohort, but T063-D shows that the exact frozen selector **fails fresh qualification on the rare worst-tail gate**. The normalized-progress rule is therefore closed as a qualification candidate pending mechanism diagnosis.
- Official LOL-v2 Real test and all cross-dataset held-out sets remain sealed.

The accepted T036/T062/T063 action family starts from each raw low image with identity state and uses the fixed 12-D EV/gamma/gain CommonRegion2/CommonBox renderer. T062/T063 keep this action space and Adam `lr=0.03`, replacing T014 energy with the fixed low-only objective `L_spa + 10 L_exp + 5 L_col`.

## Accepted T062 evidence

### T062-A — fixed zero-reference objective

On the original 100-image development cohort:

- absolute `14.9480877 dB / 0.3369354`;
- mean / median PSNR delta vs T036 `+3.7180475 / +3.8600704 dB`;
- `12/100` regressions vs T026;
- worst delta vs T026 `-7.1055215 dB` — safety fail;
- mean RGB-SSIM delta vs T036 `-0.00936695` — structure fail;
- minimum-objective selection lands at step 40 on `88/100` images.

Classification remains negative under its original contract, but the large PSNR signal shifted the bottleneck diagnosis away from renderer capacity and toward objective/stopping.

### T062-B — development-selected fixed step 27

Using only the frozen T062-A trajectory, development references select one global `k*=27`. On development:

- absolute `14.7025073 dB / 0.3582991`;
- mean / median PSNR delta vs T036 `+3.4724671 / +2.9745558 dB`;
- `5/100` regressions vs T026;
- worst delta vs T026 `-5.5802323 dB`;
- mean RGB-SSIM delta vs T036 `+0.0119967`.

This is a development hyperparameter result, not a final comparison.

### T062-C-R2 — fresh qualification of fixed step 27

On an independently frozen previously reference-unused 100-pair LOL-v2 Real Train cohort:

- absolute `15.6879652 dB / 0.4124338`;
- mean / median PSNR delta vs T036 `+3.5504315 / +3.3795780 dB`;
- `10/100` regressions vs T026;
- worst delta vs T026 `-7.3341753 dB` — immutable safety-gate fail;
- mean RGB-SSIM delta vs T036 `+0.0098393`.

Four of five gates pass, but the formal verdict is **NEGATIVE**. The fixed-global-step-27 route is closed. The replicated mean gain remains strong evidence that the T062 zero-reference trajectory is useful and that the remaining issue is concentrated in checkpoint selection/safety rather than gross renderer capacity.

## T063 stopping/selection diagnosis

### T063-A — accepted `SELECTION_HEADROOM_PRESENT`

T063-A re-renders only the already-frozen T062-C-R2 prefix `k=0..27`; all 2,800 outputs are frozen before any diagnostic reference read and every reconstructed step-27 output is bit-exact to accepted T062-C-R2. No optimizer/objective/action-space rerun occurs.

A strictly `REFERENCE_ORACLE_ONLY` safety-constrained oracle shows:

- safety-reachable prefix state: **`100/100` images**;
- mean / median PSNR delta vs T036: **`+5.2221903 / +4.2352859 dB`**;
- regressions vs T026: **`0/100`**;
- worst paired delta vs T026: **`+0.8757352 dB`**;
- mean RGB-SSIM delta vs T036: **`+0.0295648`**;
- oracle absolute mean: **`17.3597241 dB / 0.4321593 SSIM`**.

The sole fixed-step-27 safety failure, `Train/Low/low00221.png`, has safe saved states `12..24`; reference oracle step 17 gives `32.3537627 dB / 0.7653929 SSIM`, versus exact T026 `22.8328597 dB` and T062 step 27 `15.4986844 dB`.

**Scientific implication:** for the observed T062 tail, the frozen trajectory/action space already contains safe high-quality states. The primary bottleneck is checkpoint stopping/selection, not lack of reachable states. Reference oracle steps/harm labels are diagnostic only and are forbidden from deployable inference.

### T063-B — accepted `TRANSFER_NEGATIVE`

T063-B tested exactly one globally calibrated, target-free cumulative loss-balance statistic on already-frozen T062 states. Development calibration chose `tau=0.03832858496579632`, but the resulting selector chose **step 27 for all 100 development images and all 100 exposed-cohort transfer images**. Transfer therefore exactly reproduced T062-C-R2:

- mean / median PSNR delta vs T036 `+3.5504315 / +3.3795780 dB`;
- `10/100` regressions vs T026;
- worst paired delta vs T026 `-7.3341753 dB` — immutable safety-gate fail;
- mean RGB-SSIM delta vs T036 `+0.0098393`.

The selector/freeze boundary was valid and an independent verifier reproduced states, loss components, ratios, calibration, hashes, metrics, and gates. This closes the cumulative loss-balance ratio as a stopping statistic; it does not invalidate T063-A selection headroom.

### T063-C — accepted `TARGET_FREE_TRANSFER_PASS`

T063-C tests one frozen normalized objective-progress selector. On development, the unique selected global fraction is

`rho = 0.9857470621423519`.

For each image, using only its low/current rendered trajectory, define `L_best=min_{0..27} L_k`, `D=L_0-L_best`, and choose the earliest checkpoint satisfying `L_k <= L_0-rho*D` (with the frozen tiny-progress fallback). The rule inspects the complete `0..27` target-free trajectory; it is therefore a checkpoint selector, not a compute-saving early-stop claim.

On development:

- mean / median PSNR delta vs T036 `+3.5695323 / +3.0938391 dB`;
- `3/100` regressions vs T026;
- worst paired delta vs T026 `-4.0744464 dB`;
- mean RGB-SSIM delta vs T036 `+0.0150118`.

On the already reference-exposed T062-C-R2/T063-A cohort, with the selector frozen before any transfer quality read:

- absolute `15.8475933 dB / 0.4171932 RGB-SSIM`;
- mean / median PSNR delta vs T036 `+3.7100596 / +3.2033990 dB`;
- `10/100` regressions vs T026;
- worst paired delta vs T026 `-4.3235641 dB`;
- mean RGB-SSIM delta vs T036 `+0.0145986`;
- all five transfer gates pass;
- `45/100` transfer images select a checkpoint earlier than step 27.

The previous fixed-step failure `low00221.png` now selects step 22 and reaches `20.0067855 dB`, with paired delta vs T026 `-2.8260742 dB`, inside the safety envelope. Selector/output freezes precede all transfer reference/quality reads, and an independent verifier reconstructs the objective, normalized progress, candidate grid, tie-break, selected steps, hashes, metrics, and final classification.

**Scientific implication:** normalized objective progress can exploit part of the T063-A checkpoint headroom and repair the first exposed safety tail, but exposed-cohort success alone is insufficient evidence of qualification.

### T063-D — accepted `FRESH_QUALIFICATION_NEGATIVE`

The exact frozen T063-C rule and `rho=0.9857470621423519` were applied with no retuning to one deterministic 100-pair LOL-v2 Real Train cohort whose references had never previously been opened. The cohort is selected from 689 Train pairs after excluding 516 historically/reference-exposed pairs, leaving 173 untouched candidates; the sole cohort is the first 100 under `SHA256("T063D:" + normalized_relative_low_path)`.

Fresh result:

- absolute `15.4718452 dB / 0.3978969 RGB-SSIM`;
- mean / median PSNR delta vs exact T036 `+3.8619303 / +3.8144067 dB`;
- `12/100` regressions vs exact T026;
- worst paired delta vs exact T026 `-10.3649447 dB` — immutable safety-gate fail;
- mean RGB-SSIM delta vs exact T036 `+0.0184218`;
- `35/100` images select earlier than step 27;
- four of five fixed gates pass.

The worst case `Train/Low/low00262.png` selects step 25 and obtains `13.2478338 dB`, versus exact T026 `23.6127785 dB`. Two images violate the fixed safety threshold. All 300 outputs and 100 selector decisions were frozen before the first reference read; an independent verifier reconstructs the cohort, inference bindings, 2,800 T063 candidate states, control states, choices, hashes, metrics, and negative classification.

**Scientific implication:** the large mean benefit of the T062/T063 zero-reference trajectory generalizes again to genuinely fresh data, but normalized objective progress is not reliable enough for the rare safety tail. The exact selector is closed as a qualification candidate. The next question is whether the new tail is still checkpoint-selection-limited or whether some fresh images are trajectory-limited.

## Closed / retained mechanism conclusions

- T059/T060: learned field direction is real, but practical rescue failed fixed gates; line closed.
- T061: source-chosen global step `k=11` transfers poorly (`-2.2515 dB` mean vs T036; `92/100` regressions vs T026); source-global fixed stopping rejected.
- T063-B: cumulative spatial-cost/exposure-color-benefit ratio is non-discriminative under the frozen trajectory; statistic closed.
- T063-C/T063-D: normalized objective progress produces strong target-free mean gains and passes exposed transfer, but the exact frozen rule fails fresh worst-tail qualification and is closed as a qualification candidate.
- Renderer oracle studies T051/T054/T055 show substantial spatial capacity remains, but they are `REFERENCE_ORACLE_ONLY` and not deployable.

## Development versus final-evaluation protocol

The original fixed 100-image LOL-v2 Real Train-derived cohort is a **development set**. It may be used for method design, global hyperparameter selection, ablations, and failure analysis. It must not support final Ours-vs-baseline gap claims.

Final comparison rules:

- Freeze Final Ours, model assets, action space, optimizer/stopping rule, and all hyperparameters before held-out evaluation.
- Standard in-domain comparison must use the **complete official LOL-v2 Real test split** under the frozen inference protocol.
- A **cross-dataset/domain-shift held-out evaluation is required** for the unknown-degradation motivation, e.g. complete LSRW and UHD-LL test splits or equivalent fixed sets.
- No target-specific retraining/tuning is allowed on held-out sets; only per-image target-free test-time adaptation is permitted.
- Development baseline anchors are diagnostic only because strong released supervised checkpoints may be exposed to LOL-v2 Real training data.

Current development baseline anchors: Retinexformer T033-A `21.4787864 / 0.7900612`; SNR-Aware T045-A `23.3963299 / 0.8237644`.

## Information-boundary rules

- Test-time adaptation and checkpoint/state selection must never consume test labels, clean/normal-light targets, reference gradients/Jacobians, oracle values, PSNR/SSIM, degradation masks/gain maps, condition IDs, annotations, semantic image IDs, or per-image baseline outcome/harm labels.
- Quantities computed entirely from the current degraded image/current target-free intermediate image and frozen code/model are permissible.
- Development clean/reference targets may be used only offline for globally predeclared method development/hyperparameter selection; they may never become per-image inference inputs.
- Reference diagnostics may motivate only global research choices. Per-image oracle quantities remain forbidden from deployable inference.
- Fresh/final sets must remain isolated from method/hyperparameter selection until the corresponding rule is frozen.
- The official LOL-v2 Real test and cross-dataset held-out sets remain sealed.
- Fresh/test runs must fail closed on source/checkpoint/cohort/provenance mismatches.

## Current open task

**T064-A — frozen-prefix reachability diagnosis on the T063-D cohort** in `coordination/CHATGPT_TO_CODEX.md`.

Use only the accepted T063-D cohort and frozen `k=0..27` states. Re-render and freeze all 2,800 prefix outputs without rerunning optimization, then perform a strictly `REFERENCE_ORACLE_ONLY` safety-reachability diagnosis. Determine whether the new fresh tail is selection-limited or trajectory-limited. No new selector, tuning, fresh cohort, official test, or cross-dataset access is authorized.
