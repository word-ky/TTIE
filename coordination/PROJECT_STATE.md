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
- Official LOL-v2 Real test and all cross-dataset held-out sets remain sealed.

The accepted T036/T062 action family starts from each raw low image with identity state and uses the fixed 12-D EV/gamma/gain CommonRegion2/CommonBox renderer. T062 keeps this action space and Adam `lr=0.03`, replacing T014 energy with the fixed low-only objective `L_spa + 10 L_exp + 5 L_col`.

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

This is the strongest development target-time candidate, not a final result.

### T062-C-R2 — fresh qualification of fixed step 27

On an independently frozen previously reference-unused 100-pair LOL-v2 Real Train cohort:

- absolute `15.6879652 dB / 0.4124338`;
- mean / median PSNR delta vs T036 `+3.5504315 / +3.3795780 dB`;
- `10/100` regressions vs T026;
- worst delta vs T026 `-7.3341753 dB` — immutable safety-gate fail;
- mean RGB-SSIM delta vs T036 `+0.0098393`.

Four of five gates pass, but the formal verdict is **NEGATIVE**. The fixed-global-step-27 route is closed. The replicated mean gain is nevertheless strong evidence that the T062 zero-reference trajectory is useful and that the remaining issue is concentrated in a rare stopping/safety tail.

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

**Scientific implication:** for the observed T062 tail, the frozen trajectory/action space already contains safe high-quality states. The primary bottleneck is now checkpoint stopping/selection, not lack of reachable states. Reference oracle steps/harm labels are diagnostic only and are forbidden from deployable inference.

## Closed / retained mechanism conclusions

- T059/T060: learned field direction is real, but practical rescue failed fixed gates; line closed.
- T061: source-chosen global step `k=11` transfers poorly (`-2.2515 dB` mean vs T036; `92/100` regressions vs T026); source-global fixed stopping rejected.
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

**T063-B — target-free loss-balance stopping transfer audit** in `coordination/CHATGPT_TO_CODEX.md`.

Use only frozen T062 trajectories. Calibrate one global scalar threshold on the original development cohort for the predeclared target-free ratio between spatial-consistency cost and exposure/color benefit, freeze the rule, then apply it unchanged to the T062-C-R2/T063-A cohort before reading any reference-derived transfer metrics. This is an exposed-cohort transfer audit, not fresh qualification. No learned selector, second heuristic, new optimizer run, new cohort, official test, or cross-dataset access is authorized.
