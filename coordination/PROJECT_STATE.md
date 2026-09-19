# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Enhancement**

This file is the live scientific state. Detailed task-by-task history remains in Git history and `research_log/`.

## Core scientific question

Can a vision system adapt a compact spatial image-processing state per test image, without test labels or clean/normal-light targets, so that unknown and spatially heterogeneous degradation is corrected by a reference-free test-time optimization field/objective?

## Current deployable state

**T014 Sobolev Region2 TTT remains the broad fresh-qualified Ours-Core.** On LOL-v2 Real development data, the accepted fixed-validation deployable base remains **T026-A: `11.1208764 dB / 0.3737918 RGB-SSIM`**. T036-A common gain remains the strongest fresh-qualified deployable action expansion found so far, improving exact T026-A by `+0.9392571 dB / +0.0083560 RGB-SSIM` mean on its deterministic reference-unused development cohort, but with an unresolved unsafe tail (`29/100` PSNR regressions; worst `-5.614 dB`).

The accepted T036 common path starts independently from each raw low image with identity state, jointly optimizes the fixed 12-D EV/gamma/gain action with Adam `lr=0.03` for 40 active steps plus CommonBox, and uses the original T014 scalar energy for trajectory gradients and minimum-energy/earliest-tie checkpoint selection.

No T059/T060 action-transfer result is deployable. The T059/T060 rescue line is closed. The fixed-global-step T061 route is also closed after T061-C. The official LOL-v2 Real test remains sealed.

## Renderer / action-family diagnosis

Reference-only renderer studies show that the major remaining capacity is real and strongly spatial:

- **T051-A spatial exposure field:** `+0.6590052 dB` mean / `+0.4525454 dB` median PSNR over T050, `100/100` PSNR wins.
- **T054-A local-detail field:** reaches `24.3454867 / 0.7577572`, adding `+1.2383356 dB` mean / `+1.2353360 dB` median PSNR and `+0.1877170` mean SSIM over T052.
- T055-A/T055-V show that simply extending the same one-scale optimization budget adds only about `+0.0043 dB`; T056 second-scale detail and T057 chroma-detail failed their fixed materiality gates.

These are `REFERENCE_ORACLE_ONLY` diagnostics. Their clean targets, reference gradients/states, PSNR/SSIM, and per-image oracle quantities are forbidden from test-time inference.

## T059/T060 optimization-field line — closed

Accepted conclusions:

- calibrated unseen-image scalar prediction is unsupported despite strong in-sample capacity;
- the learned field transfers direction substantially better than scalar value;
- T059-E source inner-held direction is real, and degraded-image Jacobian projection can be reconstructed online without reference-dependent caches;
- the tested one-step detail integration is materially negligible on the 100-image development cohort;
- direct 8×8 exposure-field transfer fails shape/coverage;
- in the exact T036 4-D common-gain subspace, T059-E is better first-order aligned than T014 (`58/60` positive-dot vs `53/60`, median cosine `0.9064` vs `0.8009`), but the fixed finite-step T060-C-R1 hybrid does not beat T036 and misses the preregistered worst-tail safety gate;
- T060-D-R2 shows that T014 checkpoint selection mismatch is not a sufficient explanation for the T060-C-R1 near-miss. The fixed oracle-wins and regret gates fail (`35/60` wins; regret difference `+0.09195 dB`), so the T059/T060 rescue line is closed.

T060-C-R1 remains a useful mechanism clue: relative to T036 it preserves nearly all mean quality while reducing regressions from `29/100` to `20/100` and improving the worst case from `-5.614` to `-3.578 dB`, but it is not deployable.

## T061 stopping-rule diagnosis — fixed-global-step route closed

T060-D-R2 also exposed large absolute source selection regret for both trajectories (`6.637364 dB` for literal T036/T014 and `6.729313 dB` for T060-C-R1), while T037-A had already shown `0.683234 dB` mean development reference-best headroom and `18/29` harmful T036 PSNR cases that an earlier state could rescue. Checkpoint selection therefore remained worth testing as a general T036 issue.

T061-B selected one immutable source-only global step using exact frozen source T036 trajectories: **`k*=11`**, with source mean PSNR `17.566028939521 dB`. The candidate was frozen without target/development access.

**T061-C is an accepted scientific negative.** Applying that same immutable `k=11` to the fixed 100-image LOL-v2 Real development cohort gives:

- mean / median PSNR delta versus original T036: **`-2.2515477 / -1.6503042 dB`**;
- improve / regress / tie versus T036: **`10 / 90 / 0`**;
- regressions versus exact T026-A: **`92/100`**;
- worst paired PSNR delta versus T026-A: **`-7.6103631 dB`**;
- mean RGB-SSIM delta versus T036: **`-0.0702668`**.

All five preregistered transfer gates fail. The procedure is admissible: `k=11` and the evaluation intent were frozen before development-quality fields were parsed; no second step, new optimizer/render, official-test access, or cross-dataset access occurred. Exact classification: **`a single source-chosen fixed stopping step does not transfer sufficiently`**.

Scientific implication: the useful horizon is strongly image/domain dependent; a source-global stopping time is not a viable solution. Do not try another constant `k` or relax the T061 gates. Historical learned/heuristic selector work also contains multiple failures, so the next active question is whether the exact T036 action space can instead be driven by a genuinely target-time, reference-free image-space objective.

## Development versus final-evaluation protocol

The fixed 100-image LOL-v2 Real cohort drawn from the training split is explicitly a **development set**. It may be used for predeclared method design, hyperparameter selection, ablations, and failure analysis. It must **not** be used to state the final Ours-vs-baseline performance gap, because strong released supervised baselines can be training-exposed to this split.

Final comparison rules:

- Freeze the final Ours method, model assets, action space, optimizer/stopping rule, and all hyperparameters before final held-out evaluation.
- The standard in-domain comparison must use the **complete official LOL-v2 Real test split** under the same frozen inference protocol.
- A **cross-dataset / domain-shift held-out evaluation is required** for the unknown-degradation motivation. Candidate complete held-out test splits include LSRW and UHD-LL (or equivalent fixed datasets), with the same frozen Ours checkpoint/rule and no target-specific retraining or tuning.
- Development baseline numbers are diagnostic anchors only. Final baseline-gap claims must come from complete held-out test sets with training exposure/protocol disclosed.

The official LOL-v2 Real test and cross-dataset held-out test sets remain sealed until Final Ours is frozen.

## Strong baseline development anchors

- **Retinexformer T033-A:** `21.4787864 / 0.7900612`.
- **SNR-Aware T045-A:** `23.3963299 / 0.8237644`.

Both are target-free at inference under the frozen comparison protocol, but their released supervised checkpoints are exposed to LOL-v2 Real training data containing this development split. They are diagnostic anchors, not valid final Ours-vs-baseline gap estimates.

## Information-boundary rules

- Test-time adaptation and checkpoint/state selection must never consume test labels, clean/normal-light targets, reference gradients/Jacobians, oracle values, PSNR/SSIM, degradation masks/gain maps, condition IDs, annotations, semantic image IDs, or per-image baseline outcome/harm labels.
- A quantity computed entirely from the current degraded image/current target-free intermediate image and frozen model/code is permissible; anything depending on a clean/reference target is not.
- Development clean/reference targets may be used only offline for globally predeclared method development/evaluation and may never become per-image inference inputs or selectors.
- Source-training clean/reference targets may be used for source-supervised training or isolated source-domain diagnostics; they are never admissible test-time inputs.
- Reference diagnostics may motivate only global research choices; no per-image oracle quantity may enter deployable inference.
- External baselines admitted to final main comparisons must be target-free at inference, and their training exposure must be disclosed.
- Fresh/final benchmark sets must remain isolated from model/hyperparameter selection.
- The official LOL-v2 Real test and cross-dataset held-out sets remain sealed.
- Fresh/test runs must fail closed on source/checkpoint/cohort/provenance mismatches.

## Best current methods / ceilings

- **Broad fresh-qualified Ours-Core:** T014 Sobolev Region2 TTT.
- **Best fixed-validation deployable candidate:** T026-A, `11.1208764 / 0.3737918`.
- **Fresh-qualified deployable action extension:** T036-A common gain, `+0.9392571 dB` mean on its fresh cohort, unsafe tail unresolved (`29/100` regressions; worst `-5.614 dB`).
- **Best promoted non-deployable mechanism state:** T055-A, `24.3497922 / 0.7591279`.
- **T059/T060:** direction-transfer insight retained; practical rescue line closed.
- **T061:** source-global fixed stopping is rejected by T061-C; route closed.

## Current open task

**T062-A — fixed zero-reference objective control on the exact T036 action space** in `coordination/CHATGPT_TO_CODEX.md`.

Keep the exact T036 renderer/action space, identity initialization, CommonBox, Adam `0.03`, and 40-step budget. Replace only the learned scalar objective with the one fixed three-term no-reference image-space objective specified in the task. Freeze every 100-image trajectory, objective value, output, and selected step before any development reference or baseline outcome is read. No weight sweep, hybrid objective, official-test access, or cross-dataset access is authorized.
