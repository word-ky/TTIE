# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Enhancement**

This file is the live scientific state. Detailed task-by-task history remains in Git history and `research_log/`.

## Core scientific question

Can a vision system adapt a compact spatial image-processing state per test image, without test labels or clean/normal-light targets, so that unknown and spatially heterogeneous degradation is corrected by a reference-free test-time optimization field/objective?

## Current deployable state

**T014 Sobolev Region2 TTT remains the broad fresh-qualified Ours-Core.** On LOL-v2 Real development data, the accepted fixed-validation deployable base remains **T026-A: `11.1208764 dB / 0.3737918 RGB-SSIM`** on its original validation cohort. T036-A common gain remains the strongest fresh-qualified deployable action expansion found so far, improving exact T026-A by `+0.9392571 dB / +0.0083560 RGB-SSIM` mean on its deterministic reference-unused fresh development cohort, but with an unresolved unsafe tail (`29/100` PSNR regressions; worst `-5.614 dB`).

The accepted T036 common path starts independently from each raw low image with identity state, jointly optimizes the fixed 12-D EV/gamma/gain action with Adam `lr=0.03` for 40 active steps plus CommonBox, and uses the original T014 scalar energy for trajectory gradients and minimum-energy/earliest-tie checkpoint selection.

No T059/T060 action-transfer result is deployable. The T059/T060 rescue line is closed. The source-global fixed-step T061 route is closed after T061-C. **T062-B is now the strongest development-selected target-time candidate**: it keeps the T062-A zero-reference trajectory but freezes one global development-selected stop at `k=27`, passing the preregistered development safety/efficacy gates. It is not yet fresh-qualified or held-out-tested and must not be promoted to final efficacy. The official LOL-v2 Real test and cross-dataset held-out sets remain sealed.

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

## T061 stopping-rule diagnosis — source-global route closed

T060-D-R2 also exposed large absolute source selection regret for both trajectories (`6.637364 dB` for literal T036/T014 and `6.729313 dB` for T060-C-R1), while T037-A had already shown `0.683234 dB` mean development reference-best headroom and `18/29` harmful T036 PSNR cases that an earlier state could rescue. Checkpoint selection therefore remained worth testing as a general T036 issue.

T061-B selected one immutable source-only global step using exact frozen source T036 trajectories: **`k*=11`**, with source mean PSNR `17.566028939521 dB`. The candidate was frozen without target/development access.

**T061-C is an accepted scientific negative.** Applying that same immutable `k=11` to the fixed 100-image LOL-v2 Real development cohort gives:

- mean / median PSNR delta versus original T036: **`-2.2515477 / -1.6503042 dB`**;
- improve / regress / tie versus T036: **`10 / 90 / 0`**;
- regressions versus exact T026-A: **`92/100`**;
- worst paired PSNR delta versus T026-A: **`-7.6103631 dB`**;
- mean RGB-SSIM delta versus T036: **`-0.0702668`**.

All five preregistered transfer gates fail. The procedure is admissible: `k=11` and the evaluation intent were frozen before development-quality fields were parsed; no second step, new optimizer/render, official-test access, or cross-dataset access occurred. Exact classification: **`a single source-chosen fixed stopping step does not transfer sufficiently`**.

Scientific implication: a source-chosen horizon does not transfer to the target domain. Do not try another source-global constant `k` or relax the T061 gates.

## T062 target-time objective and stopping diagnosis

**T062-A is an accepted scientific negative under its four-gate contract, but it materially changed the bottleneck diagnosis.** It keeps the exact T036 12-D CommonRegion2/CommonBox action space, identity initialization, Adam `0.03 × 40`, and low-only initial gate, replacing only the T014 learned trajectory/selection energy with the fixed label-free image objective `L_spa + 10 L_exp + 5 L_col`.

All 100 trajectories and minimum-objective selections froze before any clean/reference, PSNR/SSIM, T026/T036 outcome, official-test, or cross-dataset information was opened. T062-A development result:

- absolute: **`14.9480877 dB / 0.3369354 RGB-SSIM`**;
- mean / median PSNR delta versus T036: **`+3.7180475 / +3.8600704 dB`**;
- improve / regress / tie versus T036: **`85 / 15 / 0`**;
- regressions versus T026-A: **`12/100`**;
- worst PSNR delta versus T026-A: **`-7.1055215 dB`** — fails the fixed `>= -5.614 dB` safety gate;
- mean RGB-SSIM delta versus T036: **`-0.00936695`** — fails the fixed `>= -0.001` structure gate;
- minimum-objective selected step is `40` on **`88/100`** images.

Exact T062-A classification remains **`the fixed three-term zero-reference objective is insufficient`**. Do not relax those gates or sweep its weights/exposure target.

**T062-B is an accepted development PASS and materially changes the stopping diagnosis.** It does not rerun adaptation; it evaluates the exact frozen T062-A 100×41 trajectory and permits development clean references only for one global hyperparameter. Safety-eligible steps are `17..27`, and the single preregistered rule selects **`k*=27`**. At that immutable development-selected step:

- absolute T062-B: **`14.7025073 dB / 0.3582991 RGB-SSIM`**;
- mean / median PSNR delta versus T036: **`+3.4724671 / +2.9745558 dB`**;
- improve / regress / tie versus T036: **`94 / 6 / 0`**;
- regressions versus exact T026-A: **`5/100`**;
- worst PSNR delta versus T026-A: **`-5.5802323 dB`**, passing the fixed `-5.614 dB` gate by only about `0.034 dB`;
- mean RGB-SSIM delta versus T036: **`+0.0119967`**.

Every 4,100 frozen state identity and metric was independently verified. No official-test or cross-dataset data were accessed. Exact classification: **`the T062 zero-reference trajectory is useful and a global early stop rescues its safety/structure failure on development`**.

Scientific implication: on the same compact 12-D renderer, changing the target-time objective and stopping rule yields a large development gain while restoring the fixed safety/SSIM envelope. This is the strongest evidence so far that the immediate bottleneck lies in target-time objective/stopping rather than renderer capacity alone. However, `k=27` was selected on the same 100-image development cohort and its worst-tail margin is razor-thin. The next requirement is a fresh, previously unused train-split qualification with the rule frozen; no further choice may be made from the original development result before that check.

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
- Fresh qualification/final benchmark sets must remain isolated from method/hyperparameter selection until their corresponding rule is frozen.
- The official LOL-v2 Real test and cross-dataset held-out sets remain sealed.
- Fresh/test runs must fail closed on source/checkpoint/cohort/provenance mismatches.

## Best current methods / ceilings

- **Broad fresh-qualified Ours-Core:** T014 Sobolev Region2 TTT.
- **Best fixed-validation deployable candidate:** T026-A, `11.1208764 / 0.3737918` on its original validation cohort.
- **Fresh-qualified deployable action extension:** T036-A common gain, `+0.9392571 dB` mean over exact T026-A on its fresh cohort, unsafe tail unresolved (`29/100` regressions; worst `-5.614 dB`).
- **Strongest development target-time candidate:** T062-B global step 27, `14.7025073 / 0.3582991` on the T036 development cohort, `+3.4724671 dB` mean versus T036, with all fixed development gates passing; fresh/held-out generalization is untested.
- **Strongest minimum-objective control:** T062-A, `14.9480877 / 0.3369354`, larger PSNR but non-promoted because worst-tail and SSIM gates fail.
- **Best promoted non-deployable mechanism state:** T055-A, `24.3497922 / 0.7591279`.
- **T059/T060:** direction-transfer insight retained; practical rescue line closed.
- **T061:** source-global fixed stopping is rejected by T061-C; route closed.

## Current open task

**T062-C — fresh train-split qualification of frozen T062-B step 27** in `coordination/CHATGPT_TO_CODEX.md`.

Deterministically select one previously unused 100-pair LOL-v2 Real training-split cohort after excluding the original development 100. Freeze that cohort before any reference read. Run exact accepted T026-A, exact accepted T036-A, and the exact T062 zero-reference candidate with global stop `k=27` on low images only; freeze all outputs before opening normals. Apply the same fixed gain/safety/SSIM qualification gates. No alternate step, tuning, official-test access, or cross-dataset access is authorized.