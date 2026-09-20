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
- T062/T063 establish a much stronger zero-reference trajectory: mean gains of roughly `+3.5` to `+3.9 dB` over T036 replicate across independent 100-image cohorts, but fixed/global or normalized-progress checkpoint rules fail rare worst-tail qualification.
- T063-A and T064-A independently show the relevant catastrophic tails are **selection-limited, not trajectory-limited**: every image in each diagnosed cohort has a safe prefix checkpoint.
- T063-B cumulative loss-balance, T064-B antithetic sensitivity, T065-A ridge quality regression, T065-B class-balanced snapshot logistic, and T065-C snapshot 5-NN are closed.
- T065-C showed the 11-D snapshot representation itself lacks simple cross-image transferable safety information: development LOIO identified only `1/81` unsafe states.
- **T066-A materially changes that diagnosis:** adding exactly eight trajectory-dynamics features raises development leave-one-image-out unsafe-state recall to `73/81 = 0.9012`, proving temporal dynamics carry cross-image safety signal inside the development distribution. However, the frozen all-development 19-D logistic guard changes `0/100` transfer checkpoints and assigns essentially unit safe probability to both catastrophic transfer base states. Thus the unresolved issue is now **cross-cohort transfer of the safety representation / boundary**, not absence of any development safety signal.
- **T066-B is the current bounded diagnostic:** keep the entire T066-A representation/model frozen and determine whether transfer-unsafe states are geometrically supported by development-unsafe states or instead occupy development-safe support regions.
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

- safety-reachable prefix state `100/100`;
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

- safety-reachable prefix state `100/100`;
- mean / median PSNR delta vs exact T036 `+5.4523215 / +4.8916423 dB`;
- regressions vs exact T026 `0/100`;
- worst paired delta `+0.4279520 dB`;
- mean RGB-SSIM delta `+0.0370898`;
- oracle absolute `17.0622363 dB / 0.4165648`.

Both catastrophic failures contain safe early checkpoints. Their oracle steps/ranges are reference-derived diagnostics and must never be hard-coded or used at inference.

### T065-A/B/C — snapshot safety representation: closed

The same fixed 11 target-free snapshot/state/image features fail under three different readouts:

- T065-A ridge regression: transfer worst delta remains `-10.3649447 dB` and the two catastrophic margins are severely overestimated.
- T065-B class-balanced logistic: `0/100` transfer rollback; both catastrophic states receive essentially unit safe probability.
- T065-C 5-NN: development LOIO unsafe recall only `1/81`; `0/100` transfer rollback; both catastrophic states have five safe-labeled nearest development neighbors.

This establishes that changing the readout over the exact 11 snapshot features is not a productive line.

### T066-A — 19-D trajectory-dynamics logistic guard: `TRANSFER_NEGATIVE`

T066-A appends exactly eight predeclared temporal features to the frozen 11 snapshot features and uses one class-balanced development-only logistic rollback guard.

Development leave-one-image-out state classification:

- safe→safe `2532`, safe→unsafe `187`;
- unsafe→safe `8`, unsafe→unsafe `73`;
- unsafe recall `73/81 = 0.9012346`;
- `0/100` development checkpoint choices change;
- all five development gates pass.

On the already reference-exposed T063-D/T064-A transfer cohort:

- `0/100` checkpoint choices change;
- absolute `15.4718452 dB / 0.3978969`;
- mean / median PSNR delta vs exact T036 `+3.8619303 / +3.8144067 dB`;
- `12/100` regressions vs exact T026;
- worst paired delta `-10.3649447 dB` — fail;
- mean RGB-SSIM delta `+0.0184218`;
- both catastrophic base states (steps 21/25) receive `p_safe ≈ 0.9999999999999`.

The information boundary is valid: development references are used only for offline development labels/training; transfer inference reads only degraded/current frozen trajectory quantities plus the frozen development-trained model; all choices/outputs freeze before transfer reference-quality reads. Independent verification reproduces all features, fits, probabilities, hashes, ordering, metrics, and classification.

**Scientific implication:** trajectory dynamics do contain strong cross-image safety information on development, unlike the snapshot representation, but that signal does not transfer to the exposed cohort under the frozen model. The next question is whether this is a support/representation shift versus a decision-boundary mismatch. Do not add another classifier or heuristic until that is diagnosed.

## Retained mechanism conclusions

- The 12-D renderer/trajectory has substantial usable capacity; the dominant unresolved issue is **target-free checkpoint selection/safety**, not absence of reachable good states.
- Large mean improvement of the T062/T063 zero-reference trajectory has replicated on multiple independent 100-image cohorts.
- Global fixed stopping, cumulative loss-balance, normalized progress alone, antithetic sensitivity, and multiple snapshot-based safety readouts are insufficient for the rare worst tail.
- Explicit trajectory dynamics restore strong development cross-image unsafe-state recall, but the current frozen dynamics guard fails transfer completely at the checkpoint level.
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
- Exposed transfer references may be used only for explicitly post-freeze diagnosis/evaluation; they must never feed back into the same task's feature construction, fitting, thresholding, or selection.
- Fresh/final sets must remain isolated from method/hyperparameter selection until the corresponding rule is frozen.
- Official LOL-v2 Real test and cross-dataset held-out sets remain sealed until Final Ours is frozen.
- Fresh/test runs must fail closed on source/checkpoint/cohort/provenance mismatches.

## Current open task

**T066-B — frozen 19-D cross-cohort safety-support diagnosis** in `coordination/CHATGPT_TO_CODEX.md`.

Keep the exact T066-A feature representation, normalization, model, threshold, trajectory, and checkpoint rule frozen. Freeze the full transfer target-free feature/probability table before any diagnostic reference read, then post-hoc label the already exposed transfer states and compare their development-normalized 19-D geometry to development-safe versus development-unsafe support. Classify the failure as support shift, boundary mismatch, selected-tail-specific failure, or no diagnostic failure under the predeclared criteria. No new selector, model, feature, optimizer run, new cohort, official test, or cross-dataset access is authorized.