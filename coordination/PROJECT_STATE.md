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
- T062/T063 establish a much stronger zero-reference trajectory: roughly `+3.5` to `+3.9 dB` mean PSNR over T036 replicates across independent 100-image cohorts, but fixed/global or normalized-progress stopping fails rare worst-tail qualification.
- T063-A and T064-A independently show those catastrophic tails are **selection-limited, not trajectory-limited**: every diagnosed image has a safe prefix checkpoint.
- T063-B cumulative loss-balance, T064-B antithetic sensitivity, T065-A ridge quality regression, T065-B class-balanced snapshot logistic, and T065-C snapshot 5-NN are closed.
- T066-A adds eight trajectory-dynamics features to the 11 snapshot features. Development leave-one-image-out unsafe recall rises to `73/81 = 0.9012`, proving temporal dynamics contain strong development safety information, but the frozen transfer guard still changes `0/100` checkpoints and misses both catastrophic selected states.
- T066-B is accepted as **`SELECTED_TAIL_SPECIFIC_FAILURE`**: the frozen T066-A guard detects `77/98` unsafe transfer states overall and `77/89` unsafe prefix states, yet detects `0/2` unsafe selected base checkpoints. The problem is not generic transfer blindness; it is concentrated at the late checkpoint chosen by normalized progress.
- T066-C is accepted as **`PREFIX_REENTRY_SIGNAL_ABSENT`**: neither catastrophic selected checkpoint has a predicted-safe → predicted-unsafe → predicted-safe excursion. All `77` correctly detected unsafe prefix states occur before first-safe; after first-safe there are `12` reference-unsafe prefix states with no later warning. The existing T066-A probability history therefore cannot justify hysteresis/re-entry rollback.
- T067-A is accepted as **`FIRST_SAFE_TRANSFER_NEGATIVE`**: exact earliest predicted-safe stopping protects the two previously catastrophic tails much better, but globally stops far too early (`64/100` at step 0 or 1), fails all five gates, and collapses mean utility. The T066-A crossing is therefore a **lower safety-entry signal, not a quality-optimal stopping event**.
- T067-B is accepted as **`INTERIOR_PROGRESS_DEV_CANDIDATE_FROZEN`**: the exact development-only robustness-first calibration selects strict-interior `lambda=0.875`, with mean/median PSNR delta vs T036 `+2.6360/+2.3057 dB`, `1/100` regressions vs T026, worst paired delta `-2.4274 dB`, and mean RGB-SSIM delta `+0.02044`. Endpoint `lambda=1` also passes but has a weaker worst tail (`-4.0744 dB`). This is development calibration only, not transfer evidence or qualification.
- The current bounded question is whether the exact frozen `lambda=0.875` interval selector transfers without retuning to the already-exposed T063-D/T064-A cohort. Fresh/final sets remain untouched.
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

The large mean signal is real; fixed step 27 is not qualified.

### T063-A — `SELECTION_HEADROOM_PRESENT` (`REFERENCE_ORACLE_ONLY`)

On the frozen T062-C-R2 prefix `k=0..27`:

- safety-reachable prefix state `100/100`;
- mean / median PSNR delta vs exact T036 `+5.2221903 / +4.2352859 dB`;
- regressions vs exact T026 `0/100`;
- worst paired delta `+0.8757352 dB`;
- mean RGB-SSIM delta vs T036 `+0.0295648`;
- oracle absolute `17.3597241 dB / 0.4321593`.

This proves checkpoint headroom but is diagnostic only; reference/oracle information is forbidden from deployable inference.

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
- mean RGB-SSIM delta `+0.0184218`.

The exact normalized-progress selector is closed as a qualification candidate.

### T064-A — `FRESH_TAIL_SELECTION_LIMITED` (`REFERENCE_ORACLE_ONLY`)

Re-rendering only the frozen T063-D `k=0..27` states shows:

- safety-reachable prefix state `100/100`;
- mean / median PSNR delta vs exact T036 `+5.4523215 / +4.8916423 dB`;
- regressions vs exact T026 `0/100`;
- worst paired delta `+0.4279520 dB`;
- mean RGB-SSIM delta `+0.0370898`;
- oracle absolute `17.0622363 dB / 0.4165648`.

Both catastrophic failures contain safe early checkpoints. Their oracle steps/ranges are reference-derived diagnostics and must never be hard-coded or used at inference.

### T065-A/B/C — snapshot safety representation: closed

The same fixed 11 target-free snapshot/state/image features fail under three different readouts:

- T065-A ridge regression: transfer worst delta remains `-10.3649447 dB`; catastrophic margins are severely overestimated.
- T065-B class-balanced logistic: `0/100` transfer rollback; both catastrophic states receive essentially unit safe probability.
- T065-C 5-NN: development LOIO unsafe recall only `1/81`; `0/100` transfer rollback; both catastrophic states have five safe-labeled nearest development neighbors.

Changing the readout over the exact 11 snapshot features is not a productive line.

### T066-A — 19-D trajectory-dynamics logistic guard: `TRANSFER_NEGATIVE`

T066-A appends exactly eight predeclared temporal features to the frozen 11 snapshot features and uses one class-balanced development-only logistic rollback guard.

Development leave-one-image-out state classification:

- safe→safe `2532`, safe→unsafe `187`;
- unsafe→safe `8`, unsafe→unsafe `73`;
- unsafe recall `73/81 = 0.9012346`;
- `0/100` development checkpoint choices change;
- all five development gates pass.

On the exposed T063-D/T064-A transfer cohort:

- `0/100` checkpoint choices change;
- absolute `15.4718452 dB / 0.3978969`;
- mean / median PSNR delta vs exact T036 `+3.8619303 / +3.8144067 dB`;
- `12/100` regressions vs exact T026;
- worst paired delta `-10.3649447 dB` — fail;
- mean RGB-SSIM delta `+0.0184218`;
- both catastrophic base states receive `p_safe ≈ 0.9999999999999`.

The information boundary is valid: development references are used only for offline development labels/training; transfer inference reads only degraded/current frozen trajectory quantities plus the frozen development-trained model; all choices/outputs freeze before transfer reference-quality reads.

### T066-B — frozen 19-D support diagnosis: `SELECTED_TAIL_SPECIFIC_FAILURE`

Using the exact frozen T066-A representation/model with no optimizer rerun and no model fit:

- all transfer states: safe→safe `2500`, safe→unsafe `202`, unsafe→safe `21`, unsafe→unsafe `77`; unsafe recall `77/98 = 0.7857143`;
- prefix `k<=k_rho`: safe→safe `2439`, safe→unsafe `202`, unsafe→safe `12`, unsafe→unsafe `77`; unsafe recall `77/89 = 0.8651685`;
- selected base `k=k_rho`: `98` safe bases correctly predicted safe, but both unsafe bases are predicted safe; unsafe recall `0/2`.

The two catastrophic selected states are:

- index/step `16/21`: `p_safe≈1`, true quality margin `-7.1311 dB`;
- index/step `86/25`: `p_safe≈1`, true quality margin `-10.3649 dB`.

The complete target-free tables were frozen before any transfer reference-quality read. Independent verification reproduced target-free features/probabilities, post-freeze labels, support distances, confusion matrices, tail rows, and the final category. `optimizer_runs=0`, `model_fits=0`.

### T066-C — frozen prefix re-entry diagnosis: `PREFIX_REENTRY_SIGNAL_ABSENT`

Using only frozen T066-B/T066-A `p_safe(k)` sequences and base steps, the complete 100-image event table was frozen before joining any exposed reference-derived labels.

Key results:

- safe bases without re-entry: `94`; safe bases with re-entry: `4`;
- unsafe bases without re-entry: `2`; unsafe bases with re-entry: `0`;
- all `98` reference-unsafe states are accounted for as: `77` before first-safe, `0` during a post-safe unsafe excursion, `0` after safe re-entry, `12` inside the prefix after first-safe with no prior warning, and `9` outside the fixed prefix;
- index `16`, base `21`: `first_safe=12`, then predicted safe continuously through base;
- index `86`, base `25`: `first_safe=9`, then predicted safe continuously through base;
- independent verifier PASS; `optimizer_runs=0`, `model_fits=0`.

**Scientific implication:** the T066-A classifier's useful unsafe-state recall is concentrated before its first safe crossing. Once it enters the safe region it can remain confidently safe while true quality later deteriorates. A history-aware re-entry/hysteresis rule based on the same probability sequence is therefore not supported.

### T067-A — exact first-safe audit: `FIRST_SAFE_TRANSFER_NEGATIVE`

Using the exact frozen earliest predicted-safe event on the same exposed 100-image cohort, with every choice/output frozen before reference-quality access:

- absolute `9.2430042 dB / 0.2483034 RGB-SSIM`;
- mean / median PSNR delta vs exact T036 `-2.3669107 / -2.8148540 dB`;
- `75/100` regressions vs exact T026;
- worst paired delta vs exact T026 `-6.1126334 dB`;
- mean RGB-SSIM delta vs T036 `-0.1311716`;
- all five gates fail;
- `31/100` select step 0 and `33/100` select step 1; all `100/100` choices move earlier than `k_rho`.

The two previous catastrophic normalized-progress tails become much safer under first-safe: index 16 selects step 12 with `+3.4529 dB` vs T026; index 86 selects step 9 with `-4.2099 dB` vs T026, inside the fixed `-5.614 dB` safety floor. This demonstrates that the first-safe signal contains useful lower-bound safety information but is not a useful global stopping event because it can fire before meaningful enhancement utility has accumulated. Independent verification passed with `optimizer_runs=0`, `model_fits=0` and no new/final data access.

### T067-B — development-only interval interpolation: `INTERIOR_PROGRESS_DEV_CANDIDATE_FROZEN`

Using the exact predeclared nine-value global interpolation grid between frozen `first_safe` and frozen `k_rho`, all 900 target-free candidate choices/output identities were frozen before any development-quality read. The fixed robustness-first selection chooses `lambda=0.875`.

For `lambda=0.875` on the original development cohort:

- mean / median PSNR delta vs exact T036 `+2.6360346 / +2.3057191 dB`;
- `1/100` regressions vs exact T026;
- worst paired PSNR delta vs exact T026 `-2.4273992 dB`;
- mean RGB-SSIM delta vs T036 `+0.0204407`;
- all five gates pass.

The endpoint `lambda=1` also passes with larger mean PSNR (`+3.5695323 dB`) but weaker worst paired delta (`-4.0744464 dB`), so the predeclared robustness-first ranking selects the strict-interior candidate. `optimizer_runs=0`, `model_fits=0`; independent verification reproduces all candidate choices, metrics, gates, and the selected lambda. No exposed-transfer, fresh, official-test, or cross-dataset reference was accessed.

**Scientific implication:** the development data support a simple global combination of the safety-entry endpoint and normalized-progress utility signal. This remains in-sample global calibration because both the T066-A model and `lambda` were learned/selected from the original development cohort; transfer generalization is unestablished until the exact frozen rule is audited without retuning.

## Retained mechanism conclusions

- The 12-D renderer/trajectory has substantial usable capacity; the dominant unresolved issue remains **target-free checkpoint selection/safety**, not absence of reachable good states.
- Large mean improvement of the T062/T063 zero-reference trajectory has replicated on multiple independent 100-image cohorts.
- Global fixed stopping, cumulative loss-balance, normalized progress alone, antithetic sensitivity, multiple snapshot-based safety readouts, the exact T066-A rollback guard, and exact first-safe stopping are insufficient.
- Explicit trajectory dynamics carry strong development and substantial transfer unsafe-state signal, but the current frozen dynamics classifier is not a reliable monitor after its first safe crossing.
- T067-A adds an important distinction: `first_safe` is useful as a **safety-entry endpoint** for the two known catastrophic tails, yet is far too early for many ordinary images.
- T067-B provides the first strict-interior global development candidate that explicitly combines safety-entry with target-free utility/progress and passes all five development gates; whether this compromise transfers is still unknown.
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

The original fixed 100-image LOL-v2 Real Train-derived cohort is a **development set**. It may be used for method design, global training/hyperparameter selection, ablations, and failure analysis. Exposed transfer cohorts may support diagnostics/method development only after explicit freezing; they are not fresh qualification once their references have been inspected. None of these cohorts may support final Ours-vs-baseline gap claims.

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

**T067-C — frozen `lambda=0.875` exposed-transfer audit** in `coordination/CHATGPT_TO_CODEX.md`.

Apply the exact development-frozen interpolation rule once to the already-exposed T063-D/T064-A cohort. Freeze all 100 target-free checkpoint choices/output hashes before any transfer reference-quality read, then evaluate the unchanged five gates. No retuning, alternate lambda/rule, fresh cohort, official LOL-v2 Real test, LSRW, or UHD-LL access is authorized in this cycle.