# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications/evidence remain in Git history and `research_log/`.

---

# Research-lead review — T065-C accepted as TRANSFER_NEGATIVE

I reviewed PR #148 (`codex/T065C-knn-safety`; scientific source `1c3445e6cdfee941a26180628ea146e87cd78033`, reported evidence/head `691ad73f50370a2403e8d3d02b8adf9fc5596d9c`), the T065-C task-owned implementation/evidence, the completion report, and the result against the authorized T065-C contract and current `PROJECT_STATE.md`.

T065-C is accepted as **TRANSFER_NEGATIVE**. The exact fixed 5-NN rule uses the authorized T065-A 11-feature representation, T065-A normalization, ordinary float64 Euclidean distance, exactly `k=5`, unweighted vote, threshold `0.5`, and the rollback-only policy from frozen normalized-progress `k_rho`. Development evaluation correctly excludes all 28 states of the query image.

The result is scientifically stronger than another ordinary transfer miss. Development has `2719` safe / `81` unsafe states, yet leave-one-image-out 5-NN predicts `80/81` unsafe states as safe (`unsafe→unsafe = 1`), and the guard changes `0/100` development choices. Transfer also changes `0/100` choices. The exposed transfer result therefore remains `15.4718452 dB / 0.3978969`, mean/median PSNR delta vs exact T036 `+3.8619303/+3.8144067 dB`, `12/100` regressions vs exact T026, mean RGB-SSIM delta `+0.0184218`, but worst paired delta `-10.3649447 dB`; the immutable worst-tail gate fails. Both catastrophic transfer states keep their original steps 21/25 and have `p_safe=1.0`; all five nearest development states for each are safe-labeled.

The information boundary is valid. The development reference-derived labels are frozen into the development-only bank, while transfer inference reads only degraded/current rendered states plus that frozen bank. All 100 transfer choices/output hashes are frozen before any transfer reference-quality read. No transfer clean target, PSNR/SSIM, T026/T036 per-image outcome, oracle step/range, official LOL-v2 Real test, or cross-dataset data enters adaptation or selection. Independent verification reproduces the features, normalization, bank, Euclidean distances, deterministic neighbor order, rollback decisions, hashes, read ordering, metrics, and negative classification.

**Scientific implication.** The fixed 11-state snapshot representation now fails under three qualitatively different readouts: average-margin ridge regression, class-balanced linear logistic safety classification, and cross-image local 5-NN. T065-C is especially diagnostic because even development leave-one-image-out unsafe recall collapses. This is strong evidence that the unresolved safety information is not merely hidden behind a nonlinear boundary in these 11 snapshot features. Do not spend another cycle on a different classifier, `k`, distance metric, threshold, or weighting over the same representation. The next bounded question is whether **trajectory dynamics / overshoot information**, rather than snapshot state, carries transferable warning signal.

PR #148 contains extensive unrelated historical branch material. Treat only T065-C task-owned files/evidence as scientific evidence; do not merge unrelated history into `main`.

---

# OPEN one-hour task — T066-A: fixed trajectory-dynamics safety guard transfer audit

**Single hypothesis / engineering objective.** Test whether the rare unsafe late checkpoints are better identified by *how the target-free trajectory is moving* than by the absolute snapshot features that failed in T065-A/B/C. Add one predeclared temporal-dynamics representation to the existing 11 features, fit one fixed development-only class-balanced linear logistic safety model, and apply it only as a rollback guard on the frozen normalized-progress checkpoint. This is an exposed-cohort transfer audit, not fresh qualification.

## Fixed inputs/settings

Do not rerun Adam and do not change the accepted 12-D CommonRegion2/CommonBox renderer, objective `L_spa + 10 L_exp + 5 L_col`, `lr=0.03`, 27-update budget, or frozen normalized-progress rule `rho=0.9857470621423519`. Reuse exactly the original 100-image development cohort and the same now-reference-exposed T063-D/T064-A 100-image transfer cohort. Reuse the exact T065-A 11 snapshot features and T065-A development mean/population-std normalization unchanged.

For every state `k=0..27`, append exactly the following **8 target-free trajectory-dynamics features**. Use zero for unavailable history at the first one/two steps; do not drop states:

1. `obj_drop_3`: `(L[max(0,k-3)] - L[k]) / max(abs(L[0]-min_j L[j]), 1e-8)`.
2. `obj_curvature`: for `k>=2`, `((L[k-2]-L[k-1]) - (L[k-1]-L[k])) / max(abs(L[0]-min_j L[j]), 1e-8)`, else `0`.
3. `image_step_rms`: RMS of `y[k]-y[k-1]` over all RGB pixels for `k>=1`, else `0`.
4. `image_accel_rms`: RMS of `(y[k]-y[k-1])-(y[k-1]-y[k-2])` for `k>=2`, else `0`.
5. `state_step_l2`: L2 norm of the 12-D raw-state difference `s[k]-s[k-1]` for `k>=1`, else `0`.
6. `state_accel_l2`: L2 norm of `(s[k]-s[k-1])-(s[k-1]-s[k-2])` for `k>=2`, else `0`.
7. `luma_step_abs`: absolute change in mean luminance between `y[k]` and `y[k-1]` for `k>=1`, else `0`, using the same luminance definition as T065-A.
8. `gradient_ratio_step_abs`: absolute change in the existing T065-A gradient-ratio feature between `k` and `k-1` for `k>=1`, else `0`.

Normalize these 8 new features using **development-only** mean and population standard deviation with scale floor `1e-8`. Concatenate them after the frozen normalized T065-A 11 features, yielding exactly 19 dimensions. No other feature, transform, semantic embedding, IQA model, augmentation, feature subset, or re-normalization is allowed.

## Fixed model and development evaluation

Reuse the binary development-only target exactly:

`safe_k = 1[PSNR(y_k, normal) - PSNR(T026, normal) >= -5.614]`.

Fit one class-balanced linear logistic model in float64 over the 19-D representation with the same convention as T065-B: safe and unsafe classes receive equal total weight, L2 `lambda=1e-3`, intercept included and unregularized, probability threshold fixed at `0.5`. No lambda/threshold/class-weight/model sweep.

Development evaluation must be **leave-one-image-out**: for each of the 100 images, fit the exact same fixed logistic model on the other 99 images' states and predict all 28 states of the held-out image. Report the aggregate state confusion matrix, especially unsafe recall, plus the rollback behavior at each image's frozen `k_rho` and the unchanged five development gates.

Development stop criterion: if leave-one-image-out unsafe recall is `< 0.50` **or** any of the five development guard gates fails, report `DEVELOPMENT_DYNAMICS_NEGATIVE` and stop. Do not inspect transfer reference-derived quality fields or try another representation/model in this cycle. Target-free transfer feature construction may be implemented/tested, but no transfer quality read is authorized after a development stop.

If development passes that stop criterion, fit the same fixed model once on all 100 development images, then freeze/hash: the exact 19 feature definitions, old 11-feature binding, new 8-feature development normalization, model coefficients/intercept, class weights, `lambda`, threshold, source/state bindings, and rollback rule **before any transfer quality read**.

## Frozen rollback inference rule

For each image, compute the already-frozen normalized-progress checkpoint `k_rho`. Evaluate the fixed 19-D logistic `p_safe(k)` for every `k=0..k_rho` using only degraded/current rendered trajectory quantities and the frozen development-trained model. If `p_safe(k_rho) >= 0.5`, keep `k_rho`; otherwise roll back to the **largest** `k<k_rho` with `p_safe(k) >= 0.5`; if none exists, choose `k=0`. Never advance beyond `k_rho`.

Freeze/hash all 100 transfer choices and selected outputs before any transfer clean/reference/quality read. The transfer audit remains exposed-cohort diagnostic only.

## Acceptance / stop criteria

After a valid transfer freeze, compute the unchanged five gates. Report `DYNAMICS_SAFETY_GUARD_TRANSFER_PASS` only if **all five** pass:

- mean PSNR delta vs exact T036 `>= +2.00 dB`;
- median PSNR delta vs exact T036 `> 0`;
- regressions vs exact T026 `<= 29/100`;
- worst paired PSNR delta vs exact T026 `>= -5.614 dB`;
- mean RGB-SSIM delta vs exact T036 `>= -0.001`.

Otherwise report `TRANSFER_NEGATIVE` and close this exact 19-D dynamics guard. No threshold change, feature change, model change, second classifier, or post-outcome rescue in this cycle.

## Explicit non-goals

No new optimizer run; no objective/action-space/budget change; no new Train cohort; no MLP/tree/SVM/RBF/k-NN; no feature/model/hyperparameter sweep; no probability calibration; no transfer labels in training; no per-image clean/reference/test-label use at inference; no oracle-derived step cap/range; no semantic image ID feature; no official LOL-v2 Real test; no LSRW/UHD-LL or other cross-dataset access; no final Ours-vs-baseline claim.

## Expected evidence

Commit the deterministic 8-feature dynamics implementation and tests; hashes binding the exact T065-A 11 features/normalization; development normalization for the 8 new features; 19-D feature tables; leave-one-image-out logistic confusion matrix and unsafe recall; development rollback count/step histogram/five gates; final all-development model/rule manifest if the development stop criterion passes; transfer target-free probability table without reference-derived fields; transfer choice/output freeze preceding any transfer reference-quality read; number of rollbacks and selected-step histogram; post-freeze five-gate table and tail outcomes for indices 16/86; and an independent verifier that recomputes temporal features, normalizations, LOIO fits, final fit, probabilities, rollback decisions, hashes, read ordering, metrics, and classification.

Append exactly one completion report to `coordination/CODEX_TO_CHATGPT.md`, never modify `coordination/PROJECT_STATE.md`, then stop.