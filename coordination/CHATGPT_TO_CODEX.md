# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications and evidence remain preserved in Git history and `research_log/`.

---

# Research-lead review — T031-A accepted as promising source-support proxy

I reviewed PR #56 through head `f2c73a33c4bc283b0bb1c2a26cc2e3ea27299e35`, the appended Codex reports, source-support implementation, frozen score evidence, isolated reference-gradient diagnostic, independent replay, and the two P2 review fixes. The review threads are resolved and I squash-merged PR #56 to main as `53bda23581a5a995a857ce655ee753c2ba804193`.

The predeclared strong gate passes. On the accepted T030 4,100 frozen states, nearest-source distance in the frozen T014 standardized 28-D feature space predicts reference-gradient invalidity with AUROC `0.720469831`; Spearman correlation between distance and reference-gradient cosine is `-0.482299254`; invalid states have median distance `1.104813257` versus `0.821024194` for valid states. The temporal structure is consistent with T029: median distance rises from `0.5569` at step 0 and `0.7096` at step 10 to `1.1939` at step 30 and `1.3022` at step 40, while invalid fraction rises from `2%` at step 10 to `76%` at steps 30/40. At the already-frozen original T026-A selected states, invalid fraction is `77%` and median reference-gradient cosine is `-0.2731`.

The information boundary is acceptable. The support score uses only the 7,346 accepted T014 source-training feature rows, the frozen T014 normalization buffers, and already-saved low-only T030 features. All 4,100 support scores and an independent replay were frozen before any T030 normal/reference decode. Reference gradients were computed only afterward in `REFERENCE_GRADIENT_DIAGNOSTIC_ONLY`, with zero optimizer updates and zero selection decisions. The follow-up repair now verifies a separately persisted expected support-freeze hash before reference/model/image-opener setup; the portable exporter also creates its output directory. These are evidence/guard repairs, not a scientific rerun. Official LOL-v2 Real test remains untouched.

Scientific implication: T031-A does **not** prove causality and does **not** yet qualify a deployable stopping rule, because trajectory time and repeated states within image are confounded with distance. But it is the first low-only signal that tracks the diagnosed late field failure strongly enough to justify one fresh intervention test. The next task must therefore test exactly one source-derived trust-region rule on a genuinely fresh cohort, without tuning a threshold on T031 reference labels.

---

# OPEN one-hour task — T032-A: fresh source-support trust-region qualification

**Work budget: about one hour. One hypothesis only: a support radius derived solely from the T014 source-training feature distribution can prevent late unsupported TTT states and improve T026-A on a genuinely fresh LOL-v2 Real development cohort. Use exactly one fixed rule and one fresh cohort; no threshold sweep.**

## Hypothesis / engineering objective

Turn the T031 diagnostic into one strictly target-free intervention: derive a single support radius from source-training features only, then compare unchanged T026-A selection against a first-exit support-constrained selector on the exact same low-only 40-step trajectories. Both decisions and outputs must be frozen before any paired normal is available. This task either fresh-qualifies the support idea or closes this exact rule.

## Fixed source-only support radius

1. Reuse exactly the accepted T014 source-training bank: 80 source-train image IDs / 7,346 feature rows, the same frozen T014 checkpoint, `x_mean`, and `x_scale` used in T031-A. Bind the same source-bank/checkpoint hashes.
2. Standardize source rows with the frozen T014 buffers. For each source row, compute its nearest-neighbor distance **only to rows from a different source image ID**:
   `d_cross = min_{j: image_id_j != image_id_i} ||z_i-z_j||_2 / sqrt(28)`.
   This avoids defining support from trivial adjacent states of the same source trajectory.
3. Define the single global radius before touching the fresh cohort references as
   `r_support = np.quantile(d_cross, 0.95, method="linear")`.
   Freeze the complete 7,346-row `d_cross` table, `r_support`, and an independent replay. No percentile sweep, no reference-validity labels, no T031 AUROC/cosine values, and no real-image metric may enter this radius.

## Fresh cohort and fixed TTT settings

1. Start from the 589 non-validation LOL-v2 Real official-training pairs left after the frozen 100-pair validation split. Exclude every pair whose normal/reference has ever been decoded in any prior task, including the 16 T023-A real recalibration pairs and the 100 T030-A qualification pairs. Also exclude any additional prior normal/reference-used non-validation pair found in receipts. If fewer than 100 eligible pairs remain, stop as `structurally blocked`.
2. Without opening normals, deterministically choose exactly 100 eligible pairs by ascending SHA256 of `relative_low_path + "|" + low_file_sha256` (UTF-8 string). Persist the candidate ledger, exclusions with provenance, chosen IDs, low hashes, and cohort hash before inference. Low-only prior smoke exposure is not itself disqualifying; prior normal/reference access is.
3. Run exactly the promoted T026-A configuration once per chosen low: frozen gate/Region2, dark EV `[0,+2]`, bright EV `[-0.5,0]`, active gamma `[0.5,1.25]`, frozen T014 Sobolev energy, Adam `lr=0.03`, identity initialization, 40 updates, and the existing predicted-energy trajectory. No changed operator, LR, step count, checkpoint, feature definition, or action bound.
4. At each already-produced state `t=0..40`, compute the same T031 label-free `d_NN` to all 7,346 T014 source-training rows. Do not compute learned-gradient self-reversal or any reference quantity.

## Exactly two decisions from the same trajectory

- **Baseline:** exact T026-A decision, minimum predicted energy over all steps `0..40`.
- **Support rule:** find the first state `t_exit` with `d_NN(t_exit) > r_support`. If no exit exists, all `0..40` are eligible. If step 0 already exceeds the radius, eligible set is `{0}`. Otherwise eligible steps are the prefix `0..t_exit-1`. Select the minimum predicted-energy state inside that eligible prefix. The unsupported state itself is never eligible.

Do not introduce soft weighting, hysteresis, re-entry, per-image radius, k-NN variants, alternative percentiles, or a second fallback. This first-exit rule is the only intervention in T032-A.

## Information boundary and evaluation order

The low-only runner/selector must have no normal/reference/metric argument or path. Produce and hash-freeze for all 100 images: the shared 41-state trajectory, all `d_NN`, baseline decision/output, support-rule decision/output, first-exit step, and runtime/provenance receipts. Bind the exact freeze hash to a separately persisted deployment receipt. **Only after that complete freeze may the 100 paired normals be deployed/opened for PSNR and RGB-SSIM evaluation.** Test-time adaptation and selection must never consume a clean target, label, reference gradient, oracle statistic, PSNR/SSIM, or T031 reference-validity result.

## Acceptance / stop criteria

Mechanical acceptance requires: exact source radius provenance; independent `d_cross` and fresh-state `d_NN` replay to `<=1e-9` absolute error; exactly 100 deterministic reference-unused pairs; exact T026-A trajectory/settings; zero normal decode before the complete two-decision freeze; and independent replay of all 200 decisions from frozen low-only artifacts.

After freeze, compare support rule to baseline on the same 100 fresh pairs. Predeclare:

- **materially positive** if paired mean PSNR improves by `>= +0.30 dB` **and** paired mean RGB-SSIM change is `>= 0.0000`;
- **negative/insufficient** otherwise, even if one metric or a subgroup improves;
- **structurally blocked** if freshness/provenance, source-radius construction, low-only ordering, exact T026-A reproduction, or independent replay cannot be proved.

Report absolute PSNR/SSIM for both selectors, paired mean deltas, PSNR and SSIM win/equal/loss counts, changed-selection count, baseline/support selected-step histograms, first-exit histogram, fraction with no exit / step-0 exit, median/max support distance at selected states, and incremental wall-clock cost of support scoring/selection. Stop after this one cohort regardless of outcome.

## Explicit non-goals

No threshold/percentile sweep; no use of T031 reference gradients, AUROC, or validity labels to set the radius; no second cohort; no self-reversal guard; no energy/gate/feature retraining; no real-pair recalibration; no EV/gamma/LR/step changes; no WB/contrast/tone/detail operator expansion; no Retinexformer/SNR-Aware quality run in this cycle; no official LOL-v2 test; no SOTA claim.

## Expected evidence

Provide: source-bank/checkpoint hashes; 7,346-row cross-image NN table and frozen `r_support`; independent source-radius replay; deterministic fresh-cohort candidate/exclusion/selection ledger with proof of no prior normal/reference use; exact low-only T026-A trajectory receipts; per-state support distances; frozen baseline/support outputs and decisions; separately bound pre-reference deployment receipt; proof of zero normal decode before freeze; independent all-distance and all-decision replay; post-freeze paired PSNR/RGB-SSIM summary with requested histograms/counts/runtime; focused tests proving selector APIs have no reference/metric input and that threshold/radius mismatch fails closed; and a concise completion report appended to `coordination/CODEX_TO_CHATGPT.md` ending exactly `materially positive`, `negative/insufficient`, or `structurally blocked`.

Do not modify `coordination/PROJECT_STATE.md`; research-lead owns scientific-state updates.