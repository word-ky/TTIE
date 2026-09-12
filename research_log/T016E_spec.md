# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications and evidence remain preserved in Git history and `research_log/`.

---

# Research-lead review — T016-D accepted as a controlled development negative

T016-D is accepted scientifically as a **1/5 negative for both fixed pairwise-ranking probes**. I reviewed the report/evidence at main report commit `d4fda9f8794ad6909e1dbc409b5bb57fa537741d`, frozen scientific source `c91495225b6df73c814ee6f48b7bd3ab8ff2b6c7`, final evidence `aa71d268294e35f5df67c76eada29f9bec117abe`, and PR #19 against the T016-D contract.

The implementation matches the prescribed diagnostic. Pair generation is strictly within episode, uses every non-tied `i<j` pair, and trains the unchanged `D→64→64→1` SiLU scalar head with the fixed unweighted logistic objective. Five folds remain image-grouped; train-only normalization is preserved; all ten heads and both OOF score tables are frozen/hash-bound before held-out reference evaluation. No new images, rendering, CLIP, TTT, GPU work, condition/image identifiers, clean pixels, held-out MSE, oracle rank, masks/gains, or evaluation metrics enter held-out scoring. This is development-only supervision. The non-negotiable rule remains: **test-time adaptation/selection must never use test labels or clean targets.**

The result is informative rather than merely negative. `rank28` remains unsafe (`0.03540186 = 1.01022×` Region2). `rank30` improves materially over pointwise T016-C and over `rank28`: spatial MSE `0.03421816 = 0.97645×` Region2 and `1.05081×` the nine-hard oracle; median pooled Spearman reaches `0.75`; offset improves strongly to `0.88846×` Region2. However it still misses the predeclared 3% spatial gain and 5% oracle-regret bounds, and—more importantly—harms left/right (`1.04687×`) and quadrants (`1.01626×`) beyond the 1% safety limits. Both literal clause vectors are therefore `[false,false,true,false,false]`.

The mechanistic interpretation should remain narrow. Pairwise supervision **does** recover substantially better ordering signal, especially with explicit boundary coordinates, so the evidence no longer supports a simple “no ranking signal in 30-D” story. But forced argmin over nine candidates is still unsafe. This matters because T016-A showed many episodes with little/no benefit from moving the canonical boundary, and quadrants are overwhelmingly canonical-oracle cases. The smallest unresolved confound is therefore **decision confidence / abstention**, not another loss sweep or a larger feature model.

PR #19 is accepted for scientific evidence only; do not spend this cycle repairing PR topology or merging it. Consume its immutable compact artifacts by commit/hash.

---

# OPEN one-hour task — T016-E: nested-OOF confidence-abstention audit for `rank30`

**Expected work budget: about one hour. One question only: is T016-D mainly failing because the improved rank30 scorer is forced to adapt even when its preference over canonical Region2 is weak?**

## Hypothesis

A safe boundary selector may need an explicit canonical fallback. Test the minimal mechanism without retraining any network:

> If a single training-only confidence threshold on the frozen `rank30` score margin can preserve canonical Region2 on uncertain episodes while retaining the strong offset gains, then T016-D diagnosed a decision-rule problem rather than a need for richer representation. If it still fails the same five clauses, simple confidence abstention is insufficient.

This is a **development-only OOF decision-rule diagnostic**, not a deployable/fresh result.

## Fixed inputs

Use only the immutable T016-D `rank30` OOF score table, the unchanged T016-C folds, and the accepted T016-B/T016-A reference joins already used by T016-D. No retraining, no new head, no new image, rendering, CLIP, TTT, A6000, reference-MSE recomputation, feature expansion, or candidate expansion.

Keep the same 40 development IDs, 120 episodes, 9 hard candidates and five outer folds. Canonical Region2 is candidate `(0.5,0.5,0)` / index 4.

## Label-free confidence

For each frozen OOF score vector `s[0..8]`, let `j*` be the lowest-score **noncanonical** candidate with existing lexicographic tie order. Define

`q = (s[4] - s[j*]) / max(std_pop(s), 1e-12)`.

If all scores are constant, define `q=0`. Higher `q` means stronger label-free evidence that a noncanonical boundary beats canonical. Do not use clean/reference information in `q`.

Use exactly this fixed threshold grid and no other values:

`T = {0.00, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, +inf}`.

For threshold `t`, choose `j*` only when `q > t`; otherwise choose canonical index 4. Exact `q==t` falls back to canonical.

## Nested OOF threshold calibration

For each outer fold `k`:

1. The 8 outer-held-out IDs are completely unavailable to threshold calibration.
2. Calibrate `t_k` using only the other 32 IDs **and their already-frozen T016-D OOF score vectors** (i.e. scores produced when each calibration image itself was held out from its ranker). Do not score those 32 IDs with the outer-fold training model.
3. For each of the eight fixed thresholds, compute mean selected reference MSE over all 96 calibration episodes (32 IDs × 3 conditions). Do not use condition labels or condition-specific weights/thresholds.
4. Choose the threshold with lowest calibration mean MSE; exact ties choose the **larger / safer** threshold.
5. Apply only that frozen `t_k` to the 24 outer-held-out episodes using their already-frozen rank30 scores. Persist all five thresholds, 120 decisions, `q` values and hashes **before** held-out reference evaluation.

This is the only permitted calibration. No alternate grid, per-condition threshold, post-hoc threshold, coverage target, temperature, score rescaling, or second gate.

## Acceptance and diagnostics

Evaluate the combined 120-episode OOF gated selector with the **same five clauses** as T016-C/D:

1. spatial selected MSE `<= 0.97 ×` canonical Region2;
2. spatial selected MSE `<= 1.05 ×` nine-hard oracle;
3. offset selected MSE `<= 0.95 ×` canonical Region2;
4. left/right selected MSE `<= 1.01 ×` canonical Region2;
5. quadrants selected MSE `<= 1.01 ×` canonical Region2.

Report spatial/LR/quadrants/offset MSE and ratios versus Region2, hard oracle, ungated rank30 and T016-C probe30; the five calibrated thresholds; adaptive-vs-canonical counts by fold and condition; mean/median/quantiles of `q`; beneficial/harmful/zero-gain counts among adapted episodes after reference attachment; and oracle disagreement/outside-tie rate.

Interpret literally:

- **5/5 pass:** conclude only that the existing rank30 representation/ranker becomes development-safe with a simple training-only confidence fallback; forced adaptation was the diagnosed bottleneck. Do not launch fresh evaluation yet.
- **Any failure:** preserve the negative. Conclude that simple scalar confidence abstention does not safely unlock the T016-A boundary headroom; do not tune another threshold or add a second gate.

## Non-goals / stop conditions

Do not retrain rank28/rank30; do not add features, image/spatial tokens, condition cues, candidate IDs, larger models, pair weighting, margin losses, learned gates, continuous boundaries, fresh data, detector/meta/prompt/ViT3 work, or another confidence definition. Held-out reference values must not be opened until the five thresholds and all 120 held-out decisions are frozen/hash-bound.

Commit one compact CPU-only implementation and focused tests proving: exact fixed `q`; constant-score handling; exact threshold grid/tie-to-safer rule; outer-held-out IDs absent from calibration; calibration uses only previously OOF scores for the 32 calibration IDs; mutation of held-out references cannot alter thresholds/decisions; decisions freeze before evaluation. Persist calibration tables, thresholds, decisions, source hashes and a concise `T016E_analysis.md` with the literal five clauses.

Stop after T016-E and report. Do not automatically start any follow-on model or experiment.
