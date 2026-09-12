# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications and evidence remain preserved in Git history and `research_log/`.

---

# Research-lead review — T016-F accepted as a controlled fully nested negative

I reviewed report commit `c2cf30e0063e88daea514dfc697794083555b466`, PR #21, frozen scientific source `fb9d33e9c10e50cd88b389411adf56601ea3cc68`, evidence `45bc8f6dfa536a5adc351929b335f750ab82487e`, and the nested implementation against the T016-F contract. PR #21 is accepted and squash-merged as `92d0667eb563f8c7ddbfe24f0a70c6c5eda3ce94`.

The nesting repair is methodologically sound. Each outer fold trains one rank30 head on only its 32 outer-train IDs and four fresh inner heads on deterministic 24/8 splits inside those 32 IDs. Threshold calibration uses only the 32 inner-OOF predictions and references; the eight outer-held-out IDs are excluded from all five heads and calibration quantities that determine their decisions. The fold decisions and hashes are persisted before held-out reference evaluation. No historical T016-D head/score is reused in the primary pipeline, and mutating outer-held-out references leaves the five heads, inner-OOF scores, threshold and decisions unchanged.

The literal result is **4/5**, so the T016-E 5/5 does not survive proper nesting. Spatial pooled MSE is `0.03396188 = 0.96913×` canonical Region2 and `1.04294×` the nine-hard oracle; offset remains strong at `0.90521×` Region2 and quadrants are exactly `1.00000×`. The sole failure is left/right at `1.01850×` Region2, above the predeclared `1.01` safety limit. All six harmful adaptive episodes occur in left/right. This is not a near-positive to be retuned: the fixed scalar `rank30` + single confidence-threshold cycle is closed as prescribed. Do not try another threshold grid, confidence statistic, fold split, loss, or slightly larger scalar head on these 40 inspected IDs.

The scientific implication is narrower and more useful: T016-A still establishes real boundary-placement capacity, while T016-B–F show that treating geometry primarily as a discrete candidate-ranking/abstention problem does not provide robust development-safe selection. Before building a more expressive geometry network, test whether the **reference geometry landscape itself is locally reachable from the canonical boundary through a smooth/derivative-like signal**. That determines whether the T014 derivative-supervision principle can plausibly be extended from ISP actions to spatial geometry.

The non-negotiable deployment rule remains unchanged: **test-time adaptation/selection must never consume test labels, clean targets, condition IDs, masks/gains, annotations, image IDs as semantic shortcuts, source-only reference gradients/Jacobians, or evaluation metrics.** The next task is explicitly reference-only development diagnosis; its decisions are not deployable and must never be described as label-free test-time selection.

---

# OPEN one-hour task — T017-A: reference-only local geometry-landscape viability audit

**Expected work budget: about one hour. One question only: is the accepted nine-hard boundary headroom locally recoverable from the canonical boundary using the already-rendered soft `tau=0.05` neighborhood, strongly enough to justify a future derivative-supervised fast geometry variable?**

## Hypothesis

T016-A shows `7.24%` pooled oracle gain from boundary placement, and the nine-hard subset captures essentially all of that capacity. T016-B–F show that scalar candidate ranking is unsafe. A different mechanism is justified only if the geometry landscape has useful local structure:

> If a deterministic reference-only local rule derived from the `tau=0.05` cross around canonical recovers most of the nine-hard oracle while preserving family safety, then a continuous/derivative-supervised geometry variable is scientifically plausible. If it does not, do not jump directly to a learned geometry optimizer; the current coarse local landscape is not sufficient evidence.

This task is **development-only and reference-only**. It is an optimization-landscape diagnostic, not a deployable selector.

## Fixed inputs

Use only the already-committed T016-A 120 × 27 candidate-MSE table from evidence commit `ee5d8fdaf3ab48ee7ad3654d45bdc65419be8367` (`candidate_metrics.json` and its accepted hashes/metadata). Do not read image pixels, clean image files, T015 large packs, CLIP features, TTT trajectories, or rerender anything.

The candidate set must be verified as exactly

`bx, by ∈ {0.4, 0.5, 0.6}`, `tau ∈ {0, 0.05, 0.10}`,

with canonical hard Region2 `(0.5,0.5,0)` exactly matching the accepted Region2 MSE for all 120 episodes. Recompute the nine-hard oracle from the `tau=0` rows and verify it matches the accepted T016-B/T016-A hard-oracle aggregate before any new interpretation.

## Fixed local rule

For each episode define `L(bx,by,tau)` from the existing table. Use **only the five `tau=0.05` values on the canonical cross**:

- `L(0.5,0.5,0.05)`;
- `L(0.4,0.5,0.05)`, `L(0.6,0.5,0.05)`;
- `L(0.5,0.4,0.05)`, `L(0.5,0.6,0.05)`.

Choose the x-coordinate independently from `{0.4,0.5,0.6}` by the minimum of the three x-axis soft values with `by=0.5`. Choose the y-coordinate independently analogously with `bx=0.5`. Exact ties must prefer `0.5`; remaining equal-distance ties prefer the lower coordinate. Call the resulting pair `(bx_local, by_local)`.

The primary diagnostic output is the **already-rendered hard candidate** `L(bx_local,by_local,0)`. There is no interpolation, rerendering, optimizer, learned model, threshold or condition-specific rule.

Also report the central finite differences, for diagnosis only:

`gx = [L(0.6,0.5,0.05) - L(0.4,0.5,0.05)] / 0.2`

`gy = [L(0.5,0.6,0.05) - L(0.5,0.4,0.05)] / 0.2`.

Do not use condition labels or image IDs in the local rule. They may be attached only after all 120 local choices are frozen for reporting by family.

## Predeclared acceptance / stopping rule

After all 120 local choices are persisted, compare the selected hard MSE with canonical Region2 and the nine-hard reference oracle. `T017-A` is a **positive local-geometry viability diagnostic only if all five clauses pass**:

1. spatial pooled MSE `<= 0.97 ×` canonical Region2;
2. spatial pooled MSE `<= 1.03 ×` nine-hard oracle;
3. offset MSE `<= 0.95 ×` canonical Region2;
4. left/right MSE `<= 1.01 ×` canonical Region2;
5. quadrants MSE `<= 1.01 ×` canonical Region2.

Interpret literally:

- **5/5:** conclude only that the existing boundary headroom is locally accessible through a smooth reference-neighborhood signal, which justifies considering a later source-trained derivative-supervised geometry objective. Do not train that objective in this cycle.
- **Any failure:** preserve the negative/inconclusive result. Conclude only that this coarse `tau=0.05` local-axis diagnostic does not justify continuous geometry optimization yet; do not infer that all learned spatial bases are impossible.

No post-result rule changes are allowed.

## Required evidence

Commit a compact CPU-only analysis script, focused tests, and receipts proving:

- exact input artifact/hash binding to the accepted T016-A table;
- exactly 120 episodes × 27 candidates and the exact candidate tuples;
- canonical hard MSE equality and nine-hard oracle reproduction before the new diagnostic;
- the local rule reads only the five fixed `tau=0.05` cross values per episode;
- exact tie behavior (`0.5` first, then lower coordinate if still tied);
- no condition/image-ID input to the local rule;
- all 120 `(bx_local,by_local)` decisions are persisted/hashed before family labels are used for aggregate reporting;
- independent recomputation of all five clauses from the frozen decisions and existing MSE table.

Report pooled/LR/quadrants/offset MSE and ratios, nine-hard oracle gap, fraction of oracle gain captured `(Region2-local)/(Region2-oracle)` with safe handling when denominator is zero, local boundary counts, hard-oracle boundary counts, local-vs-oracle disagreement/tie-set rate, `gx/gy` sign/quantile summaries, and per-condition gain distributions. Preserve zero-gain and tied-oracle cases explicitly.

## Non-goals

No model training, OOF/CV, threshold calibration, new confidence rule, alternate `tau`, wider/finer boundary grid, continuous optimizer, rerendering, image/CLIP reads, TTT, A6000 run, new data, fresh split, learned basis, spatial token network, detector/meta/prompt/ViT3 work, or source-side geometry head yet.

Stop after T017-A and report. Do not automatically start derivative-supervised geometry training or any fresh evaluation.