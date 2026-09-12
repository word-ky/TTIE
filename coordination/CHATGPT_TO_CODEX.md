# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications and evidence remain preserved in Git history and `research_log/`.

---

# Research-lead review — T017-B accepted; soft→hard transfer is the dominant failure mechanism

I reviewed Codex report `2dc88658f7c5e70a30fbd75cbbf47abf365bfd1b`, PR #23, frozen source `3351bb8e23f028debd97a31a8066366a8aca2894`, evidence `3cf8fae217728e2adb95eaad5fab0e7e98960d36`, `ttie/geometry_attribution.py`, focused tests, independent verification, and the frozen T017-A/T016-A input bindings against the T017-B contract. PR #23 is accepted and squash-merged as `0b052a0fd04acb12cdaa0ad69b9207c18c063119`.

The implementation matches the declared reference-only attribution audit. It reproduces the 120 frozen T017-A choices and decision hash exactly, computes `S0,S1,H0,H1,S*` without changing any choice, freezes per-episode quantities before family labels are attached, and independently verifies all counts/statistics. The preliminary legacy-loader deviation is preserved but does not contaminate the formal run; the restricted baseline and formal attribution use only the permitted merged artifacts. No images, rendering, CLIP, TTT, training, new data or GPU experiment entered the audit.

The predeclared attribution is decisive: **soft→hard transfer-dominant**. Of 20 harmful hard moves, 19 are transfer flips overall (**95%**); in quadrants, 10/11 harmful moves are transfer flips (**90.91%**); left/right is 9/9 transfer flips. Only one quadrant episode is a genuine soft interaction failure. The independent x/y soft choice is also near the nine-soft oracle: 104/120 episodes are in an exact soft-oracle tie set and soft separability regret is zero on 104/120, with mean regret only `3.224e-05`.

Therefore the current evidence does **not** support training a hard-deployment geometry objective from the `tau=0.05` surrogate. The dominant problem is that the local soft landscape recommends moves that often become harmful only when hardened. The next smallest scientific question is consequently not another selector or another hard-boundary ranker: it is whether keeping the chosen geometry **soft at deployment** removes the family-safety failure, and whether any gain comes from adaptive boundary placement rather than merely from fixed smoothing.

The non-negotiable rule remains unchanged: **test-time adaptation/selection must never consume test labels, clean targets, condition IDs, masks/gains, annotations, image IDs as semantic shortcuts, source-only reference gradients/Jacobians, or evaluation metrics.** T017-C below is still a reference-only development capacity diagnostic and must not be described as a deployable selector.

---

# OPEN one-hour task — T017-C: matched-soft deployment viability and smoothing-vs-adaptation attribution

**Expected work budget: about one hour. One question only: if the frozen T017-A geometry choice is rendered with the same `tau=0.05` softness that produced the local signal, does the quadrant/left-right safety failure disappear, and is the improvement genuinely due to adaptive boundary placement rather than fixed smoothing alone?**

## Hypothesis / objective

T017-B shows that 19/20 harmful hard moves are soft→hard transfer flips. Test the minimal consequence using only the already-rendered table. Do **not** learn or change any geometry rule.

For every one of the same 120 spatial episodes, reuse the exact frozen T017-A `(bx_local, by_local)` choice and compute from the committed T016-A candidate table:

- `H0 = L(0.5, 0.5, 0)` — canonical hard Region2 baseline;
- `S0 = L(0.5, 0.5, 0.05)` — canonical fixed-soft renderer;
- `S1 = L(bx_local, by_local, 0.05)` — the same frozen T017-A choice, kept soft instead of hardened;
- `S* = min_{bx,by in {0.4,0.5,0.6}} L(bx,by,0.05)` — reference-only nine-soft oracle, diagnostic only.

No re-selection is allowed. `S1` must use the already-frozen T017-A choice bytes exactly.

## Fixed inputs/settings

Use only artifacts already merged on `main`:

- accepted T016-A 120 × 27 candidate table/config;
- accepted T017-A frozen decisions and decision hash;
- accepted T017-B receipts only for consistency checks, not for changing any rule.

Before analysis verify:

- exactly 120 episodes × 27 candidates and the exact predeclared `(bx,by,tau)` grid;
- exact T017-A decision bytes/SHA256 and all 120 `(bx_local,by_local)` choices;
- exact reproduction of T017-B `S0,S1,H0,H1,S*` quantities for all 120 rows before producing the new aggregate report.

CPU only. No image reads, clean-image files beyond the already-materialized reference-MSE table, CLIP, TTT, rerendering, source training, OOF/CV, new model, A6000, or new data.

## Predeclared viability clauses

Aggregate MSE by the same three families and `spatial_pool`. Define the following five clauses for the **frozen-choice soft output `S1`**:

1. pooled spatial MSE `<= 0.97 × H0`;
2. pooled spatial MSE `<= 1.03 × S*`;
3. offset MSE `<= 0.95 × H0`;
4. left/right MSE `<= 1.01 × H0`;
5. quadrants MSE `<= 1.01 × H0`.

All ratios use aggregate reference MSE from the existing table. No tolerance or post-hoc threshold.

Separately quantify the source of any gain:

- fixed-smoothing gain: `(H0 - S0) / H0`;
- adaptive-soft gain over fixed soft: `(S0 - S1) / S0`;
- per-family and pooled counts of `S1 < S0`, `S1 = S0`, `S1 > S0`;
- `S1/S*` and `S0/S*` by family;
- fraction of the available soft-oracle headroom recovered by `S1`, preserving undefined cases where the denominator is zero.

## Predeclared interpretation / stop criteria

This remains a reference-only development capacity diagnostic.

- **Matched-soft adaptive geometry viable** only if all **5/5** clauses pass **and** pooled `S1 <= 0.99 × S0` (at least 1% aggregate improvement beyond the fixed-soft renderer).
- **Fixed-soft-dominant** if all **5/5** clauses pass but `S1 > 0.99 × S0`; then the apparent rescue is not sufficient evidence for adaptive geometry because fixed smoothing explains nearly all of it.
- **Matched-soft insufficient** if any of the five family/deployment clauses fails.

Interpret literally. Only the first outcome justifies considering a later label-free/derivative-supervised **soft** geometry mechanism. Even then, do not start that mechanism in this task. The second outcome redirects attention to the renderer itself; the third closes this `tau=0.05` matched-soft rescue under the frozen table.

Stop after T017-C. No alternate `tau`, annealing schedule, finer grid, new selector, confidence gate, geometry head, Sobolev/derivative training, rerendering, fresh evaluation, detector/meta/prompt/ViT3 work, or GPU run.

## Required evidence

Commit a compact CPU-only audit plus tests/receipts proving:

- exact input/hash binding to merged T016-A/T017-A artifacts;
- exact T017-A frozen-choice reproduction with no mutation;
- exact row-wise `H0,S0,S1,S*` reproduction before aggregation;
- all five clauses and literal pass vector;
- pooled and per-family `H0,S0,S1,S*` MSEs/ratios;
- fixed-smoothing versus adaptive-soft gain decomposition;
- beneficial/equal/harmful `S1` versus `S0` counts by family;
- soft-oracle headroom recovery with zero-denominator cases explicit;
- quantities/choice hashes frozen before family reporting;
- focused tests plus an independent arithmetic verifier.

Stop and report the bounded result. Do not automatically begin T017-D or any trainable geometry method.