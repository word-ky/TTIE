# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications and evidence remain preserved in Git history and `research_log/`.

---

# Research-lead review — T017-C accepted; matched-soft rescue is closed

I reviewed Codex report `df1ec2744b335ad30fb898b9f408c626038d189b`, PR #24, frozen source `1512d03347dbc83e85837e835067dfd8255066b9`, evidence `d4eeede5e2368b9bf9a0366ea734ef25776994be`, `ttie/matched_soft.py`, focused tests, independent verification, and the immutable T016-A/T017-A/T017-B input bindings against the T017-C contract. PR #24 is accepted and squash-merged as `de73871600a8089e176ec21f06fb2d54b4486faf`.

The implementation matches the declared CPU-only reference diagnostic. It reproduces all 120 frozen T017-A choices and all T017-B per-episode quantities exactly before aggregation, keeps the original decision bytes unchanged, binds the eight immutable JSON inputs and six source files, freezes quantities before family labels, and independently verifies the arithmetic. No images, rendering, CLIP, TTT, training, new data, fresh evaluation or GPU work entered T017-C.

The scientific result is a clear bounded negative: **matched-soft is insufficient, 2/5**. Keeping the same frozen geometry choices at `tau=0.05` is nearly optimal *within that soft renderer* (`S1/S* = 1.00088`) and adaptive movement improves pooled soft MSE by 4.02% over fixed soft, but the renderer itself is poor relative to canonical hard Region2. Fixed `tau=0.05` worsens pooled MSE by 8.91%, left/right by 9.22%, and quadrants by 23.03%. After adaptive movement, `S1/H0` is still `1.04531` pooled, `1.05573` left/right, and `1.22136` quadrants; only offset benefits (`0.90292`). Thus the T017-B soft→hard mismatch was real, but simply deploying the soft surrogate is not a solution. The `tau=0.05` matched-soft path is closed; do not retune softness or train a soft-geometry model from it.

The next smallest unresolved question is now on the **hard renderer itself**. T016-A already established that nine shifted hard boundaries contain almost all useful geometry capacity. Before learning any geometry mechanism, determine whether a local, factorized hard-boundary descent target exists around canonical Region2 when the reference diagnostic uses the same hard renderer on both sides. This removes the soft-renderer confound entirely and tests whether future geometry supervision can be defined as hard finite-step direction rather than a soft surrogate derivative.

The non-negotiable rule remains unchanged: **test-time adaptation/selection must never consume test labels, clean targets, condition IDs, masks/gains, annotations, image IDs as semantic shortcuts, source-only reference gradients/Jacobians, or evaluation metrics.** T018-A below is reference-only development diagnosis; it is not a deployable selector.

---

# OPEN one-hour task — T018-A: hard-renderer local-direction viability audit

**Expected work budget: about one hour. One question only: does the existing nine-hard reference landscape admit a safe local factorized `(bx,by)` descent target around canonical Region2, without any soft renderer?**

## Hypothesis / objective

Use only the already-rendered T016-A `tau=0` candidates. For each of the same 120 spatial episodes define:

- `H0 = L(0.5, 0.5, 0)`;
- x-axis local cross: `Hx- = L(0.4,0.5,0)`, `Hx0 = H0`, `Hx+ = L(0.6,0.5,0)`;
- y-axis local cross: `Hy- = L(0.5,0.4,0)`, `Hy0 = H0`, `Hy+ = L(0.5,0.6,0)`.

Choose `bx_local_hard` as the minimum-MSE member of `{0.5, 0.4, 0.6}` on the x-axis cross and `by_local_hard` independently on the y-axis cross. **Tie order is literal: center `0.5` first, then lower `0.4`, then upper `0.6`**, so exact ties prefer no movement. Then evaluate only the already-existing combined hard candidate

`H1 = L(bx_local_hard, by_local_hard, 0)`.

Also compute the reference-only nine-hard oracle

`H* = min_{bx,by in {0.4,0.5,0.6}} L(bx,by,0)`

with the same deterministic candidate order used in T016-A. No candidate is rendered or optimized in this task.

This is deliberately a **reference-only target-viability audit**. `bx_local_hard/by_local_hard` are not available at deployment. The purpose is to test whether the hard loss landscape itself provides a simple local directional target worth learning later.

## Fixed inputs / settings

Use only merged artifacts already on `main`:

- accepted T016-A 120 × 27 candidate-MSE table and config;
- accepted T017-A/T017-B/T017-C artifacts only for hash/consistency checks and comparative reporting, never to change the T018-A rule.

Before analysis verify exactly 120 episodes × 27 candidates and the exact predeclared `(bx,by,tau)` grid. Verify the five hard-cross entries and all nine hard entries are present once per episode. CPU only.

No image reads, clean-image files beyond the already-materialized reference-MSE table, CLIP, TTT, rerendering, source training, OOF/CV, learned model, threshold, abstention, confidence rule, alternate grid, alternate tie rule, A6000, or new data. Do not use condition or image ID in any choice.

## Predeclared viability clauses

Aggregate by the same three families and `spatial_pool`. `H1` qualifies as a viable **reference local hard-direction target** only if all five hold:

1. pooled `H1 <= 0.97 × H0`;
2. pooled `H1 <= 1.03 × H*`;
3. offset `H1 <= 0.95 × H0`;
4. left/right `H1 <= 1.01 × H0`;
5. quadrants `H1 <= 1.01 × H0`.

No tolerance and no post-hoc fallback.

Also report, without changing the decision rule:

- pooled/per-family `H0`, `H1`, `H*`, ratios and oracle-headroom recovery;
- counts of no-move / x-only / y-only / both-axis moves;
- counts of `H1 < H0`, `H1 = H0`, `H1 > H0` by family;
- exact match rate of `(bx_local_hard,by_local_hard)` to the nine-hard oracle tie set;
- factorization regret `H1-H*` distribution;
- for every harmful combined move (`H1>H0`), whether each chosen axis move was individually non-worse than center. Such a case is a pure x/y interaction failure; preserve exact counts and examples by row index only, attaching family labels only after quantities are frozen.

## Predeclared interpretation / stop criteria

- **Hard local-direction target viable** only if all **5/5** clauses pass. This would justify a later task asking whether a label-free/source-supervised mechanism can predict the *hard-renderer local direction*; it would **not** itself justify deployment or fresh evaluation.
- **Hard local factorization insufficient** if any clause fails. Preserve the negative and stop; do not repair it with a 2-D oracle, confidence gate, condition-specific rule, different step size, or learned model in this task.

If harmful combined moves occur even though both selected axis moves are individually non-worse, report them as interaction evidence; do not silently redefine the selector.

Stop after T018-A. No geometry predictor/head, Sobolev training, finite-difference network, soft renderer, alternate `tau`, finer boundary grid, fresh split, detector/meta/prompt/ViT3 work, or GPU run.

## Required evidence

Commit a compact CPU-only audit plus tests/receipts proving:

- exact input/hash binding to the merged T016-A table/config;
- literal center-first/lower/upper tie order on both axes;
- choices depend only on the five reference hard-cross values in this diagnostic, never on family metadata;
- all per-episode choices/`H0,H1,H*` quantities are persisted and hashed before family labels/reporting;
- all five clauses and exact pass vector;
- pooled/per-family MSEs, ratios, movement counts, oracle tie-set match, factorization regret, and harmful interaction attribution;
- focused boundary/tie/freeze tests plus an independent arithmetic verifier.

Stop and report the bounded result. Do not automatically begin T018-B or any trainable geometry method.