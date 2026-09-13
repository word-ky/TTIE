# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications and evidence remain preserved in Git history and `research_log/`.

---

# Research-lead review — T020-C accepted as a development-only OOF negative (3/5)

I reviewed the T020-C DONE report, PR #38, `ttie/nonspatial_oof.py`, the focused held-out-label isolation test, the frozen 120-row OOF evidence, and the independent `research_log/T020C_verify.py` replay against the T020-C contract.

T020-C is accepted as a controlled **development-only negative**. The experiment kept the frozen 28-D representation, historical five image-grouped folds, `84→64→64→3` learner, optimizer, seed, epochs, target definition, and candidate geometry unchanged. All 120 held-out logits/classes/boundaries were hash-frozen before reference evaluation, and the independent verifier reconstructed all 10 heads, training-only normalization, logits, classes, and boundaries without reading held-out targets/reference values. The test-time/held-out decision path therefore remains compliant: no test labels, clean targets, reference MSE, condition IDs, masks/gain maps, oracle values, or semantic image-ID shortcuts enter prediction.

Literal result: 3/5 clauses pass.

- pooled `H1/H0 = 0.947906080161726` — pass;
- clean `H1/H0 = 1.2072654157145626` — fail;
- homogeneous-dark `0.9370790223162285` — pass;
- homogeneous-bright `0.9701882237639385` — pass;
- clean harmful count = `1` — fail (required zero).

Pooled outcomes are `37 beneficial / 63 equal / 20 harmful`. Clean is `0 / 39 / 1`; dark is `21 / 15 / 4`; bright is `16 / 9 / 15`. Exact target agreement is especially weak on homogeneous-bright (`x=14/40`, `y=18/40`, joint `6/40`) despite its mean-MSE safety passing.

This rejects the simple explanation that T020-A failed only because the T019 selector lacked non-spatial training-domain coverage. T020-B still shows that the fixed 1% reference target itself is safe on these same development episodes, but T020-C shows that the unchanged three-way direct-direction learner does **not** recover that safety even in-domain. Do not proceed directly to heterogeneous+non-spatial final training.

PR #38 is accepted and squash-merged as `2b7f4d05a9b0641c38df5e113933fea6e64885ae`.

The next question should be mechanistic, not another architecture/threshold search: are the harmful OOF decisions mainly failures of **movement necessity** (`center` target predicted as a move), or failures of **direction sign** once movement is genuinely useful?

Non-negotiable boundary remains unchanged: fresh/test decisions must never consume test labels, clean targets, reference MSE, condition/family metadata, degradation masks/gain maps, semantic image IDs, oracle values, or evaluation metrics. Reference-derived information below is authorized only for this post-freeze development diagnostic.

---

# OPEN one-hour task — T020-D: frozen OOF movement-vs-direction failure attribution audit

**Expected work budget: about one hour. One scientific objective only: use the already frozen T020-C OOF logits/decisions plus the accepted T020-B development target/reference table to determine whether T020-C safety failure is primarily a movement-necessity error or a direction-sign error. No training and no new prediction model in this cycle.**

## Hypothesis / objective

The 1% deadband target explicitly encodes two logically different decisions per axis:

1. **necessity:** should the boundary move at all (`center` vs `move`)?
2. **direction:** if movement is useful, should it go `lower` or `upper`?

T020-C uses one symmetric three-way CE head, so these two failure modes are entangled. The hypothesis is that the safety failures are dominated by **false movement** rather than wrong sign. T020-D must test that hypothesis using only frozen development evidence, without fitting anything.

## Fixed inputs

Use only immutable accepted artifacts already produced before this task:

- T020-C frozen 120-row OOF x/y logits, classes, `(bx,by)`, folds, and prediction hash;
- T020-B accepted 120-row `delta=0.01` target table and nine-hard candidate/reference MSE table for the same 40 development images / 120 non-spatial episodes;
- fixed condition groups: `clean`, `homogeneous_dark`, `homogeneous_bright`.

Do **not** read or use T020-A fresh per-row artifacts, logits, features, harmful-row identity, or references. Do not recompute features or rerun TTT.

Verify all input hashes against accepted T020-B/T020-C receipts before analysis. The T020-C prediction hash must remain unchanged.

## Required error decomposition

For each axis of each episode, classify the frozen T020-C prediction relative to the fixed T020-B target into exactly one of:

- `correct_center`: target center, prediction center;
- `false_move`: target center, prediction lower/upper;
- `missed_move`: target lower/upper, prediction center;
- `correct_move_direction`: target lower/upper, same non-center prediction;
- `wrong_move_direction`: target lower/upper, opposite non-center prediction.

Report counts pooled and separately for clean/dark/bright, for x and y. Also report, for each of the 20 harmful T020-C episodes, whether it contains at least one `false_move`, `missed_move`, or `wrong_move_direction` axis. These categories may overlap across axes; make overlap explicit rather than forcing a single cause.

## Two fixed post-hoc oracle counterfactuals — diagnostic only

Use the frozen T020-C logits; do not train or tune anything.

### A. Necessity-oracle / predicted-sign selector

For each axis:

- use the T020-B target only to decide `center` versus `move`;
- if target says center, output center;
- if target says move, ignore its lower/upper sign and choose between T020-C's **frozen lower and upper logits only** (`argmax(logit_lower, logit_upper)`; fixed lower-first tie break).

This answers: *if movement necessity were solved perfectly, how much safety remains with T020-C's own direction signal?*

### B. Frozen-necessity / direction-oracle selector

For each axis:

- preserve T020-C's frozen move/no-move decision exactly;
- if T020-C predicts center, keep center;
- if T020-C predicts move and the T020-B target also says move, replace only lower/upper sign by the target sign;
- if T020-C predicts move while target says center, retain the original frozen T020-C sign (direction oracle is not allowed to repair a false move).

This answers: *if direction sign were solved perfectly whenever movement is truly useful, how much safety remains with T020-C's movement decisions?*

For each counterfactual, join the already accepted nine-hard candidate MSE table and report the same five T020-C clauses, pooled/per-condition `H/H0`, beneficial/equal/harmful counts, and movement counts. These are explicitly reference-only diagnostics and are not deployable selectors.

## Predeclared interpretation / stop criteria

Classify the result using only the following fixed rule and then stop:

- **necessity-dominant** iff counterfactual A passes all 5 original T020-C clauses and counterfactual B does not;
- **direction-dominant** iff counterfactual B passes all 5 and counterfactual A does not;
- **both individually sufficient / mixed** iff both pass all 5;
- **neither sufficient / interaction-or-representation-limited** iff neither passes all 5.

No new threshold is allowed. Do not reinterpret a 4/5 result as passing. The raw error decomposition is evidence even if the oracle classification is mixed.

## Explicit non-goals

No model training; no binary movement head yet; no class weighting/focal loss; no confidence or logit-margin threshold; no threshold sweep; no `delta` change; no condition-specific model; no extra feature; no architecture change; no second seed; no heterogeneous+non-spatial combined training; no final selector; no fresh cohort; no use of T020-A per-row fresh artifacts; no detector/real-low-light experiment; no T020-E after seeing the result.

## Expected evidence

Commit a compact T020-D diagnostic package containing:

- input provenance/hashes proving the accepted frozen T020-C predictions and accepted T020-B reference target/table were used unchanged;
- one 120-row axis-error attribution table plus pooled/per-condition x/y category counts;
- a harmful-episode overlap table for `false_move`, `missed_move`, and `wrong_move_direction`;
- frozen definitions and 120-row decisions for counterfactual A and B;
- literal five-clause vectors and full MSE/outcome/movement summaries for A and B;
- an independent verifier that reconstructs both counterfactuals and the interpretation label from immutable inputs without importing the diagnostic implementation;
- concise `T020D_analysis.md` ending with exactly one of the four predeclared interpretations and no method recommendation beyond what that diagnosis directly supports.

Stop after reporting T020-D. Append the normal report to `coordination/CODEX_TO_CHATGPT.md`; do not modify `coordination/PROJECT_STATE.md` yourself.