# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications and evidence remain preserved in Git history and `research_log/`.

---

# Research-lead review — T019-A accepted as a development-only utility-target positive

I reviewed the T019-A DONE report, PR #32, `ttie/utility_deadband.py`, focused threshold/tie/freeze tests, the independent `T019A_verify.py`, and the frozen target/evaluation receipts against the T019-A contract.

The scientific result is accepted. With the single predeclared `delta = 0.01` per-axis relative-MSE deadband, the 120 development targets achieve pooled `H_delta/H0 = 0.930928`, pooled `H_delta/H* = 1.001826`, offset `0.850867`, left/right `0.964534`, and quadrants `1.000000`. All five established clauses pass, and the additional target-safety condition also passes: `56 beneficial / 64 equal / 0 harmful`. The deadband suppresses 26 original non-center axis labels while retaining `97.6018%` of pooled nine-hard oracle headroom. In particular, all 40 development quadrants remain canonical center.

The audit respects its information boundary. It is reference-only development analysis, not a deployable selector. It reads only the accepted T016-A/T018-A development candidate table; T018-E fresh reference MSE, logits/features, family labels and outcomes are not inputs. The target choices are frozen before family aggregation. No new image rendering, CLIP/TTT run, model training, GPU work, threshold search, family rule, or confidence gate occurred.

The implication is now sharper: the geometry problem is not lack of hard-boundary headroom, and it is not necessary to label every microscopic local improvement as a move. A 1% utility-aware target removes low-value movement labels with almost no loss of oracle headroom. The next causal question is therefore whether **the exact same frozen representation and exact same classifier recipe as T018-C become safer when only the supervision target changes**.

PR #31, the bounded T018-E preparation-binding repair, has been accepted and squash-merged as `417141bcd697eb088e35e91686e338cb36885eef`. This does not change the T018-E 4/5 numbers. It does correct the historical evidence claim: the original T018-E run did not contemporaneously hash-bind `prepared.json`, so do not claim that the old verifier proved the complete pre-inference row-mapping chain. The retrospective 120/120 mapping/input audit is consistent, but it is not a retroactive cryptographic binding.

PR #32 is scientifically accepted but became non-mergeable after that independent provenance repair landed on `main`. Do **not** rerun or change T019-A to resolve that bookkeeping conflict. For the task below, the accepted frozen T019-A decisions are the artifact with SHA256 `d874bed74b0ebf68b680c8b6e60c8c5a44c9d82ce3cb7fd5730e16e53a980d45` on PR #32 head `91f750e871d2c133624bbe4972f3fb3086f25a6c`; record that exact origin if consumed from the PR branch.

---

# OPEN one-hour task — T019-B: grouped-OOF direct-direction probe with fixed 1% utility-deadband targets

**Expected work budget: about one hour. One hypothesis only: replacing T018-C's exact local-argmin labels with the accepted T019-A 1% utility-deadband labels, while changing nothing else about representation, folds, model, or training recipe, will reduce unnecessary learned boundary moves without destroying the established geometry gain.**

## Hypothesis / engineering objective

Run a controlled label-only intervention. Reuse the exact T018-C grouped-OOF pipeline and frozen 28-D candidate representation. The only scientific change is the x/y training target: use the accepted T019-A deadband class (`0.4`, `0.5`, `0.6`) instead of the T018-A exact local-argmin class.

This is development-only cross-validation. It is not a fresh qualification and does not authorize a new final selector in this cycle.

## Fixed inputs and settings

Use exactly:

- the same 120 development spatial episodes used by T018-C;
- the same frozen five hard-cross 28-D candidate features used by T018-C;
- the same five image-grouped folds and the same train/held-out image IDs as T018-C;
- the accepted T019-A frozen x/y targets, decisions SHA256 `d874bed74b0ebf68b680c8b6e60c8c5a44c9d82ce3cb7fd5730e16e53a980d45`;
- per-axis input `z = concat(f0, f_minus - f0, f_plus - f0)`, 84-D;
- separate x and y heads, therefore exactly 10 OOF heads total.

Freeze the model/training recipe to the literal T018-C recipe; no search or substitution:

- MLP `84 -> 64 -> 64 -> 3`;
- SiLU;
- ordinary unweighted cross-entropy;
- AdamW, learning rate `1e-3`, weight decay `1e-4`;
- batch size `256`;
- seed `7`;
- `100` epochs;
- final epoch only;
- normalization statistics computed from each fold's training image IDs only.

For each fold, reference-derived deadband labels may be opened only for the training IDs. Held-out inference must receive only the frozen label-free 84-D features and the trained head. Persist/hash all held-out logits, classes and final `(bx, by)` choices for all 120 rows **before** opening held-out reference MSE, T019-A held-out labels, family/condition metadata, or oracle quantities for evaluation.

The test-time rule remains non-negotiable: no test label, clean target, reference MSE, degradation mask/gain, condition/family ID, image-ID semantic shortcut, or evaluation metric may enter the held-out selection path.

## Fixed acceptance / stop criteria

After the complete 120-row OOF decision table is frozen, evaluate the same five deployment-facing development clauses:

1. pooled selected `H1 <= 0.97 * H0`;
2. pooled selected `H1 <= 1.03 * H*`;
3. offset selected `H1 <= 0.95 * H0`;
4. left/right selected `H1 <= 1.01 * H0`;
5. quadrants selected `H1 <= 1.01 * H0`.

Add two predeclared safety-transfer requirements tied directly to the T018-C baseline and the T018-E failure mode:

6. pooled harmful development episodes must be **strictly fewer than T018-C's 11**, i.e. `harmful <= 10`;
7. quadrants harmful episodes must be exactly `0`.

**Acceptance:** all seven requirements pass. Record `T019-B grouped-OOF deadband-direction positive` and stop. This would justify, but not execute, a later all-development freeze and a new fresh qualification.

**Stop:** if any requirement fails, record the literal negative and stop. Do not tune the deadband, architecture, class weights, epochs, features, confidence, abstention, margin, fold assignment, or safety threshold in this cycle.

## Required diagnostics

Report, without using them to change the method:

- pooled/family `H0`, selected `H1`, `H*`, ratios, and beneficial/equal/harmful counts;
- x/y/joint agreement with the T019-A deadband targets, after OOF decisions are frozen;
- predicted x/y/joint class distributions;
- number of non-center predictions on axes whose T019-A target is center;
- direct comparison to the accepted T018-C OOF result: pooled/family ratios, harmful count (`11` baseline), and movement counts;
- specifically for the 26 axes suppressed by T019-A relative to T018-A, how often T019-B predicts center versus a move;
- quadrants movement and harmful-case table, if any.

## Explicit non-goals

No T018-E fresh references/logits/features/outcomes for training or tuning; no reuse of the burned T018-E cohort; no fresh cohort; no final all-development fit; no new final selector; no rerendering; no new CLIP/TTT/features; no GPU requirement if the fixed small heads are faster/reproducible on CPU; no threshold sweep; no confidence/entropy/margin gate; no class weighting or resampling; no focal loss; no architecture expansion; no feature engineering; no family-specific logic; no scalar-energy fallback; no second seed; no alternate folds; no post-hoc calibration.

## Expected evidence

Commit a compact T019-B package containing:

- exact SHA bindings for the T018-C frozen features/folds and T019-A frozen target artifact;
- training receipts proving exactly 10 heads and the unchanged recipe;
- per-fold train/held-out image-ID membership and train-only normalization hashes;
- the 120-row OOF logits/classes/choices artifact plus SHA256 frozen before any held-out reference/family evaluation;
- focused tests showing held-out targets/references/metadata cannot affect inference;
- an independent verifier that reloads the saved heads/normalization and reproduces all 120 OOF logits/classes/choices without reading reference targets, then separately verifies the seven acceptance conditions after decision freeze;
- a concise `T019B_analysis.md` with the fixed acceptance vector and explicit statement that T018-E fresh data were not used for method development.

Stop after reporting T019-B. Do not begin an all-development fit or a fresh qualification until the next research-lead review.