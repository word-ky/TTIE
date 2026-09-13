# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications and evidence remain preserved in Git history and `research_log/`.

---

# Research-lead review — T018-A accepted; hard local target is viable

I reviewed Codex report `d6ef07b4149889cf21e9805e21b604cedd9f30df`, PR #25, frozen source `c9f3a0ac763f2f0e5b3a8a67d009f2792f8fe118`, evidence `8a67f0532210dbdf67be8e0f5b181ac6a7582085`, `ttie/hard_local.py`, the focused tests, independent verifier, and the accepted T016-A table/config bindings against the T018-A contract. PR #25 is accepted and squash-merged as `5ecf598763c499b2275994be53f9218a3757245c`.

The implementation matches the declared reference-only CPU audit. Selection reads only the five predeclared hard-cross reference values, uses the literal center→lower→upper tie order independently on x and y, and freezes both choices and `H0/H1/H*` quantities before family labels are attached. The independent verifier recomputes all 120 choices and group statistics without importing the audit implementation. No images, rendering, CLIP, TTT, training, new data, or GPU work entered the result.

The scientific result is a strong development-only positive: **hard local-direction target viable, 5/5**. Pooled `H1/H0 = 0.930464`, pooled `H1/H* = 1.001327`, offset `H1/H0 = 0.850731`, left/right `0.963289`, and quadrants `0.999946`. The factorized local target recovers **98.26%** of pooled nine-hard oracle headroom; `115/120` choices lie in the exact nine-hard oracle tie set. There are **69 beneficial, 51 unchanged, and 0 harmful** combined moves, with zero observed pure x/y interaction failures. This resolves the previous ambiguity: the hard renderer itself has a simple center-local factorized target; the remaining problem is not target capacity but whether a label-free signal can recover that target.

The next smallest question is therefore deliberately narrower than training a new geometry network. T016-B already saved the frozen T014 Sobolev energy on the same nine hard candidates. Before adding any new predictor, test whether that existing label-free energy contains useful **local directional** information when used in the same factorized hard-cross rule that T018-A validated. T016-B's failed global nine-way argmin does not answer this because scalar global ranking and local axis direction are different decision rules.

The non-negotiable rule remains unchanged: **test-time adaptation/selection must never consume test labels, clean targets, condition IDs, masks/gains, annotations, image IDs as semantic shortcuts, source-only reference gradients/Jacobians, or evaluation metrics.** In T018-B, the boundary decision must be finalized from already-saved label-free energy scores before any reference MSE or family metadata is read.

---

# OPEN one-hour task — T018-B: frozen-T014 hard-cross local-direction audit

**Expected work budget: about one hour. One question only: does the already-frozen T014 Sobolev energy recover the viable T018-A hard local direction when selection is factorized on the hard cross, without any reference information at decision time?**

## Hypothesis / engineering objective

Reuse the accepted T016-B label-free scoring artifact for the same 120 spatial episodes and nine `tau=0` hard candidates. Do **not** rerun the model. For each episode, read only the already-saved frozen T014 scalar energy/score for:

- canonical `E0 = E(0.5,0.5,0)`;
- x cross `Ex- = E(0.4,0.5,0)`, `Ex0 = E0`, `Ex+ = E(0.6,0.5,0)`;
- y cross `Ey- = E(0.5,0.4,0)`, `Ey0 = E0`, `Ey+ = E(0.5,0.6,0)`.

Choose `bx_energy` as the minimum-energy member of `{0.5,0.4,0.6}` on the x cross and `by_energy` independently on the y cross. **Tie order is literal: center `0.5`, then lower `0.4`, then upper `0.6`.** The selected output is the already-existing hard candidate `(bx_energy, by_energy, 0)`.

This decision stage is label-free. It must not read reference MSE, T018-A target choices, condition, image ID, oracle rank, or evaluation metrics. Persist and hash all 120 decisions first. Only after that freeze may an evaluation stage join the accepted T018-A/T016-A reference table and family labels.

## Fixed inputs / settings

Use only committed accepted artifacts already on `main`:

- T016-B evidence commit `4062e01cb93de731c394015c5ac741d6c08e04d8`, specifically its frozen nine-hard candidate order and saved T014 energy/score vectors plus scoring receipt/config;
- merged T018-A artifacts from PR #25 / merge `5ecf598763c499b2275994be53f9218a3757245c` for post-freeze reference evaluation and target-agreement diagnostics only;
- the accepted T016-A hard-candidate reference table only as the post-freeze metric source if T018-A does not already expose the required per-candidate values.

Precheck exact identity of the 120 episodes and the nine hard candidate coordinates across artifacts. Bind all inputs by commit/path/SHA256. CPU only.

## Non-goals

No training or fine-tuning; no new geometry head; no pointwise/pairwise loss; no confidence/abstention threshold; no condition-specific rule; no score calibration or normalization change; no alternate tie order; no alternate boundary step/grid; no soft renderer or `tau>0`; no image reads; no rerendering; no CLIP/TTT rerun; no A6000; no new/fresh images; no detector/meta/prompt/ViT3 work. Do not use T018-A reference choices to repair energy decisions.

## Acceptance / stop criteria

After the 120 label-free decisions are frozen, evaluate the selected hard candidate against canonical Region2 `H0` and the accepted nine-hard reference oracle `H*`. Qualify **only if all five predeclared clauses pass**:

1. pooled selected MSE `<= 0.97 × H0`;
2. pooled selected MSE `<= 1.03 × H*`;
3. offset selected MSE `<= 0.95 × H0`;
4. left/right selected MSE `<= 1.01 × H0`;
5. quadrants selected MSE `<= 1.01 × H0`.

No tolerance, fallback, or second rule.

- **5/5:** conclude that the existing frozen T014 Sobolev energy contains development-level label-free local hard-geometry signal under the factorized rule. Stop; do not launch fresh qualification or integrate it into TTT in this cycle.
- **Anything below 5/5:** preserve the negative and conclude that T018-A target viability does not transfer through the existing frozen T014 scalar energy. Stop; do not train a dedicated geometry predictor or tune a gate in this cycle.

## Required evidence

Commit a compact CPU-only audit plus focused tests and an independent verifier proving:

- exact input/hash binding and exact nine-hard candidate identity;
- the decision stage reads only the five saved T014 hard-cross scores per episode;
- literal center→lower→upper tie handling on each axis;
- all 120 `(bx_energy,by_energy)` decisions are persisted and hash-frozen before reference MSE, condition, image ID, T018-A target choices, or oracle information is attached;
- the exact five-clause pass vector and pooled/per-family selected MSE ratios;
- diagnostic-only agreement with T018-A after freeze: x-axis match, y-axis match, joint target match, movement counts, and counts of beneficial/equal/harmful selected outputs by family;
- for harmful selections, report row index, energy-chosen axes, T018-A reference axes, and post-freeze `H0/Hselected/H*` only; do not use those examples to alter the rule;
- independent arithmetic/freeze-order verification.

Stop after T018-B and report the bounded result. Do not automatically begin T018-C or any trainable geometry method.