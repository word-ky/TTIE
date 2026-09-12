# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior task specifications and evidence remain preserved in Git history and `research_log/`.

---

# Research-lead review — T016-B accepted as a controlled development negative

T016-B is accepted scientifically as a **0/5 negative**. The implementation respects the requested leakage boundary: the label-free scoring process consumes only persisted degraded pixels, the saved four Region2 EV/gamma corners, original frozen gate constants, frozen T006/T007 scorer assets, and the accepted T014 Sobolev head. It writes and hashes the complete 120×9 score/feature/selection artifact before the separate evaluator opens the accepted T016-A reference-MSE table. No clean image, condition label, image ID, reference MSE, oracle rank, mask/gain, or evaluation metric enters candidate scoring or selection. The non-negotiable rule remains: **test-time adaptation/selection must never use test labels or clean targets.**

The negative result is decisive for the raw frozen-energy selector. Spatial-pool selected MSE is `0.03785726`, which is `1.08029×` canonical Region2 and `1.16257×` the nine-hard oracle. Left/right is `1.07706×` Region2, quadrants `1.23780×`, and offset improves Region2 by only 3.68% versus the required 5%. All five predeclared clauses fail. Selector/oracle disagreement is 72.5%; mean per-episode Spearman between frozen energy and reference MSE is only `0.2398` over the 111 nonconstant cases.

At the same time, the capacity diagnosis from T016-A survives intact: the **nine-hard oracle** is only `1.00172×` the full 27-candidate oracle. Thus almost all of the previously observed geometric headroom is already present in the nine shifted-hard candidates; the failure is selection/ranking, not lack of candidate capacity.

A useful implementation detail now matters scientifically. The frozen T014 feature vector contains the gate constants, candidate-dependent CLIP exposure evidence, and the four EV/gamma corner values, but **no explicit `(b_x,b_y)` boundary coordinates**. In T016-B the corner values are identical across all nine candidates, so geometry is represented only indirectly through how the rendered pixels change the CLIP evidence. The current data therefore cannot tell us whether (a) the 28-D representation already contains enough information but the T014 head was never trained for cross-boundary ranking, or (b) explicit geometry state is required.

The reported inference-only numeric drift relative to the saved gradient-enabled T015 canonical path (max energy difference about `4.1e-6`) is acknowledged. It does not plausibly rescue this result: the performance gaps are large and the median winner margin is about `1.23e-2`. Do not rerun or tune T016-B to chase bitwise equality.

PR #17 is an experiment-delivery branch and currently conflicts with advancing `main`. Do not spend this cycle repairing PR topology or rerunning the GPU experiment. Use its compact immutable artifacts by commit/hash as inputs to the offline diagnostic below.

---

# OPEN one-hour task — T016-C: grouped OOF boundary-ranking feature-sufficiency probe

**Expected work budget: about one hour. One question only: is T016-B primarily a head/training-distribution failure, or is explicit boundary geometry missing from the 28-D representation?**

## Scientific hypothesis

T016-B shows that the frozen T014 head cannot rank shifted boundaries, while the nine-hard oracle shows strong capacity. Before designing a learned boundary predictor or changing the TTT state, run a strictly development-only supervised probe:

> If a small fixed-recipe head trained on the existing 28-D candidate features can rank boundaries on held-out images, then the representation is probably sufficient and the bottleneck is the T014 head/training distribution. If that probe fails but adding only `(b_x,b_y)` succeeds, explicit geometry coordinates are the missing state variable.

This is a **representation-sufficiency diagnostic**, not a deployable selector and not a fresh result.

## Fixed inputs

Use only the already-inspected T016-B/T016-A compact artifacts. Do not rerender, rerun CLIP, rerun TTT, or use any new image ID.

From evidence commit `4062e01cb93de731c394015c5ac741d6c08e04d8`, reuse and hash-verify:

- the 120×9 saved T016-B 28-D feature table and candidate order;
- the accepted T016-A nine-hard reference MSE values already joined in the T016-B evaluation artifact;
- canonical Region2 and nine-hard oracle MSEs for reporting only.

The 40 unique image IDs are now permanent development data. Image ID and condition may be used **only to construct folds/report metrics**, never as model inputs.

## Deterministic grouped cross-validation

Create exactly five folds by sorted unique image ID: assign image `j` in sorted order to fold `j mod 5`. Keep all three spatial conditions and all nine candidates for the same image in the same fold. No random split and no fold search.

For every fold, train on 32 image IDs and evaluate on the held-out 8 image IDs. Concatenate only held-out predictions to form one 120-episode out-of-fold result. All normalization statistics must come from the training folds only.

## Exactly two fixed probes

Train exactly two candidate-value heads with **no hyperparameter search**:

1. `probe28`: input is the saved 28-D T016-B feature vector only.
2. `probe30`: input is the same 28-D vector plus two explicit geometry coordinates
   `gx=(b_x-0.5)/0.1`, `gy=(b_y-0.5)/0.1`.

Use the T014 value-head recipe for both, except for input dimension:

- MLP `D -> 64 -> 64 -> 1`, SiLU;
- target `log(MSE + 1e-6)`;
- train-only x/y standardization;
- Huber loss, delta 1;
- AdamW `lr=1e-3`, weight decay `1e-4`;
- batch 256, seed 7, 100 epochs, final epoch only.

Train one head per fold per probe (10 tiny heads total). Do not use Sobolev/gradient supervision, pairwise ranking loss, condition labels, image embeddings, image IDs, candidate IDs, masks, or any feature other than those stated. We are isolating representation, not searching for the best selector.

At held-out inference, score all nine candidates independently and choose the minimum predicted value with the existing lexicographic tie order. Reference MSE is used only after each held-out fold's predictions/selections are finalized.

## Required diagnostics

For `probe28` and `probe30`, report out-of-fold:

- spatial-pool selected MSE and ratios to canonical Region2, nine-hard oracle, and the frozen T016-B selector;
- the same for `left_right`, `quadrants`, and `offset_left_right_40`;
- boundary-selection counts and oracle disagreement;
- Spearman between predicted values and reference MSE per episode, with null/constant cases explicit;
- per-fold selected MSE so one fold cannot hide a collapse;
- final train Huber for each fold (diagnostic only; never select an epoch/model from it).

Also report `probe30 / probe28` spatial MSE and median-Spearman difference.

## Fixed interpretation / stop criteria

Evaluate each probe with the **same five clauses used in T016-B**:

1. spatial selected MSE `<= 0.97 ×` canonical Region2;
2. spatial selected MSE `<= 1.05 ×` nine-hard oracle;
3. offset selected MSE `<= 0.95 ×` canonical Region2;
4. left/right selected MSE `<= 1.01 ×` canonical Region2;
5. quadrants selected MSE `<= 1.01 ×` canonical Region2.

Interpret only after all OOF predictions are frozen:

- If `probe28` passes all five, conclude only that the **existing 28-D representation is development-sufficient for boundary ranking**; the frozen T014 head/training distribution is the likely bottleneck. Do not yet train a deployable selector.
- If `probe28` fails but `probe30` passes all five, conclude only that **explicit boundary coordinates materially restore development rankability**; geometry state is a justified candidate for the next design.
- If both fail, conclude that this simple feature family/probe does not establish rankability; do not enlarge the model or add spatial features in this cycle.
- If both pass, treat 28-D as sufficient; explicit coordinates are not necessary evidence.

No alternate threshold, fold, epoch, architecture, or loss may become a pass route after seeing results.

## Non-goals

Do not:

- use new/fresh images or make a fresh-generalization claim;
- rerender candidates or rerun CLIP/A6000 scoring;
- modify/retrain T014 or T016-B scientific assets;
- train a deployable boundary predictor;
- add learned masks, spatial tokens, convolutional/image features, condition cues, or ViT-style geometry encoders;
- optimize boundary coordinates continuously;
- run detector/meta-learning/prompt retraining/ViT3;
- start the next experiment automatically.

## Expected evidence / DONE condition

Within this cycle, commit:

- a small offline script/module implementing the deterministic grouped folds and the two fixed probes;
- focused tests for group isolation, train-only normalization, exact input dimensions, deterministic tie handling, and reference not entering held-out scoring;
- immutable OOF prediction/selection tables for both probes;
- a concise `T016C_analysis.md` containing all fixed clauses, per-condition/per-fold metrics, Spearman diagnostics, and exact source-artifact hashes.

CPU is sufficient. Stop after reporting T016-C. Do not automatically begin a learned boundary model or fresh validation.
