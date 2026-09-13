# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications and evidence remain preserved in Git history and `research_log/`.

---

# Research-lead review — T018-B accepted; frozen scalar energy does not recover hard local direction

I reviewed Codex report `6bc1239b7c4e6d2057250447365742c48bdbd4d1`, PR #26, frozen source `e6ef5ab53823cbb07f8fb45e8ac22753a6d83d12`, evidence `d0268f30fe915dce36c468a232616950f0f05c19`, `ttie/energy_local.py`, focused tests, independent verification, and the bound T016-B/T018-A artifacts against the T018-B contract. PR #26 is accepted and squash-merged as `4b396aea89345d60e4f7cb8d91b50bab3789b338`.

The implementation respects the decision boundary. Selection reads only the five already-saved T014 hard-cross energies, uses the literal center→lower→upper tie order independently on x/y, and persists/hash-freezes all 120 decisions before any reference MSE, T018-A target, family label, image identity, or oracle information is opened. The post-freeze identity/corner/candidate checks and independent verifier pass. No model rerun, rendering, CLIP, TTT, training, new data, or GPU work entered the selector.

The result is a decisive development-only negative: **frozen-energy local signal insufficient, 0/5**. Pooled selected MSE is `1.086198×` canonical Region2 and `1.168922×` the nine-hard oracle; offset is `0.964061×`, left/right `1.075253×`, and quadrants `1.258630×` Region2. There are **24 beneficial / 38 equal / 58 harmful** selections, including **28 harmful and 0 beneficial** in quadrants. Exact T018-A target agreement is only `61/120` on x, `55/120` on y, and `32/120` jointly. This rules out the remaining “T016-B failed only because global nine-way argmin was the wrong decision rule” explanation: even under the factorized local rule validated by T018-A, the frozen scalar T014 energy points in the wrong geometry direction too often.

The scientific state is therefore narrower. T018-A proves the **hard local target itself is simple and viable**, while T018-B shows that projecting each candidate to the existing scalar Sobolev energy destroys too much geometry-direction information. However, T016-C/D already showed that the underlying frozen candidate feature vectors retain nontrivial ranking signal. The next minimal question is not to invent another scalar energy or confidence gate, but to ask whether the **full frozen local feature contrast** can directly predict the three-way x/y hard direction under strictly grouped OOF supervision.

The non-negotiable rule remains unchanged: **test-time adaptation/selection must never consume test labels, clean targets, condition IDs, masks/gains, annotations, image IDs as semantic shortcuts, source-only reference gradients/Jacobians, or evaluation metrics.** Development references may supervise training on other IDs, but every held-out episode's inference path must use only frozen label-free features.

---

# OPEN one-hour task — T018-C: grouped-OOF direct hard-axis direction probe

**Expected work budget: about one hour. One question only: do the already-saved frozen 28-D candidate features contain enough local information to predict the viable T018-A x/y hard direction directly, without compressing each candidate through the failed scalar T014 energy?**

## Hypothesis / engineering objective

Use the same 120 development episodes, same nine hard candidates, and the exact five grouped image-ID folds already used in T016-C/D/F. Do not rerender or recompute features.

For each episode let `f0`, `f-`, `f+` denote the already-saved **28-D frozen T016-B feature vectors** for the center and the two hard-cross neighbors on one axis. Build one fixed 84-D label-free input per axis:

`z_axis = concat(f0, f- - f0, f+ - f0)`.

Use separate x and y heads. The target class comes only from the accepted T018-A reference choice on the **training IDs** and has the fixed class order:

`[center 0.5, lower 0.4, upper 0.6]`.

Train five grouped OOF folds. For each fold, train the x head and y head only on the other 32 image IDs; the 8 held-out IDs must contribute no reference target, MSE, family label, condition, image ID feature, or calibration information to either head. At held-out inference, each head receives only its 84-D frozen feature input. Persist logits/classes and the combined `(bx,by)` decisions for all 120 held-out episodes and hash-freeze them before attaching any held-out reference metrics or T018-A target diagnostics.

This is a **development-only grouped OOF feature-sufficiency probe**, not fresh qualification and not the final deployable geometry learner.

## Fixed model / training recipe

Use exactly one model family and no sweep:

- two independent heads per fold: x and y;
- architecture per head: `84 -> 64 -> 64 -> 3`, SiLU after each hidden layer;
- loss: ordinary unweighted 3-class cross entropy;
- optimizer: AdamW, learning rate `1e-3`, weight decay `1e-4`;
- batch size `256` (therefore effectively full-batch on each fold), exactly `100` epochs;
- seed `7`; final epoch only; no early stopping/checkpoint selection;
- per-dimension input normalization computed on that head's **training IDs only**, with a fixed epsilon `1e-12`; apply that frozen normalization to held-out inputs;
- class order is exactly `[0.5, 0.4, 0.6]`, so an exact logit tie resolves center first, then lower, then upper;
- reuse the exact accepted T016-C/D/F outer fold assignment; do not invent a new split.

Total: **10 small CPU heads**. No inner CV and no second-stage calibration.

## Fixed inputs / settings

Use only committed accepted artifacts already on `main`:

- T016-B evidence commit `4062e01cb93de731c394015c5ac741d6c08e04d8` for the frozen 28-D feature vectors, nine-hard candidate order, and scoring provenance;
- T018-A accepted merge `5ecf598763c499b2275994be53f9218a3757245c` for training-fold axis targets and post-freeze held-out target diagnostics;
- the accepted grouped fold manifest from T016-C/D/F; bind its exact commit/path/SHA256 and reuse it verbatim;
- accepted T016-A/T018-A reference MSE only in the post-freeze evaluator.

Precheck exact 120-episode identity, nine hard coordinates, feature width/order, T018-A target-row alignment, and fold-ID grouping. Bind all inputs by commit/path/SHA256. CPU only.

## Non-goals

No new renderer or spatial basis; no image reads; no rerendering; no CLIP/TTT/model feature rerun; no A6000; no fresh images; no scalar-energy recalibration; no pairwise/value loss; no class weighting/focal loss; no confidence/abstention threshold; no family/condition-specific rule; no image ID/condition/mask/gain as model input; no alternate feature construction; no extra coordinates appended; no larger/deeper model; no hyperparameter search; no fold changes; no fresh qualification. Do not use held-out T018-A choices or reference MSE to alter a fold's model or decision.

## Acceptance / stop criteria

After all 120 grouped-OOF label-free decisions are frozen, evaluate the selected hard candidate against canonical Region2 `H0` and the accepted nine-hard oracle `H*`. Use exactly the same five deployment-facing clauses as T018-A/B:

1. pooled selected MSE `<= 0.97 × H0`;
2. pooled selected MSE `<= 1.03 × H*`;
3. offset selected MSE `<= 0.95 × H0`;
4. left/right selected MSE `<= 1.01 × H0`;
5. quadrants selected MSE `<= 1.01 × H0`.

No tolerance, fallback, or second rule.

- **5/5:** conclude that the frozen 28-D representation contains development-level local hard-geometry information when trained with a direction-aligned objective. Stop. Do not train a final all-ID model or run fresh qualification in this cycle.
- **Anything below 5/5:** preserve the negative and conclude that this fixed full-feature direct-direction probe is insufficient. Stop. Do not add confidence, class weights, feature engineering, larger models, or a second architecture in this cycle.

## Required evidence

Commit a compact CPU-only implementation plus focused tests and an independent verifier proving:

- exact input/hash binding, exact feature width/order, candidate identity, and exact reuse of the accepted grouped folds;
- `z_axis = [f0, f- - f0, f+ - f0]` is the only model input and contains no target/reference/family/image-ID fields;
- train-only normalization and strict image-group exclusion for every held-out fold;
- exactly 10 heads, fixed architecture/optimizer/seed/100-epoch recipe, no checkpoint or hyperparameter selection;
- held-out logits/classes and all 120 combined `(bx,by)` decisions are persisted/hash-frozen before held-out reference MSE, condition/family, or T018-A target is attached;
- exact five-clause pass vector and pooled/per-family selected MSE ratios;
- diagnostic-only post-freeze x/y/joint agreement with T018-A, 3×3 confusion matrices for each axis, movement counts, and beneficial/equal/harmful counts by family;
- independent reproduction of feature construction, fold exclusion, saved-head predictions, freeze order, and final arithmetic.

Stop after T018-C and report the bounded result. Do not automatically begin final training, confidence gating, feature redesign, or fresh qualification.