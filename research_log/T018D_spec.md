# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications and evidence remain preserved in Git history and `research_log/`.

---

# Research-lead review — T018-C accepted; direct direction supervision recovers geometry signal

I reviewed Codex report `33197ebaf7247d13398e75747b2ba0667cee197f`, PR #27, frozen source `8fc63deeb825c87257c46546bc34e2ab3b4bffd8`, evidence `68f4333bf582258e3daf27d4f7d98ba482693d8b`, `ttie/direction_probe.py`, `ttie/direction_probe_run.py`, focused tests, independent verification, and the bound T016-B/T018-A artifacts against the T018-C contract. PR #27 is accepted and squash-merged as `b45bf9563e7c3f7e8aa27c923a35f1aabb95a5cb`.

The implementation respects the information boundary. Each held-out axis head receives only `concat(f0, f- - f0, f+ - f0)` from the frozen 28-D candidate features. Five folds use the exact accepted image-grouped split; normalization is fit only on each fold's 32 training IDs; held-out targets are not passed to fitting or inference. All ten heads and all 120 OOF logits/classes/combined decisions were persisted and hash-frozen before reference MSE, family/condition metadata, actual image-ID cross-checks, or T018-A held-out target diagnostics were opened. The independent verifier reproduces the saved heads/predictions and confirms freeze order and grouping. No clean target, test label, condition, image ID, mask/gain, oracle metric, or source-only derivative enters held-out inference.

T018-C is a genuine development-only positive: **5/5 clauses pass**. Pooled selected/H0 is `0.946340`, selected/H* is `1.018412`, offset/H0 `0.874913`, left/right/H0 `0.983709`, and quadrants/H0 `1.000000`. Joint T018-A target agreement rises to `87/120`; outcomes are **57 beneficial / 52 equal / 11 harmful**. All 40 quadrant episodes correctly remain at canonical center, while all 11 harmful selections occur in left/right. This is materially different from T018-B's 0/5 frozen scalar-energy failure and supports the specific causal interpretation that the frozen representation retains useful hard-geometry information, but the previous scalar value/energy projection and objective discard or distort too much direction information.

Do not over-interpret the 5/5 as qualification. The labels came from development references and the result is grouped OOF on the same 40 development IDs used to establish the mechanism. Further development retuning would now risk converting a clean positive into iterative overfitting. The next justified step is therefore **not another diagnostic, threshold, class-weighting pass, feature redesign, or fresh run in the same cycle**. First freeze the exact final two-head selector trained once on all accepted development rows, together with a minimal label-free inference interface. Fresh qualification will be a separate later task after that immutable barrier exists.

The non-negotiable rule remains unchanged: **test-time adaptation/selection must never consume test labels, clean targets, condition IDs, masks/gains, annotations, image IDs as semantic shortcuts, source-only reference gradients/Jacobians, or evaluation metrics.** Development references may supervise the final development-trained heads, but any future fresh image's inference path must use only the frozen label-free candidate features.

---

# OPEN one-hour task — T018-D: freeze the final all-development direct-direction selector

**Expected work budget: about one hour. One objective only: turn the accepted T018-C development mechanism into an immutable two-head inference artifact, without changing the feature construction, model family, training recipe, renderer, or decision rule. Do not run fresh qualification in this cycle.**

## Hypothesis / engineering objective

Train exactly one final x head and one final y head on **all 120 accepted T018 development episodes** using the same direction targets and the exact T018-C input:

`z_axis = concat(f0, f- - f0, f+ - f0)`.

The purpose is engineering freeze, not another scientific comparison. After training, expose a minimal inference function that accepts only the five required hard-cross candidate feature vectors (`center`, `x-lower`, `x-upper`, `y-lower`, `y-upper`; each 28-D), constructs the two 84-D inputs internally, applies the frozen all-development normalization/head, and returns x/y logits, classes, `(bx,by)`, and the corresponding hard candidate index. The inference API must have no reference/target/condition/image-ID arguments.

## Fixed inputs / settings

Use only accepted artifacts now on `main`:

- T016-B frozen 28-D candidate features and schema/provenance;
- T018-A accepted hard-axis reference targets, **training supervision only**;
- T018-C accepted code/recipe and class order.

Train two heads exactly once with the unchanged T018-C recipe:

- `84 -> 64 -> 64 -> 3`, SiLU after each hidden layer;
- ordinary unweighted 3-class cross entropy;
- AdamW, lr `1e-3`, weight decay `1e-4`;
- batch size `256`, exactly `100` epochs;
- seed `7`, final epoch only;
- class order `[0.5, 0.4, 0.6]` with argmax tie order center→lower→upper;
- per-dimension mean/population-std normalization fit on **all 120 development rows for that axis**, std clamp `1e-12`;
- CPU only.

Bind training inputs, source code, schema, target artifact, final model files, normalization buffers, histories, and final-selector receipt by commit/path/SHA256.

## Explicit non-goals

No new folds or OOF experiment; no fresh images or fresh manifest; no fresh qualification; no rerendering; no image reads; no CLIP/TTT/feature recomputation; no A6000; no extra candidates; no scalar energy; no confidence/abstention; no class weighting/focal loss; no alternate feature construction; no coordinate/family/condition/image-ID input; no larger/deeper model; no hyperparameter sweep; no early stopping; no checkpoint selection; no threshold or rule fit; no development metric used to choose or modify the final heads. Do not issue or execute T018-E in this cycle.

## Acceptance / stop criteria

This task is an **immutable-artifact gate**, not a new MSE gate.

Accept T018-D only if all of the following are true:

1. exactly two final heads are trained once from the fixed 120-row development set with the literal T018-C recipe;
2. training normalization uses all and only the 120 development rows, and final artifacts record those statistics exactly;
3. the inference API consumes only the five 28-D frozen label-free candidate features and returns deterministic decisions; mutating/removing any reference MSE, T018-A target, family/condition metadata, image ID, or oracle table after training cannot change inference;
4. final head files, normalization, class order, feature schema, source SHA, training-target/input hashes, and inference code are frozen in one immutable receipt suitable for a later fail-closed fresh launcher;
5. an independent verifier reloads the frozen artifacts and exactly reproduces logits/classes/decisions on a fixed replay set without reading reference metrics or targets.

If any item fails, stop and report the engineering blocker. Do not repair by changing architecture, features, training objective, or decision rule.

Development resubstitution accuracy/MSE may be reported only as descriptive diagnostics after the final artifacts are frozen; it is **not** an acceptance criterion and must not trigger retraining.

## Expected evidence

Commit a compact implementation plus focused tests and an independent verifier showing:

- exact two-head architecture/optimizer/seed/100-epoch recipe and one-time training;
- exact 120-row input/target provenance and all-development normalization;
- final `head_x`, `head_y`, histories, normalization buffers, schema/class order, source/input hashes, and immutable receipt;
- a reference-free inference API over exactly the five hard-cross 28-D feature vectors;
- mutation tests proving reference/target/family/image-ID/oracle artifacts are absent from and cannot influence inference;
- deterministic save/reload replay with exact logits/classes/combined `(bx,by)` decisions;
- independent verification of all hashes and replay arithmetic.

Stop after T018-D and report. **Do not generate or inspect a new fresh cohort and do not run qualification until the next research-lead review.**