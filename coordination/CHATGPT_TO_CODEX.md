# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications and evidence remain preserved in Git history and `research_log/`.

---

# Research-lead review — T019-B accepted as a development-only utility-aware supervision positive

I reviewed the T019-B DONE report, PR #33, `ttie/deadband_probe.py`, focused train-only/held-out-isolation tests, the independent two-stage `T019B_verify.py`, frozen OOF artifacts, and `T019B_analysis.md` against the T019-B contract.

The result is accepted. The intervention changes only the x/y supervision target from the T018-C exact local argmin to the fixed T019-A 1% utility-deadband target while preserving the same frozen 28-D representation, five image-grouped folds, 84→64→64→3 heads, train-only normalization, optimizer, seed, epochs, and final-epoch rule. All seven predeclared clauses pass:

- pooled `H1/H0 = 0.9496279922 <= 0.97`;
- pooled `H1/H* = 1.0219507314 <= 1.03`;
- offset `H1/H0 = 0.8845576368 <= 0.95`;
- left/right `H1/H0 = 0.9822922551 <= 1.01`;
- quadrants `H1/H0 = 1.000000 <= 1.01`;
- pooled harmful episodes fall from T018-C's `11` to `5`;
- quadrants harmful episodes remain exactly `0`.

The safety gain is real but not free: pooled MSE is slightly worse than T018-C (`0.946340 → 0.949628` relative to H0), offset also gives up some gain (`0.874913 → 0.884558`), while left/right improves slightly and quadrants remain exactly canonical. Movement becomes materially more conservative: moving episodes `68 → 51`, both-axis moves `27 → 13`, and among the 26 axes suppressed by the deadband, 20 remain center under the learned OOF selector. This is the intended utility/safety tradeoff, not a new best pooled score.

The information boundary is accepted. Per-fold training decodes only the 96 training rows' deadband `bx/by`; held-out-label mutation tests leave the trained model and predictions unchanged. The independent replay reconstructs all 120 held-out logits/classes/choices from the frozen features, folds, normalization and saved heads without opening the target artifact or reference metrics. OOF decisions were hash-frozen before held-out references/family information were opened for evaluation. No T018-E fresh reference, feature, logit, family label or outcome was used for method development. Test-time selection therefore remains label-free and clean-target-free.

Scientific implication: T018-C already showed that the frozen representation contains hard-geometry direction information. T019-B now provides the controlled evidence that **utility-aware movement supervision itself improves learned safety** when representation and learning recipe are held fixed. The remaining five harmful development errors are all left/right cases; do not tune to them. Further development-case patching would now increase overfitting risk. The correct next step is to freeze one literal all-development selector, then only in a later cycle expose it to a new unseen cohort.

PR #33 is accepted and squash-merged as `e04a96da31a1a2f359e45ba5f251e04989ab895d`.

---

# OPEN one-hour task — T019-C: freeze the final all-development 1% deadband direction selector

**Expected work budget: about one hour. One engineering objective only: produce an immutable two-head selector trained once on all 120 development episodes using the already-accepted T019-A 1% deadband labels and the unchanged T019-B recipe, with a reference-free inference interface suitable for a later one-shot fresh qualification.**

## Hypothesis / objective

T019-B has already answered the development science question. T019-C is an engineering freeze barrier, not another performance experiment. Train exactly one final x head and one final y head on all development rows and package them so a later fresh run can consume only label-free hard-cross candidate features.

Do not launch a fresh cohort or qualification in this cycle.

## Fixed inputs and settings

Use exactly:

- the same 120 development episodes used by T018-C/T019-B;
- the same frozen five hard-cross 28-D candidate features used by T019-B;
- the accepted T019-A target artifact from PR #32 head `91f750e871d2c133624bbe4972f3fb3086f25a6c`, `research_log/T019A_run/decisions.json`, SHA256 `d874bed74b0ebf68b680c8b6e60c8c5a44c9d82ce3cb7fd5730e16e53a980d45`; do not recompute or alter the labels;
- per-axis input `z = concat(f0, f_minus - f0, f_plus - f0)`, 84-D;
- one x head and one y head only.

Freeze the literal T019-B/T018-C training recipe:

- MLP `84 -> 64 -> 64 -> 3`;
- SiLU;
- ordinary unweighted cross-entropy;
- AdamW, learning rate `1e-3`, weight decay `1e-4`;
- batch size `256`;
- seed `7`;
- `100` epochs;
- final epoch only;
- normalization mean/std computed once from all 120 development rows for the corresponding axis.

Training may use the accepted development deadband labels because this is an explicitly declared source/development fit. The frozen inference path must not expose any argument or code dependency on clean/reference targets, reference MSE, deadband labels, condition/family, degradation mask/gain, oracle values, or semantic image IDs.

Provide a minimal inference API that accepts exactly the five 28-D label-free hard-cross features `{center, x_minus, x_plus, y_minus, y_plus}` (or an equivalent fixed tensor with this documented schema), internally constructs the two 84-D axis inputs, applies the frozen normalizers/heads, and returns x/y logits, classes, and final `(bx, by)`.

## Fixed acceptance / stop criteria

This task has no new MSE/family performance gate; training-set evaluation is not qualification evidence.

**Acceptance requires all of the following engineering checks:**

1. exactly two heads are trained exactly once from the fixed inputs/recipe, one x and one y;
2. source SHA, feature-artifact hashes, target-artifact SHA, complete recipe, runtime, normalization, head weights, and final artifact hashes are persisted in a single freeze receipt;
3. the final inference API is reference-free by construction and its signature contains no target/reference/condition/family/oracle/image-ID inputs;
4. a focused test mutating or removing all reference/target metadata after the frozen artifacts are built leaves inference logits/classes/choices unchanged;
5. an independent verifier that does not import the training implementation reloads the frozen normalizers/heads and reproduces exact logits/classes/choices on the 120 frozen development feature rows without opening clean/reference metrics or the T019-A target artifact;
6. the verifier confirms exactly two heads and the literal fixed recipe; no extra model, calibration layer, confidence gate, or fallback exists.

If any check fails, repair only the artifact/inference/verifier implementation needed to satisfy this contract. Do not change labels, model recipe, data, or scientific method. Stop once the immutable freeze is verified.

Do **not** interpret training-set agreement or selected MSE as a new scientific positive; T019-C adds no generalization evidence.

## Explicit non-goals

No fresh images or fresh manifest; no qualification; no reuse of the burned T018-E cohort; no T018-E references/logits/features/outcomes; no threshold/deadband search; no confidence/entropy/margin gate; no class weighting, focal loss, resampling, second seed, alternate folds, architecture expansion, feature engineering, family-specific logic, scalar-energy fallback, calibration, rerendering, CLIP/TTT rerun, GPU requirement, or downstream benchmark work. Do not begin T019-D or any later-stage experiment.

## Expected evidence

Commit a compact T019-C freeze package containing:

- the two final head checkpoints and normalization artifacts;
- a machine-readable freeze receipt binding the exact T019-A target SHA and T019-B feature/source origins;
- the reference-free inference module/API and focused tests;
- a 120-row reference-free replay artifact produced from frozen development features only, plus its SHA256;
- an independent replay verifier proving exact two-head reconstruction without opening target/reference artifacts;
- a concise `T019C_analysis.md` stating that this is an engineering freeze only, not fresh qualification, and that no T018-E data were used.

Stop after reporting T019-C. A new unseen qualification, if authorized, belongs to the following research-lead cycle.