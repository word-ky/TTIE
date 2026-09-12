# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications and evidence remain preserved in Git history and `research_log/`.

---

# Research-lead review — T016-C accepted as a controlled development negative

T016-C is accepted scientifically as a **0/5 negative for both fixed pointwise-value probes**. I reviewed the report/evidence at main report commit `49d7373d2d3ec0e1495d10a37969b89ba1cef5eb`, engineering/evidence head `433683eccad24dc763450be6a546072a72e0910b`, and PR #18 against the T016-C contract.

The implementation respects the intended leakage boundary. The five folds are deterministic and grouped by image ID; all three conditions and all nine boundary candidates for one image stay in the same fold. Fold fitting reads reference MSE only for the 32 training IDs, uses train-only normalization, and held-out scoring receives candidate feature tensors only. The ten fold heads and both 120-episode OOF tables are frozen/hash-bound before the evaluator attaches held-out reference metrics. Image ID, condition, candidate ID, clean pixels, reference MSE, oracle rank, masks/gains, or evaluation metrics are not model inputs. This remains development-only supervision, not a deployable or fresh result. The non-negotiable rule remains: **test-time adaptation/selection must never use test labels or clean targets.**

The literal outcome is negative. `probe28` gives spatial-pool MSE `0.03580669 = 1.02178×` canonical Region2 and `1.09959×` the nine-hard oracle. `probe30`, which adds only explicit boundary coordinates `(g_x,g_y)`, improves to `0.03545103 = 1.01163×` Region2 and `1.08867×` oracle, but still fails all five clauses. Its offset ratio `0.95132×` Region2 narrowly misses the required `0.95`, while left/right is `1.02438×` and quadrants `1.07720×`, both violating the 1% non-inferiority requirements. Adding coordinates lowers pooled MSE only about 0.99% relative to `probe28` and does **not** establish that geometry coordinates alone solve boundary ranking.

Do not over-interpret this as universal representation insufficiency. A key remaining confound is the **training objective**. Both probes regress absolute `log(MSE)` independently across images, while deployment only needs the *within-episode ordering of nine candidates*. Absolute image difficulty is therefore a nuisance component in the target. T016-C improves mean/median rank correlation (`probe30` pooled mean Spearman `0.5275`, median `0.5833`) and substantially beats the failed frozen T016-B selector, yet argmin selection is still unsafe—especially on quadrants, where the reference oracle is canonical for 39/40 episodes. This makes a rank-aligned training-objective diagnostic more informative than adding a larger model or richer spatial features now.

PR #18 is accepted for its scientific evidence, but do not spend this cycle repairing PR topology or launching new data. Consume its immutable compact artifacts by commit/hash. No fresh-generalization claim is authorized.

---

# OPEN one-hour task — T016-D: grouped OOF pairwise boundary-ranking probe

**Expected work budget: about one hour. One question only: did T016-C fail mainly because absolute value regression is misaligned with the within-image ranking problem?**

## Scientific hypothesis

For each test image/condition, only the ordering of the nine shifted-hard boundaries matters. T016-C trained on absolute restoration value, which includes large image-level difficulty variation that cancels out at selection time. Test the minimal alternative:

> If the same small scalar head and the same candidate features succeed when trained only on within-episode pairwise ordering, then the feature representation is development-rankable and the main bottleneck was pointwise value supervision. If pairwise supervision still fails, do not enlarge the model or feature family in this cycle.

This is a **development-only loss-alignment diagnostic**, not a deployable selector and not fresh evaluation.

## Fixed inputs and folds

Use only the already-inspected T016-C/T016-B compact artifacts. No new image, rendering, CLIP, TTT, A6000 work, or recomputation of accepted reference MSE.

Use the same 40 development IDs, the exact same sorted-ID `j mod 5` folds, the same 120 episodes, the same nine lexicographic hard-boundary candidates, and the same saved candidate features/reference MSEs used by T016-C. Hash-verify all source artifacts before training.

Keep every image's three conditions and nine candidates in one fold. For each fold: 32 training IDs, 8 held-out IDs. Held-out reference values must not be read by the training/scoring path.

## Exactly two fixed rank probes

Train exactly:

1. `rank28`: saved 28-D candidate feature only;
2. `rank30`: the same 28-D feature plus the same two coordinates used in T016-C, `gx=(b_x-0.5)/0.1`, `gy=(b_y-0.5)/0.1`.

Architecture and optimizer are fixed for both:

- scalar MLP `D -> 64 -> 64 -> 1`, SiLU;
- train-only input standardization;
- seed 7;
- AdamW `lr=1e-3`, weight decay `1e-4`;
- pair batch size 256;
- exactly 100 epochs; final epoch only;
- no architecture/lr/epoch/fold search and no model selection.

Do **not** use scalar-MSE Huber in T016-D. Use exactly one unweighted within-episode pairwise logistic objective. For every training episode generate all lexicographic unordered candidate pairs `i<j`. Let `m_i,m_j` be their fixed reference MSEs and `s_i,s_j` the scalar head outputs, where lower score means better candidate. Skip only exact MSE ties. Define `r=+1` when `m_i < m_j`, otherwise `r=-1`, and train with

`L_pair = mean( softplus( r * (s_i - s_j) ) )`.

Pairs must never cross images/conditions. Use every non-tied training pair exactly once per epoch in a deterministic base order followed by the fixed seed-7 epoch permutation. No margin weighting, hard-negative mining, MSE-difference weighting, pair subsampling, temperature, auxiliary value loss, or calibration.

At held-out inference, evaluate the scalar head independently on all nine candidates and choose the minimum score with the existing lexicographic exact-tie rule. Freeze all ten heads and both OOF score/selection tables before any held-out reference evaluation.

## Required diagnostics

For `rank28` and `rank30`, report exactly the same deployment-facing diagnostics as T016-C:

- spatial-pool selected MSE and ratios to canonical Region2, nine-hard oracle, frozen T016-B, and the corresponding T016-C value probe;
- `left_right`, `quadrants`, and `offset_left_right_40` MSE/ratios;
- candidate-selection counts, oracle counts, disagreement/outside-oracle-tie rates;
- per-episode Spearman between rank scores and reference MSE, with constant/null cases explicit;
- per-fold held-out selected MSE;
- final training pairwise loss and number of non-tied training pairs per fold.

Also report `rank30 / rank28` spatial MSE and each rank probe relative to its pointwise T016-C counterpart.

## Fixed five-clause acceptance / interpretation

Use the unchanged five clauses:

1. spatial selected MSE `<= 0.97 ×` canonical Region2;
2. spatial selected MSE `<= 1.05 ×` nine-hard oracle;
3. offset selected MSE `<= 0.95 ×` canonical Region2;
4. left/right selected MSE `<= 1.01 ×` canonical Region2;
5. quadrants selected MSE `<= 1.01 ×` canonical Region2.

Interpret only after OOF outputs are frozen:

- If `rank28` passes all five: conclude only that **the existing 28-D representation is development-rankable under rank-aligned supervision**; T016-C's absolute-value objective was the main diagnosed mismatch. Do not yet launch fresh evaluation.
- If `rank28` fails but `rank30` passes: conclude only that **explicit geometry plus rank-aligned supervision jointly restores development rankability**. Do not yet call geometry coordinates sufficient in isolation.
- If both pass: treat 28-D as sufficient; explicit coordinates are not necessary evidence.
- If both fail: conclude that this small scalar-head feature family still does not establish safe boundary ranking even with directly aligned supervision. Stop; do not add a larger model, image tokens, continuous boundary optimizer, or fresh data in this cycle.

No alternate threshold, abstention rule, pair weighting, loss, fold, or epoch may become a pass route after seeing results.

## Non-goals

Do not train a deployable predictor; do not add confidence gating/abstention, image embeddings, spatial maps/tokens, condition cues, candidate IDs, detector features, ViT-style encoders, learned masks, continuous boundaries, meta-learning, prompt retraining, or ViT3. Do not rerun T014/T016-B/C or touch fresh IDs. Do not use held-out clean/reference values before OOF scoring is frozen.

## Expected evidence / DONE condition

Commit one compact CPU-only implementation plus focused tests proving: image-grouped folds; pairs never cross episodes; exact ties are excluded deterministically; train-only input normalization; exact 28/30-D inputs; held-out targets cannot enter training/scoring; OOF outputs are frozen before evaluation; lexicographic inference ties are deterministic. Persist all ten heads/training histories, two OOF score tables, source hashes, and a concise `T016D_analysis.md` with the literal five clauses and diagnostics above.

Stop after T016-D and report. Do not automatically start any follow-on model or experiment.
