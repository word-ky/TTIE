# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications and evidence remain preserved in Git history and `research_log/`.

---

# Research-lead review — T017-A accepted as a controlled reference-only negative

I reviewed Codex report `717bbf1546dfb5de56ee057f322a254d6a3c48ae`, PR #22, frozen source `5247690887078540fd5bfe74c90052efe0f4fb93`, evidence `bb45bd71dbaf512ce957f041cbcee1bcde6c9138`, `ttie/local_geometry.py`, focused tests, and the frozen evaluation receipts against the T017-A contract. PR #22 is accepted and squash-merged as `42bafee15964b75c9194e0b92d7f4bae23d049fc`.

The implementation matches the declared reference-only diagnostic. `local_choice()` receives only the five fixed `tau=0.05` cross MSE values, uses literal center-first/lower-coordinate tie handling, and never reads image IDs or condition labels. The 120 local decisions are persisted and hash-bound before family labels are attached. The input contract reproduces all 120 canonical Region2 values, all 1080 hard-candidate values, the accepted nine-hard oracle, and the full T016-A oracle before the new analysis. This is explicitly a development/reference landscape diagnostic, not a label-free test-time selector; no test label, clean target, reference metric, condition ID, or image ID enters any deployable path because no deployable path is claimed here.

The literal result is **4/5**. The local soft-cross rule is strong in aggregate: spatial MSE `0.03331791 = 0.95076×` Region2, `1.02317×` the nine-hard oracle, capturing `69.58%` of pooled hard-oracle headroom. Left/right improves to `0.98782×` Region2 and offset to `0.85782×`, capturing `93.26%` of offset headroom. But quadrants are `1.03303×` Region2 and fail the `1.01` safety clause. This is decisive because quadrants have essentially no boundary-placement headroom: `39/40` episodes have zero hard-oracle gain, yet the local rule makes 11 harmful moves and no beneficial quadrant move. The pooled gain therefore does **not** justify launching a derivative-supervised geometry optimizer.

The important unresolved mechanism is now narrower. T017-A chooses coordinates from a **soft (`tau=0.05`) local landscape** but deploys the corresponding **hard (`tau=0`) candidate**, and it composes x/y choices independently. The quadrant failure may therefore come from either (a) soft→hard renderer-transfer mismatch, where the soft landscape recommends a move that is harmful once hardened, or (b) x/y non-separability, where individually favorable axis moves combine into a poor 2-D soft candidate. Before training any geometry objective, distinguish these two mechanisms using only the already-rendered table.

The non-negotiable rule remains unchanged: **test-time adaptation/selection must never consume test labels, clean targets, condition IDs, masks/gains, annotations, image IDs as semantic shortcuts, source-only reference gradients/Jacobians, or evaluation metrics.** T017-B is again reference-only development diagnosis and must not be described as a deployable selector.

---

# OPEN one-hour task — T017-B: soft→hard local-geometry failure-attribution audit

**Expected work budget: about one hour. One question only: did T017-A fail mainly because the `tau=0.05` soft neighborhood is an unfaithful surrogate for the hard boundary, or because independent x/y local choices are non-separable even inside the soft landscape?**

## Hypothesis / objective

T017-A already shows substantial local reference signal, especially on offset, but unsafe quadrants. Do not try to rescue performance. Attribute the existing failures without training or changing any decision rule.

For each of the same 120 episodes, reproduce the frozen T017-A choice `(bx_local, by_local)` exactly, then compare that *same choice* in the soft and hard renderers:

- `S0 = L(0.5, 0.5, 0.05)` — canonical soft;
- `S1 = L(bx_local, by_local, 0.05)` — jointly selected soft candidate;
- `H0 = L(0.5, 0.5, 0)` — canonical hard Region2;
- `H1 = L(bx_local, by_local, 0)` — T017-A selected hard candidate.

Also compute the best of the nine already-rendered soft candidates at `tau=0.05`, `S*`, only as a diagnostic of x/y separability; it must not change the frozen T017-A choice.

Define, with exact arithmetic from the committed table:

- `soft_joint_gain = S0 - S1`;
- `hard_gain = H0 - H1`;
- `soft_separability_regret = S1 - S*`.

For every episode with `hard_gain < 0` (a harmful T017-A hard move), classify it exclusively as:

1. **transfer_flip** if `soft_joint_gain > 0`: the chosen move improves the soft renderer but becomes harmful after hardening;
2. **soft_interaction_failure** if `soft_joint_gain < 0`: independent axis choices already combine into a harmful 2-D soft move;
3. **zero/tie** if `soft_joint_gain == 0`.

Do not introduce tolerances, clipping, thresholds, alternative candidate choices, or post-hoc rescue rules.

## Fixed inputs/settings

Use only artifacts already merged from T016-A/T017-A on `main`:

- the accepted T016-A 120 × 27 `candidate_metrics.json` and its recorded hashes;
- the accepted T017-A frozen decisions/receipts from merge `42bafee15964b75c9194e0b92d7f4bae23d049fc`.

Verify before analysis:

- exactly 120 episodes × 27 candidates;
- exact candidate tuples `bx,by ∈ {0.4,0.5,0.6}`, `tau ∈ {0,0.05,0.10}`;
- exact reproduction of every frozen T017-A `(bx_local,by_local)` and decision hash;
- exact reproduction of T017-A hard selected MSEs and 4/5 clause vector.

CPU only. No image reads, clean-image files, CLIP, TTT, rerendering, model training, OOF/CV, source training, A6000, or new data.

## Predeclared interpretation / stop criteria

This is an attribution audit, not a qualification experiment. After assigning every harmful hard move to the three categories above, report the dominant mechanism globally and separately for quadrants.

- **Soft→hard transfer-dominant** only if `transfer_flip` is at least **2/3 of harmful hard moves** both overall and within quadrants.
- **Soft interaction/non-separability-dominant** only if `soft_interaction_failure` is at least **2/3 of harmful hard moves** both overall and within quadrants.
- Otherwise conclude **mixed/inconclusive**.

Additionally report `soft_separability_regret` distributions and the fraction of episodes where the independent T017-A soft choice equals the nine-soft oracle, but do not use these secondary statistics to override the fixed 2/3 attribution rule.

Interpret literally:

- transfer-dominant → the current `tau=0.05` derivative neighborhood is not a faithful surrogate for hard-boundary deployment; do **not** train a hard-deployment geometry objective from it in the next cycle;
- interaction-dominant → local signal exists but independent coordinate treatment is inadequate; do **not** infer that a scalar 1-D-per-axis derivative rule is sufficient;
- mixed/inconclusive → stop and preserve ambiguity; do not select a preferred mechanism from exploratory correlations.

In all cases, stop after T017-B. Do not automatically train a geometry objective or launch fresh evaluation.

## Required evidence

Commit a compact CPU-only audit plus focused tests/receipts proving:

- input/hash binding to the merged T016-A and T017-A artifacts;
- exact T017-A choice/hash/MSE/clause reproduction before attribution;
- no mutation of the 120 frozen choices;
- exact definitions of `S0,S1,H0,H1,S*` and the three exclusive harmful-move categories;
- counts and fractions overall, left/right, quadrants, and offset;
- soft/hard gain sign contingency table by family;
- `soft_separability_regret` mean/median/p95/max and zero count by family;
- fraction of independent choices equal to the nine-soft oracle, including oracle ties explicitly;
- all family labels used only for reporting after per-episode quantities are computed.

## Non-goals

No new selector, confidence gate, threshold, alternate `tau`, finer boundary grid, coupled optimizer, learned geometry head, derivative/Sobolev geometry training, soft-deployment method, annealing schedule, rerendering, image/CLIP reads, TTT, detector/meta/prompt/ViT3 work, fresh split, or GPU run.

Stop after T017-B and report the bounded attribution result.