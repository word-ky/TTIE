# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior task specifications and evidence remain preserved in Git history and `research_log/`.

---

# Research-lead review — T016-A accepted as a positive development-only capacity diagnostic

T016-A is accepted scientifically. The implementation matches the predeclared renderer-transfer screen: exactly 27 `(b_x,b_y,tau)` candidates are applied to the **same saved T015 selected Region2 physical EV/gamma corners**, with no CLIP/Sobolev rerun, no checkpoint reselection, no Adam/projection/action fitting, no new image IDs, and no learned basis. The canonical `(0.5,0.5,0)` renderer reproduces all 120 accepted Region2 outputs and MSEs exactly; all 3240 identity checks pass; all candidates preserve the same per-episode corner grid.

The result materially changes the spatial-basis diagnosis. The reference-only 27-renderer oracle reaches spatial-pool MSE `0.03250770`, which is `0.92764×` the accepted hard Region2 (`0.03504357`) and `0.94597×` the accepted T015 three-basis oracle (`0.03436452`). All three predeclared strong-headroom clauses pass. The best **single fixed** candidate is still canonical hard Region2, so this is not evidence for replacing Region2 with one globally fixed soft renderer.

The mechanism is more specific than “softness helps.” Oracle selections use `tau=0` in **113/120** episodes; 72 of those are shifted hard boundaries and 41 are canonical. Only 7/120 selections use `tau>0`. The largest gain is on `offset_left_right_40` (`0.84483×` Region2), while exact quadrants are essentially unchanged. The supported conclusion is therefore:

> **There is meaningful image-dependent boundary-placement headroom inside the existing four-corner action representation; sigmoid smoothing itself is not the demonstrated source of the gain.**

This remains a development-only, reference-only renderer-transfer result. It does not show that a label-free system can choose the boundary, does not show fresh generalization, and does not establish independently optimized soft-basis capacity. The non-negotiable rule remains: **test-time adaptation/selection must never use test labels, clean targets, condition IDs, degradation masks/gains, annotations, image IDs as shortcuts, source-only reference gradients/Jacobians, or evaluation metrics.**

PR #16 currently conflicts with `main` because Codex coordination/evidence commits advanced `main` while the implementation branch remained based on the earlier research-lead commit. Do not spend this cycle trying to preserve PR topology. For the task below, start from current `main` and port/reuse only the frozen T016-A scientific modules needed for scoring; do not change the accepted T016-A metrics or rerun T016-A.

---

# OPEN one-hour task — T016-B: frozen-Sobolev hard-boundary selection audit

**Expected work budget: about one hour. One question only: can the already-frozen T014 Sobolev energy select the useful shifted hard boundary without reference information?**

## Scientific hypothesis

T016-A shows that most renderer headroom comes from moving a hard vertical/horizontal split while keeping the four selected EV/gamma actions unchanged. Before training any boundary predictor or learned spatial basis, test the minimal deployable hypothesis:

> The existing frozen Sobolev restoration energy may already rank the nine shifted-hard renderers well enough to recover a meaningful fraction of the reference-only boundary-placement headroom.

This is a **selection audit**, not new training and not a new TTT trajectory.

## Fixed data and candidate family

Use only the same 120 already-inspected T015 spatial episodes (`left_right`, `quadrants`, `offset_left_right_40`). No new image IDs, no new manifest, no fresh claims.

For each episode reuse:

- the accepted T015 degraded input pixels from the persisted label-free `identity` output (do not reconstruct the input from a clean reference inside the scoring program);
- the accepted selected `region2_ttt_energy_sobolev` physical 2×2 EV/gamma corner grid;
- the frozen T006/T007 scorer/gate receipt;
- the **accepted frozen T014 Sobolev head, normalization, and exact 28-feature schema**.

Evaluate exactly the **nine hard** candidates

`b_x,b_y ∈ {0.40,0.50,0.60}`, `tau=0`,

using the T016-A hard renderer. Candidate order/tie order is lexicographic `b_x`, then `b_y`; `(0.5,0.5)` is the canonical Region2 candidate. Do not include `tau>0` in this cycle: T016-A already showed that 113/120 oracle choices are hard, so this task isolates boundary placement rather than smoothing.

The four EV/gamma corners are identical across all nine candidates. Do not rerun TTT, checkpoint selection, Adam, projection, or action fitting.

## Label-free selector

For each of the nine rendered candidate outputs, compute the frozen T014 Sobolev energy exactly as deployed in T014:

- the original episode's frozen gate constants remain fixed;
- candidate-dependent current CLIP exposure evidence is recomputed from the candidate pixels;
- the current physical EV/gamma corner values are the same saved corners for all candidates;
- no condition name, clean pixels, reference MSE, candidate oracle rank, image ID, mask/gain, or metadata may enter the 28 features or selector.

Select the candidate with the **lowest frozen Sobolev energy**, exact ties resolved by the fixed lexicographic candidate order. No score calibration, candidate-specific offset, threshold, temperature, normalization change, or learned router is allowed.

Batching the nine candidate CLIP forwards is allowed if numerically equivalent.

## Hard separation between selection and reference evaluation

Implement two phases/scripts or an equivalently auditable separation:

1. **Label-free scoring phase**: inputs are only persisted degraded pixels, saved corners, frozen gate/scorer/head assets. It writes and hashes the complete `120 × 9` table of candidate energies (and enough feature/hash evidence to reproduce them), selected candidate per episode, and source/asset hashes. This phase must not accept or read clean references or T016-A reference-MSE tables.
2. **Evaluation phase**: only after the selection artifact is finalized, read the already-accepted T016-A `candidate_metrics.json` to attach the corresponding reference MSEs and hard-boundary oracle diagnostics. Do not rerender or change selections after reference access.

Add a test proving that replacing reference/evaluation metadata cannot change candidate energies or selected boundaries.

## Required diagnostics

From the existing T016-A table first derive the **nine-hard-candidate oracle** using indices corresponding to `tau=0`; report its spatial/per-condition MSE and its ratio to the full 27-candidate oracle. This is diagnostic only.

Then report for the frozen-energy selector:

- spatial-pool selected MSE and ratios to canonical Region2, nine-hard oracle, full T016-A oracle, and accepted T015 three-basis oracle;
- the same for `left_right`, `quadrants`, and `offset_left_right_40`;
- selected boundary counts by condition;
- hard-oracle boundary counts and selector/oracle disagreement rate;
- winner–runner-up Sobolev-energy margin distribution;
- regret conditioned on selected boundary;
- Spearman rank correlation between the nine frozen-energy scores and the nine reference MSEs per episode, summarized over the 120 episodes (evaluation-only diagnostic; never used for selection).

Zero denominators/ties must remain explicit.

## Predeclared acceptance / stop criteria

Call this **label-free boundary-selection evidence** only if all five clauses hold:

1. spatial-pool selected MSE `<= 0.97 ×` canonical Region2;
2. spatial-pool selected MSE `<= 1.05 ×` nine-hard reference oracle;
3. `offset_left_right_40` selected MSE `<= 0.95 ×` canonical Region2;
4. `left_right` selected MSE `<= 1.01 ×` canonical Region2;
5. `quadrants` selected MSE `<= 1.01 ×` canonical Region2.

These thresholds are fixed now. Do not change them after seeing scores.

If all five pass, stop and report; the next hourly review will decide whether to integrate boundary placement into the TTT state or perform a fresh validation. If any clause fails, preserve the negative result and stop. **Do not** train a boundary predictor, recalibrate the Sobolev energy, extend its features, add condition cues, or start a learned basis in this cycle.

## Non-goals

Do not:

- use new/fresh images;
- modify or retrain T014 Sobolev energy;
- optimize EV/gamma or boundary parameters;
- use clean/reference information during selection;
- add soft candidates or a larger boundary grid;
- train any router/mask/basis network;
- touch detector/meta-learning/prompt retraining/ViT3;
- modify accepted T014/T015/T016-A results.

## Expected evidence / DONE condition

Within this cycle, commit:

- the minimal scoring/evaluation scripts and focused leakage/tie-order tests;
- the complete 120×9 frozen-energy score table and immutable selection artifact;
- a concise `T016B_analysis.md` with all five clauses and diagnostics above;
- exact Git/source/model/head/receipt hashes and provenance-guard evidence.

Run focused local tests and use the A6000 only as needed for the 1080 frozen CLIP/energy evaluations. Stop after T016-B reporting. Do not automatically begin the next research stage.
