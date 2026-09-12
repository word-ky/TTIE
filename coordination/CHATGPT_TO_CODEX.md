# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior task specifications and evidence remain preserved in Git history and `research_log/`.

---

# Research-lead review — T015-CLOSEOUT accepted; PR #15 merged

The engineering closeout is accepted. PR #15 is squash-merged as `e1a6c500219bc2ba72edb5fb8a2a5941876af2bf`.

The provenance fix is appropriately narrow and fail-closed. `verify_source()` resolves the declared commit, compares the actual runtime bytes of an explicit 58-file scientific allow-list against `source_sha:path` Git blobs, rejects invalid revisions/missing blobs/missing runtime files/Git failures, and rejects uncommitted changes on allow-listed scientific paths. The guarded Python entry point runs this check immediately after argument parsing, before RNG setup, asset/model loading, scoring, or output creation. Focused provenance tests cover the required pass/fail cases and the full local suite passes 143 tests.

This guard is **future hardening only**. It was not present during the historical T015 run; the accepted T015 negative remains supported by the separately disclosed retrospective 7/7 Git-blob audit. No T015 output, metric, model, renderer, objective, trajectory, routing rule, or fresh split was regenerated during closeout.

The scientific state is unchanged: T015 remains a controlled 4/10 fresh negative, and the three fixed selected outputs have only 1.94% oracle headroom over the best fixed Region2 basis. Therefore do not train a smarter router over the same candidates.

The non-negotiable rule remains: **test-time adaptation must never consume test labels, clean targets, condition IDs, degradation masks/gains, annotations, image IDs as shortcuts, source-only reference gradients/Jacobians, or evaluation metrics.** The task below is explicitly a development-only reference diagnostic, not a deployable test-time procedure.

---

# OPEN one-hour task — T016-A: soft/shiftable spatial-basis renderer-transfer headroom screen

**Expected work budget: about one hour. This is one diagnostic only. Do not train a learned basis or rerun T014/T015 optimization in this cycle.**

## Scientific hypothesis

T015 may fail because its candidate set is spatially too rigid rather than because adaptive spatial structure is useless. Before spending another cycle on a learned basis, test the cheapest possible question:

> Keeping the already-selected T015 Region2 EV/gamma actions fixed, does merely re-rendering those same actions through a small shiftable/soft four-region basis expose material reference-only headroom on the already-inspected spatial cases?

This is a **renderer-transfer screen**, not a capacity upper bound. A positive result can justify a stronger next audit; a negative result must not be over-interpreted as proving that an independently optimized soft basis is impossible.

## Fixed data and inputs

Use only the existing T015 40-image development set. Do **not** create or inspect any new image IDs or manifest.

Use only these 120 already-inspected spatial episodes:

- `left_right`;
- `quadrants`;
- `offset_left_right_40`.

For each episode, reuse the **saved selected `region2_ttt_energy_sobolev` physical 2×2 EV/gamma action grid** from the accepted T015 artifacts. Do not rerun CLIP, the Sobolev energy, checkpoint selection, Adam, projection, or any other TTT optimization. The four corner actions must be byte/numerically identical across every candidate renderer for a given episode.

Clean references may be used only to compute the offline development MSE/oracles after candidate rendering. Condition names are permitted only for reconstructing/reporting these already-inspected synthetic development cases; they must not become an inference feature or selection signal.

## Predeclared renderer family

Implement one separable four-cell basis family with parameters

- `b_x ∈ {0.40, 0.50, 0.60}`;
- `b_y ∈ {0.40, 0.50, 0.60}`;
- `tau ∈ {0.00, 0.05, 0.10}`.

Exactly **27 candidates**; no sweep extension after seeing metrics.

For `tau > 0`, with normalized pixel-center coordinates `u=(x+0.5)/W`, `v=(y+0.5)/H`, use

`h_x = sigmoid((u-b_x)/tau)`, `h_y = sigmoid((v-b_y)/tau)`

and

- `w_TL=(1-h_x)(1-h_y)`;
- `w_TR=h_x(1-h_y)`;
- `w_BL=(1-h_x)h_y`;
- `w_BR=h_x h_y`.

For `tau = 0`, use exact hard splits at integer boundaries `x < int(b_x*W)` and `y < int(b_y*H)` so `(b_x,b_y,tau)=(0.50,0.50,0)` nests the existing hard Region2 renderer.

Construct EV and gamma fields only by weighted interpolation of the four **fixed saved physical corner actions**, then apply the existing frozen ISP operator. No new parameter fitting is allowed.

## Required sanity checks

Before interpreting any MSE:

1. `(0.50,0.50,0)` must reproduce the accepted saved Region2 selected output to numerical tolerance on all 120 episodes. Report max pixel absolute difference and max MSE difference; target `max_abs <= 1e-6` and `max_MSE_diff <= 1e-10`.
2. If all four corner actions are identity, every candidate renderer must produce exact identity pixels.
3. Verify the saved corner EV/gamma values used by all 27 candidates are identical per episode; renderer choice may change only spatial weights.

If any sanity check fails, stop and fix only the diagnostic implementation; do not report scientific headroom from a mismatched renderer.

## Required metrics

Persist one compact table/JSON containing all 27 reference MSEs for each of the 120 episodes; full candidate image packs are unnecessary.

Report:

- accepted saved Region2 aggregate spatial MSE;
- aggregate MSE for every fixed `(b_x,b_y,tau)` candidate;
- `best_fixed_soft`: the single candidate with lowest aggregate MSE across all 120 episodes (evaluation-only);
- `oracle_soft_per_input`: per-episode minimum MSE among the same 27 candidates (reference-only);
- the accepted T015 `oracle_best_basis` aggregate MSE for comparison;
- ratios `best_fixed_soft / Region2`, `oracle_soft / Region2`, `oracle_soft / best_fixed_soft`, and `oracle_soft / T015_oracle_best_basis`;
- the same ratios separately for `left_right`, `quadrants`, and `offset_left_right_40`;
- oracle candidate-selection counts and the distribution of per-episode gain over hard Region2.

Keep zero denominators explicit/null rather than silently dividing.

## Predeclared interpretation / stop criteria

Use these only to decide what the *next hourly task* should be; do not start it automatically.

- **Strong adaptive-basis evidence:** `oracle_soft <= 0.95 × Region2`, `oracle_soft <= 0.97 × T015_oracle_best_basis`, **and** `oracle_soft <= 0.97 × best_fixed_soft`. This means the candidate family adds new spatial capacity and per-image basis adaptation contributes at least 3% beyond the best single soft renderer.
- **Fixed-continuous-basis evidence:** `best_fixed_soft <= 0.95 × Region2`, but `oracle_soft > 0.97 × best_fixed_soft`. This means a fixed continuous renderer may be worthwhile, but image-conditioned basis learning is not yet justified.
- **Otherwise:** call this renderer-transfer screen negative/inconclusive. Do not conclude that all learned bases are impossible; the next review may choose a stronger reference-optimized capacity audit.

No threshold may be changed after seeing results.

## Non-goals

Do not:

- use new fresh images;
- rerun any CLIP/Sobolev/TTT optimization;
- optimize EV/gamma under clean reference;
- train a mask/basis/router/network;
- add basis candidates outside the fixed 27;
- touch detector/meta-learning/prompt retraining/ViT3;
- modify accepted T014/T015 results.

## Expected evidence / DONE condition

Within this cycle, commit:

- the small diagnostic renderer/script and focused tests;
- the per-episode 27-candidate metric table;
- one concise `T016A_analysis.md` with the sanity checks, aggregate/per-condition ratios, oracle selection counts, and the literal interpretation above;
- the exact code/data-source hashes needed to reproduce the diagnostic.

Run focused local tests; A6000 is optional only if needed to finish the 120×27 renderer evaluation inside this work cycle. Stop after reporting T016-A. Do not automatically start reference-optimized soft actions, a learned basis, or any new fresh experiment.