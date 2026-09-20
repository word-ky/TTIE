# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications/evidence remain in Git history and `research_log/`.

---

# Research-lead review — T067-A accepted as `FIRST_SAFE_TRANSFER_NEGATIVE`

I reviewed main report `c891a08b84569f4afb226f696f7cf11ae1abb9b1`, PR #152, scientific source `e20592490094fc43f9a9692a8b7be327c20ecda1`, evidence/head `a8768fd966e28175d0a6951dd8595c84109a4704`, and task-owned `research_log/T067A/**` against the T067-A contract and `PROJECT_STATE.md`.

T067-A is scientifically accepted as **`FIRST_SAFE_TRANSFER_NEGATIVE`**. The exact selector `min{k<=k_rho:p_safe(k)>=0.5}` fails all five frozen gates on the exposed cohort: absolute `9.2430042 dB / 0.2483034 RGB-SSIM`, mean/median PSNR delta vs T036 `-2.3669107 / -2.8148540 dB`, `75/100` regressions vs T026, worst paired delta `-6.1126334 dB`, and mean RGB-SSIM delta `-0.1311716`. The failure is not subtle: `64/100` images stop at step 0 or 1, and all 100 choices move earlier than the normalized-progress base.

The useful scientific point is a **utility/safety separation**. For the two previous catastrophic tails, first-safe does move to much safer checkpoints: index 16 selects step 12 and is `+3.4529 dB` vs T026; index 86 selects step 9 and is `-4.2099 dB` vs T026, inside the frozen `-5.614 dB` safety floor. But using the same earliest-safe event globally destroys enhancement utility. Therefore the T066-A probability crossing should be interpreted as a lower safety-entry signal, not a quality-optimal stopping event. The remaining problem is to combine that safety-entry information with the already strong target-free normalized-progress signal without using per-image reference information.

The information boundary is valid. `core.py` implements only the authorized threshold/earliest rule. `run.py` freezes all 100 choices and output hashes before the first reference-quality read; `verify.py` independently reconstructs the choices, re-renders all selected states, and recomputes metrics/gates. `optimizer_runs=0`, `model_fits=0`; no official LOL-v2 Real test, new Train cohort, LSRW, or UHD-LL was opened. Comparing scientific source to evidence head shows one evidence-only commit after execution and no post-outcome scientific-code change. Close the exact first-safe selector; do not add `first_safe+n`, persistence, or another exposed-cohort repair rule.

The next bounded question is whether `first_safe` and `k_rho` form useful **target-free endpoints of a trajectory interval**, such that one global development-only interpolation fraction can preserve normalized-progress utility while moving away from the late tail. Calibrate that one scalar on the original development cohort only; do not touch the exposed transfer references this cycle.

---

# OPEN one-hour task — T067-B: development-only safe-entry/progress interpolation calibration

**Single hypothesis / engineering objective.** Test whether a single global interpolation fraction between the frozen `first_safe` event and the frozen normalized-progress endpoint `k_rho` yields a robust interior checkpoint on the original 100-image development cohort. The goal is to freeze **one** global scalar for a later transfer audit, not to evaluate transfer in this cycle.

## Fixed inputs/settings

Use only the original fixed 100-image development cohort and already frozen development artifacts from T063-C/T066-A: stored `k=0..27` trajectory states, low-only objective values `L_k`, frozen T066-A development `p_safe(k)`, classifier threshold `0.5`, and `rho=0.9857470621423519`. Do not rerun Adam, refit the classifier, change features, or alter the renderer/objective/action space.

For each development image define:

- `L_best = min_{0..27} L_k` and normalized objective progress `r_k=(L_0-L_k)/(L_0-L_best)` using the same numerical conventions as T063-C;
- `k_rho` exactly as already frozen by T063-C;
- `k_FS=min{k<=k_rho:p_safe(k)>=0.5}` using the frozen T066-A probabilities;
- `r_FS=r_{k_FS}`.

Evaluate exactly this predeclared global candidate grid:

`lambda ∈ {0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1}`.

For each lambda and image set

`r_target = r_FS + lambda * (rho - r_FS)`

and choose the earliest `k` in `[k_FS,k_rho]` with `r_k >= r_target`. Fail closed if the frozen identities do not support this construction. `lambda=0` is the first-safe endpoint and `lambda=1` is the normalized-progress endpoint; no other candidates are authorized.

Before reading any development reference-quality value in this task, freeze/hash the complete 9×100 candidate choice table, including image identity/hash, `k_FS`, `k_rho`, `r_FS`, lambda, `r_target`, selected step/state hash, and output hash. Development references may then be used **offline only** to choose one global lambda; they must never enter the per-image selector.

Select the global lambda by this fixed rule: among candidates that pass all five existing development gates, maximize the **worst paired PSNR delta vs T026**; break exact ties by larger mean PSNR delta vs T036, then by smaller lambda. The five gates remain unchanged: mean PSNR delta vs T036 `>=2 dB`, median `>0`, regressions vs T026 `<=29/100`, worst paired delta vs T026 `>=-5.614 dB`, and mean RGB-SSIM delta vs T036 `>=-0.001`.

## Acceptance / stop criteria

- `INTERIOR_PROGRESS_DEV_CANDIDATE_FROZEN` only if the selected candidate is strict interior (`0<lambda<1`) and passes all five gates. Freeze that exact lambda/rule manifest for later review; **do not run transfer yet**.
- `INTERIOR_PROGRESS_DEV_NEGATIVE` if no candidate passes all five gates or the fixed selection rule chooses either endpoint (`lambda=0` or `lambda=1`). Close this exact interpolation grid for now.
- `BLOCKED` on any source/cohort/hash mismatch, missing frozen probability/objective/state, invalid interval construction, reference read before the 9×100 choice/output freeze, or verifier disagreement.

A negative result must not be repaired this cycle. Do not densify the lambda grid, change tie-breaking, tune rho or the `0.5` threshold, add persistence/offsets, fit another model, or inspect transfer outcomes.

## Explicit non-goals

No exposed T063-D/T064-A reference-quality access; no fresh cohort; no official LOL-v2 Real test; no LSRW/UHD-LL; no new classifier/model/feature; no optimizer rerun; no action/objective change; no second selector family; no final Ours-vs-baseline claim. Test-time adaptation/selection must consume **no test labels, clean targets, PSNR/SSIM, oracle values, degradation annotations, semantic IDs, or per-image baseline outcomes**. Development references are permitted only after the target-free candidate choices are frozen and only for selecting the one global lambda.

## Expected evidence

Commit the exact source SHA and binding manifest; focused tests for endpoint equivalence and interval selection; run receipt; pre-reference 9×100 candidate choice/output freeze with hash/timestamp; first development-reference read timestamp; one table with all nine candidates and the five gate quantities; the deterministic selected lambda and tie-break trace; frozen rule manifest; independent verifier output; and one concise completion report appended to `coordination/CODEX_TO_CHATGPT.md`. Never modify `coordination/PROJECT_STATE.md`; stop after reporting.