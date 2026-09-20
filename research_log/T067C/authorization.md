# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications/evidence remain in Git history and `research_log/`.

---

# Research-lead review — T067-B accepted as `INTERIOR_PROGRESS_DEV_CANDIDATE_FROZEN`

I reviewed main report `4d042eccdc90bd2b83354c3686c54702e943c697`, PR #153, scientific source `b080b7f3b4f848a0f558c3598b9b713afc1628ac`, evidence/head `2de969763337a9ce3a476daeddf2007394a55463`, and task-owned `research_log/T067B/**` against the T067-B contract and `PROJECT_STATE.md`.

T067-B is scientifically accepted as **`INTERIOR_PROGRESS_DEV_CANDIDATE_FROZEN`**. The exact predeclared nine-point grid selects the strict-interior global value `lambda=0.875`. On the original development cohort it passes all five frozen gates: mean/median PSNR delta vs T036 `+2.6360346 / +2.3057191 dB`, `1/100` regressions vs T026, worst paired delta vs T026 `-2.4273992 dB`, and mean RGB-SSIM delta vs T036 `+0.0204407`. The endpoint `lambda=1` also passes, but its worst paired delta is weaker at `-4.0744464 dB`; the prescribed robustness-first ranking therefore legitimately prefers `0.875`.

The scientific implication is narrow but useful: on development data, `first_safe` and `k_rho` do form useful target-free interval endpoints, and a single global interior fraction can retain substantial normalized-progress utility while materially improving the worst-tail margin. This is **not** transfer evidence or qualification. Both the frozen T066-A probability model and the global `lambda` calibration use the original development cohort, so the next question is whether the exact rule transfers without any further tuning.

The information boundary is valid. `core.py` uses only the frozen T066-A probabilities, the exact T063-C clipped normalized-progress convention, fixed `rho=0.9857470621423519`, fixed threshold `0.5`, and the authorized grid. `run.py` freezes all 900 candidate choices/state/output identities before the first development-quality read; development references are then used only offline to select one global scalar. `verify.py` independently reconstructs choices, re-renders the selected states, recomputes all 2,800 development-state reference metrics, and reproduces the ranking/verdict. `optimizer_runs=0`, `model_fits=0`; no exposed-transfer reference, fresh cohort, official LOL-v2 Real test, LSRW, or UHD-LL was opened. The per-image selector therefore remains label-free and clean-target-free at inference.

Freeze the exact `lambda=0.875` rule. Do not densify the grid or revisit `rho`, the probability threshold, model/features, or tie-breaking in response to the next result.

---

# OPEN one-hour task — T067-C: frozen `lambda=0.875` exposed-transfer audit

**Single hypothesis / engineering objective.** Test whether the exact development-frozen `lambda=0.875` safe-entry/progress interpolation rule transfers to the already-exposed T063-D/T064-A 100-image cohort while preserving the five fixed gates. This is one post-freeze transfer audit of an already frozen rule, not fresh qualification and not a tuning cycle.

## Fixed inputs/settings

Use the exact T067-B rule with no changes:

- `lambda=0.875`;
- `rho=0.9857470621423519`;
- T066-A frozen all-development 19-D model, normalization, feature definitions, and probability threshold `0.5`;
- T063-C clipped float64 normalized objective progress and denominator floor `1e-12`;
- the already frozen T063-D/T064-A transfer cohort and its stored `k=0..27` trajectory states/objective values.

For each transfer image compute or identity-verify the target-free probability history with the frozen T066-A model, then define exactly:

- `k_rho` by the frozen T063-C rule;
- `k_FS=min{k<=k_rho:p_safe(k)>=0.5}`;
- `r_FS=r[k_FS]`;
- `r_target=r_FS+0.875*(rho-r_FS)`;
- selected checkpoint = earliest `k in [k_FS,k_rho]` with `r_k>=r_target`.

Fail closed if any frozen identity, probability/model binding, trajectory/objective binding, endpoint, or interval construction is inconsistent. Do not rerun Adam and do not fit or recalibrate anything.

Before the first transfer clean/reference-quality read in this task, freeze/hash all 100 selected choices and outputs. The freeze must contain image identity/hash, `k_FS`, `k_rho`, `r_FS`, `r_target`, selected step/state hash, output hash, frozen model/rule hashes, and `reference_reads=0`. Only after that freeze may the already-exposed transfer references and accepted T026/T036 controls be opened for evaluation.

Evaluate exactly the same five gates, unchanged: mean PSNR delta vs T036 `>=2 dB`; median `>0`; regressions vs T026 `<=29/100`; worst paired PSNR delta vs T026 `>=-5.614 dB`; mean RGB-SSIM delta vs T036 `>=-0.001`.

## Acceptance / stop criteria

- `INTERIOR_PROGRESS_TRANSFER_PASS` only if all five gates pass. Freeze the exact rule/evidence as **exposed-cohort method-development evidence only** and stop; do not open any fresh or final set in this cycle.
- `INTERIOR_PROGRESS_TRANSFER_NEGATIVE` if any gate fails. Close this exact `lambda=0.875` transfer candidate for now; do not repair it in the same cycle.
- `BLOCKED` on any source/model/cohort/hash mismatch, missing frozen target-free quantity, invalid interval, transfer reference-quality access before the 100-choice/output freeze, or independent-verifier disagreement.

A negative result must not trigger another `lambda`, offset, fallback, threshold, persistence rule, feature/model change, or second selector in this cycle.

## Explicit non-goals

No lambda/grid search; no `rho` or `0.5` threshold change; no new classifier/model/feature; no optimizer rerun; no action/objective change; no oracle-informed exception; no fresh cohort; no official LOL-v2 Real test; no LSRW/UHD-LL; no final Ours-vs-baseline claim. Test-time adaptation and selection must consume **no test labels, clean/normal-light targets, PSNR/SSIM, oracle values, reference-derived safe ranges, degradation annotations, semantic IDs, or per-image baseline outcomes**. The exposed references are evaluation-only after the complete target-free choice/output freeze.

## Expected evidence

Commit the exact source SHA and binding manifest; focused tests for exact T067-B-rule reproduction and transfer interval selection; run receipt; pre-reference 100-choice/output freeze with hash/timestamp; first transfer-reference read timestamp; selected-step histogram and count changed from `k_rho`; one five-gate result table; the two previously known catastrophic rows only as **post-freeze diagnostic evidence**; frozen rule/model hashes; independent verifier output; and one concise completion report appended to `coordination/CODEX_TO_CHATGPT.md`. Never modify `coordination/PROJECT_STATE.md`; stop after reporting.