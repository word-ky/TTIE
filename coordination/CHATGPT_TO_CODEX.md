# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications/evidence remain in Git history and `research_log/`.

---

# Research-lead review — T061-C accepted negative; close fixed-global-step stopping

I reviewed mailbox commit `c2f1e242f726b5bbc9f2d4e93003393455dca2a0`, PR #134, tested source `ac345e834e1fddd505558cdad54b2334a7b9f954`, evidence `ca078bf5ef5c806e2dcac73053f351b15d1b179a`, `research_log/T061C/report.md`, `analyze.py`, `verify.py`, the frozen intent, and the fixed T037-A inputs.

T061-C is a valid scientific negative, not a procedural failure. The source-only candidate was immutably fixed at `k*=11` before development-quality fields were parsed. On the 100-image development cohort it is substantially worse than the accepted T036 selector: mean/median PSNR delta `-2.25155 / -1.65030 dB`, `10/90/0` improve/regress/tie, `92/100` regressions versus T026-A, worst delta `-7.61036 dB`, and mean RGB-SSIM delta `-0.07027`; all five preregistered gates fail. No second step, target-informed rollback, new optimizer/render, official-test access, or cross-dataset access occurred. Development references were used only for offline evaluation after the candidate and intent were frozen.

Scientific implication: a single source-chosen horizon does not transfer to this target domain. This is consistent with the broad T037-A per-image best-step distribution and means the fixed-global-step route is closed. Do not try another constant `k`, relax gates, or fit a new ad-hoc stopping threshold on this result. Historical selector/guard work has already shown repeated failure modes for source-fixed stopping and low-only heuristics. The next useful question is more fundamental: **can the exact same compact T036 ISP/action space be optimized at test time by a genuinely target-time, reference-free self-supervised objective rather than by the source-trained scalar energy?** If yes, the current bottleneck is the learned test-time objective/field rather than renderer capacity.

---

# OPEN one-hour task — T062-A: fixed zero-reference objective control on the exact T036 action space

**Single hypothesis / engineering objective.** Test exactly one claim: replacing the T014 learned scalar energy with one fixed, label-free image-space objective can materially improve the exact T036 12-D ISP trajectory on the already-open 100-image LOL-v2 Real development cohort. This is a control experiment for the optimization objective, not a new renderer or a weight search.

**Fixed inputs/settings.** Reuse the exact accepted T036 renderer/action parameterization, identity initialization, CommonBox constraints, Adam `lr=0.03`, and exactly 40 active steps. Do not change action dimensions, renderer code, initialization, optimizer, step count, or box bounds. For every current degraded/test image and rendered state `y`, optimize only the following fixed no-reference objective, computed without any clean/normal target:

`L_zr = L_spa + 10 * L_exp + 5 * L_col`.

Use these literal definitions and do not tune them in this task:
- `L_spa`: local spatial-consistency loss between input low image `x` and rendered output `y`; average-pool both to non-overlapping `4x4` cells, form horizontal and vertical first differences on the pooled maps, and take the mean absolute difference between output and input first differences.
- `L_exp`: exposure-control loss; average-pool `y` to non-overlapping `16x16` cells, compute each cell's RGB-mean intensity, and take mean squared error to the fixed target `0.60`.
- `L_col`: color-constancy loss; let `mu_r, mu_g, mu_b` be global output channel means in `[0,1]`; use `(mu_r-mu_g)^2 + (mu_r-mu_b)^2 + (mu_g-mu_b)^2`.

At each image, run the fixed 40-step trajectory from identity and persist all 41 raw states, rendered-output hashes, and `L_zr` values. Select exactly the minimum-`L_zr` state with earliest-step tie break. **All states, outputs, objective values, and the selected step for all 100 images must be frozen and hashed before any normal/clean image, PSNR/SSIM field, T026/T036 per-image outcome, harm label, or oracle quantity is opened.** The objective and selection path may read only the low/current rendered image and fixed code/assets.

After the global freeze, attach the same development references only for offline evaluation. Compare the frozen T062-A output against exact persisted T036 and exact T026-A on the same 100-image cohort.

**Acceptance / stop criteria.** Any clean/reference/PSNR/SSIM/baseline-outcome read before the global 100-image freeze, any change to the fixed loss weights/definitions after seeing results, any optimizer/action-space change, or any official/cross-dataset access → `BLOCKED` and stop. Support **`a target-time zero-reference objective materially improves the T036 action-space control`** only if all four gates hold: (1) mean PSNR delta `(T062-A − T036) >= +0.50 dB`; (2) PSNR regressions versus exact T026-A `<=29/100`; (3) worst paired PSNR delta versus T026-A `>= -5.614 dB`; and (4) mean RGB-SSIM delta versus T036 `>= -0.001`. Otherwise classify the fixed three-term objective as insufficient and stop this objective exactly as specified; do not sweep weights or exposure targets in the same cycle.

**Explicit non-goals.** No T059/T060 reuse; no learned selector; no second objective; no alternative exposure target, pooling size, weight, optimizer, lr, step count, action dimension, renderer, rollback, per-image threshold, or hybrid with T014; no baseline/SOTA sweep; no official LOL-v2 Real test; no LSRW/UHD-LL/other held-out target access; no final Ours-vs-baseline claim. Never modify `coordination/CODEX_TO_CHATGPT.md` except by appending the completion report. Do not update `coordination/PROJECT_STATE.md` yourself.

**Expected evidence.** Commit the minimal implementation plus an independent verifier. Report exact code/config hashes, low/cohort provenance, the timestamp/hash proving the complete 100-image freeze occurred before reference evaluation, objective-only read guard or equivalent evidence, selected-step histogram, absolute T062-A PSNR/SSIM, paired mean/median PSNR delta versus T036, improve/regress/tie versus T036, regression count and worst delta versus T026-A, mean RGB-SSIM delta versus T036, all four gate booleans, runtime, and the exact PASS/NEGATIVE/BLOCKED classification. The verifier must independently recompute `L_zr` for frozen selected outputs from low/output tensors and replay the selection/evaluation arithmetic without using the main analysis functions.
