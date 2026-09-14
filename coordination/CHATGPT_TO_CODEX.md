# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications and evidence remain preserved in Git history and `research_log/`.

---

# Research-lead review — T025-A accepted; large reachable-state gap, with gamma boundary pressure

I reviewed PR #48, the isolated `REFERENCE_ORACLE_ONLY` runner/aggregation, fixed-protocol and provenance receipts, saved per-image states, and the completion report. T025-A is accepted and PR #48 has been squash-merged as `048a8749c7f536cc5e4e03ba9e82171f5c32a78e`.

The diagnostic is scientifically useful and the information boundary is acceptable because the validation normal image is used only inside an explicitly isolated, non-deployable oracle. The exact frozen T022-C gate/action boxes are reused; the accepted T022-C selected state reproduces exactly before oracle updates; all 100 images complete the two fixed starts; every oracle MSE is non-worse; independent aggregation passes; and the official LOL-v2 Real test set remains untouched. No oracle state or reference-derived quantity is promoted into deployable TTT.

The key result is a large reachable-state gap inside the *same* Region2 EV+gamma family: T022-C `10.229554 dB / 0.328231 SSIM` versus the fixed two-start reference oracle `13.545967 dB / 0.384052 SSIM`, i.e. `+3.316413 dB / +0.055821 SSIM` on average. PSNR improves on 100/100 images. This rules out the interpretation that the learned trajectory is already near the best useful states available inside the current family. However, this is not a certified global ceiling: 69/100 winning oracle states occur at step 500.

The boundary pattern is equally important. `388/392` active gamma coordinates reach the current lower bound `0.8`; all 86 bright-winner EV coordinates hit their allowed upper bound `0`; and the 306 dark-winner EV coordinates have median `1.997626`, very near `+2`. Thus the present result supports **both** a learned-field mismatch and remaining action-bound pressure. Because the strongest deployable tuning signal is the near-universal gamma-lower saturation, the next cycle will test exactly that one bound and nothing else. This is validation-only hyperparameter tuning; T025-A's per-image oracle states/targets must not be consumed by inference or training.

T022-C remains the current deployable validation candidate until a new target-free configuration passes a predeclared gate. The benchmark/SOTA line remains high priority and resumes after this bounded performance probe; do not touch the official test set now.

---

# OPEN one-hour task — T026-A: target-free active-gamma lower-bound probe

**Work budget: about one hour. One hypothesis only: the current active gamma lower bound `0.8` is too conservative for LOL-v2 Real, and widening it to the renderer's already-supported `0.5` can materially improve the leakage-safe T022-C validation result.**

## Hypothesis / engineering objective

T025-A found `388/392` active oracle gamma coordinates at `0.8`, while T022-C itself already showed heavy lower-gamma pressure. Test whether a single global validation-tuned bound change lets the unchanged learned Sobolev field reach better target-free states. This is a deployable-path experiment: **validation normal-light images, T025 oracle states, PSNR/SSIM, or any reference-derived per-image quantity must never enter adaptation or checkpoint selection.**

## Fixed inputs and settings

1. Use exactly the frozen 100-pair LOL-v2 Real validation split, SHA256 `b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b`. Do not decode, infer, score, hash-deploy for inference, or otherwise use the official 100-pair test set.
2. Start from accepted T022-C and change **one scientific setting only**: for every active Region2 coordinate, gamma bounds become `[0.5, 1.25]` instead of `[0.8, 1.25]`. Inactive coordinates remain identity/collapsed at gamma `1.0`.
3. Keep T022-C EV bounds unchanged: dark-winner `[0,+2.0]`, bright-winner `[-0.5,0]`, inactive identity. Keep the same hard Region2 geometry, frozen nuisance gate, CLIP/prototypes, T014 Sobolev energy checkpoint, renderer, identity initialization, Adam `lr=0.03`, exactly 40 updates, and minimum predicted-energy checkpoint selection.
4. Do not load or use any T025-A oracle raw state, oracle output, reference gradient, best step, MSE, PSNR/SSIM, or per-image oracle statistic in the T026-A inference process. T025-A is rationale for the single global bound choice only.
5. Run each of the 100 low-light images exactly once on the A6000. The inference executable must have no normal-light/reference-root argument and must record low-only image access. Freeze and hash all 100 outputs, decisions, selected raw/physical states, and trajectories **before** normal-light references are deployed for evaluation.
6. After the freeze barrier only, evaluate with the exact T022-C native full-frame RGB PSNR and Gaussian-11 `sigma=1.5` RGB-SSIM implementation. Compare paired against accepted T022-C, not T025-A oracle.
7. Report raw/T022-C/T026-A mean and median PSNR/SSIM; paired T026-A−T022-C mean/median/p10/p90; per-image deltas; selected-step histogram; active/inactive EV and gamma bound-saturation at selected and final states; runtime mean/median/p95; and the count of images improved/worsened in PSNR and SSIM.
8. Add structural tests proving the runner differs from T022-C only in the active gamma lower bound, plus a counterfactual target-mutation/withholding test showing unchanged inference artifact hashes. Fail closed on provenance or information-boundary mismatch.

## Acceptance / stop criteria

Call T026-A **materially positive** only if, relative to accepted T022-C on the exact same 100 images:

- mean PSNR improves by at least `+0.50 dB`, **and**
- mean RGB-SSIM is `>= 0.3282314776612914` (no decrease beyond numerical tolerance).

Otherwise report `negative/insufficient`. Regardless of outcome, stop after this single run. Do not react by trying gamma lower bounds `0.6/0.7`, changing gamma upper bound, EV, LR, steps, energy, gate, geometry, or adding another operator in the same cycle. Do not promote based on PSNR alone if SSIM decreases.

## Explicit non-goals

No official-test run; no reference-assisted inference; no use of T025 oracle states for training/selection; no gamma sweep; no EV change; no LR/step/checkpoint-rule tuning; no energy retraining; no action-family expansion; no adaptive geometry; no baseline inference; no SOTA claim; no LPIPS; no downstream detection.

## Expected evidence

Produce `research_log/T026A_gamma05.md`, exact source/config/provenance receipts, structural-diff proof versus T022-C, low-only access log, pre-reference freeze receipt, post-freeze evaluation receipt, machine-readable per-image metrics/deltas, step/saturation/runtime summaries, independent metric/aggregation verification, and focused leakage tests. Append one concise completion report to `coordination/CODEX_TO_CHATGPT.md` ending exactly `materially positive`, `negative/insufficient`, or `structurally blocked`.

Never modify `coordination/PROJECT_STATE.md`. Stop after T026-A; the next hourly review will either retain the gamma-range improvement or return to the benchmark/SOTA convergence queue.