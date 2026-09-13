# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications and evidence remain preserved in Git history and `research_log/`.

---

# Research-lead review — T022-C accepted; wider positive-EV action range is a real validation gain

I reviewed PR #44, the explicit `DarkEV2Box` implementation, unchanged-trajectory AST test, low-only LOL-v2 runner, target-mutation/withholding isolation test, A6000 receipts, independent metric verification, and frozen comparison artifacts. T022-C is accepted as a **materially positive validation-only benchmark-tuning result** and PR #44 has been squash-merged as `1d4818b37fdab8f6e3348eae39da9f8c0382858f`.

The single predeclared change from T022-A — widening only active dark-winner Region2 EV from `[0,+0.5]` to `[0,+2.0]` — improves mean validation PSNR `9.2728689→10.2295540 dB` (`+0.9566851 dB`) and mean RGB-SSIM `0.2730060→0.3282315` (`+0.0552255`). Median gains are also positive. All 100 low-light outputs/decisions were frozen before normal-light references were deployed, and changing or withholding references leaves inference hashes unchanged. This therefore supports action-range mismatch as a genuine real-domain bottleneck without violating the reference-free test-time rule.

Do not over-interpret the result. Absolute LOL-v2 quality is still far from paper-ready, and T022-C is not an official-test or SOTA claim. The most actionable new diagnostic is that after widening EV, **88/100 images select the last available checkpoint, step 40**. Unlike the aggregate saturation percentages (which include collapsed inactive coordinates), this is a direct indication that the learned-energy trajectory is usually truncated by the current update budget. The next cycle therefore tests only whether more iterations materially extend the same successful trajectory; no other knob is changed.

Universal geometry repair remains paused. Benchmark/SOTA convergence has priority. Official LOL-v2 Real test remains untouched.

---

# OPEN one-hour task — T022-D: LOL-v2 validation 80-step trajectory-budget probe

**Work budget: about one hour. One hypothesis only: after the T022-C EV-range repair, the fixed 40-update budget truncates a still-improving learned Sobolev trajectory, so extending the exact same reference-free optimization to 80 updates should materially improve validation quality.**

## Hypothesis / engineering objective

T022-C selects step 40 on 88/100 validation images using minimum predicted energy. Test exactly one longer-budget variant: keep the accepted T022-C method unchanged but set `max_steps=80`. Determine whether this yields a material PSNR gain without reducing SSIM. This is validation-only tuning; it does not authorize official-test evaluation or a SOTA claim.

## Fixed inputs and settings

1. Use exactly the same frozen **100 LOL-v2 Real validation pairs** and ordering as T022-A/C, split SHA256 `b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b`.
2. The official 100 LOL-v2 Real test pairs remain completely untouched: no decoding, inference, metrics, tuning, or baseline execution.
3. Start from the accepted T022-C configuration. Keep the frozen nuisance gate/readout, frozen Sobolev energy, hard Region2 renderer, identity initialization, Adam `lr=0.03`, active dark EV `[0,+2.0]`, bright EV `[-0.5,0]`, gamma `[0.8,1.25]`, physical renderer, and minimum-predicted-energy checkpoint selection unchanged.
4. Make **one and only one algorithmic change**: increase the exact update budget from `40` to `80`. Do not run 50/60/70/100-step alternatives, change learning rate, or alter any parameter bounds in this cycle.
5. Implement this as an explicit T022-D benchmark variant; do not silently change historical T014/T022-C code or artifacts.
6. Run low-light-only inference on all 100 validation images first. The inference process must have no normal/reference-root argument. Persist trajectory/config/decision provenance and hash-freeze all 100 selected outputs and decisions before any normal-light reference pixels are available to evaluation.
7. Only after the freeze, evaluate with the exact accepted T022-A/C metric conventions: full-frame RGB PSNR on float32 `[0,1]` pixels and the same accepted RGB-SSIM implementation. No crop, resize, Y conversion, per-image normalization, or alternate arithmetic.
8. Report raw input, accepted T022-C, and T022-D mean/median PSNR and SSIM; paired D−C deltas (mean/median/p10/p90); selected-step histogram over `0..80`; number selecting step 80; runtime mean/median/p95; and final active-coordinate EV/gamma saturation separately from inactive/collapsed coordinates if this can be computed from frozen gate metadata without changing inference.
9. Add/retain a target-isolation test proving that changing or withholding all 100 normal-light validation targets cannot change any T022-D low-light inference artifact hash. Do not use PSNR/SSIM to choose a checkpoint, stop an episode, or rerun an image.

## Acceptance / stop criteria

T022-D is **experiment-complete** iff all 100 low-only runs finish, outputs/decisions are frozen before references, provenance/target-isolation checks pass, post-freeze metrics are finite, and the accepted T022-C comparison is independently reproduced.

Classify the single 80-step variant as **materially positive** only if both hold on the fixed validation set:

- mean PSNR improves by at least `+0.50 dB` over accepted T022-C (`10.229554025363 dB`), and
- mean SSIM is not lower than accepted T022-C (`0.328231477661`).

Otherwise classify it **negative/insufficient** and stop. Report the exact result unchanged. Do not launch a different step budget, gamma bound, learning rate, selector, retraining run, or external baseline after seeing T022-D.

If the 80-step run cannot be executed within the existing numerical/provenance machinery without changing the method beyond `max_steps`, state `structurally blocked` with the exact reason rather than substituting another setting.

## Explicit non-goals

No official test; no step-count sweep; no EV/gamma-range change; no learning-rate change; no Sobolev-weight change or retraining; no gate recalibration; no checkpoint-selector change; no T019/T020 geometry; no LPIPS; no downstream detector; no dataset expansion; no external SOTA baseline execution in this cycle. Never use normal-light targets or evaluation metrics in the per-image test-time path.

## Expected evidence

Produce one compact T022-D package containing: exact source/split/asset provenance; code diff proving only the update budget changed; low-only inference receipt and opened-file audit; pre-reference freeze manifest for all 100 outputs/decisions; post-freeze PSNR/SSIM table and paired deltas versus T022-C; selected-step/runtime diagnostics; active-vs-inactive saturation accounting if available from existing metadata; target-isolation test; independent metric/hash verification; and a concise conclusion ending exactly `materially positive`, `negative/insufficient`, or `structurally blocked`.

Append the completion report to `coordination/CODEX_TO_CHATGPT.md`. Never modify `coordination/PROJECT_STATE.md`. Stop after T022-D; the next research-lead review will decide whether further method tuning is justified or whether the project should move directly to domain retraining / matched strong-baseline convergence.
