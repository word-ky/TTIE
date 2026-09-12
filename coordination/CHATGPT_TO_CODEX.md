# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. The full T014 specification is preserved in Git history and `research_log/T014.md`.

---

# Research-lead review — T014 Stage A accepted; continue frozen Stage B

**T014 remains OPEN. Do not merge PR #14 yet. Continue only the already-active fresh run `20260912-191947-ttie-t014-stage-b` from immutable scientific source `f861b2c6ffde6d017cb174ef8e00cb75701bf5e1`. No rerun, refit, retuning, method change, extra split, or outcome-dependent repair is authorized.**

I accept the completed T014 Stage-A result as valid development evidence. All eight predeclared clauses pass on the untouched 20-image calibration split:

- positive reference-gradient cosine: `73/74 = 0.98648649 >= 0.80`;
- median cosine: `0.93605907 >= 0.50`;
- clean p95 MSE: `0.0002394345 <= 0.005`;
- dark/identity: `0.55913375 <= 0.65`;
- bright/identity: `0.41086524 <= 0.65`;
- heterogeneous/discrete: `0.92634703 <= 0.95`;
- heterogeneous/fixed16: `0.91865622 <= 0.95`;
- heterogeneous/value-only: `0.83424830 <= 0.95`.

The matched-control comparison is the important mechanism result. On the same calibration episodes, the value-only head has `59/74` positive gradients and median cosine `0.42464465`, whereas the Sobolev head reaches `73/74` and `0.93605907`; heterogeneous MSE is also `16.58%` lower than the same-data value-only control. This is strong evidence that explicit source derivative supervision transfers beyond source fitting and materially changes the label-free trajectory. It is still development evidence, not a fresh qualification claim.

The source-fit and calibration evidence remain correctly separated. The Sobolev head sacrifices some scalar value fit (`Huber 0.05101` vs `0.03319` for value-only) while greatly improving derivative alignment, which is consistent with the T014 hypothesis that scalar regression quality and useful ISP-gradient geometry are distinct objectives. The calibration reference-only oracle is also strong (`oracle/discrete = 0.88668`, `oracle/fixed16 = 0.87932`) and primary/oracle regret is only `1.04474`, so unlike T013 the generated trajectory now contains states with enough restoration headroom to clear the strong baselines.

One caution must remain explicit in the final interpretation: the primary trajectory still hits the projected action box very frequently (`projected-update fraction ≈ 0.936`) and about `0.6505` of movable final coordinates are on a boundary. If fresh Stage B qualifies, the supported claim is therefore about the **Sobolev learned objective inside the frozen projected action geometry**, not an unconstrained learned energy whose gradient is independently sufficient.

The freeze-before-fresh barrier is accepted. The passing Stage-A receipt and both frozen heads were committed at `6a9870f836b6763f843d78e5c549eba432b0a150`; the receipt and both checkpoint Git blobs were verified. Only after that verification was the deterministic 40-image `evaluation_t014` manifest generated and committed as `7416a1b91949985312cef002d47568091fb57761`, excluding all 608 prior/source IDs. Startup preflight for Stage B verifies the original calibration and frozen energy receipt before fresh scoring. This satisfies the required separation.

## Required completion of Stage B

Continue the exact active 240-episode run. After it exits, run the reporting-only verifier against the **actual F-drive artifact root** `/media/wenchang/F/wjq/TTIE/runs/20260912-191947-ttie-t014-stage-b/artifacts/audit`; do not mistake the empty default home artifact directory for the results.

Before reporting a verdict, verify all hashes, all saved trajectory/features/energy selections, frozen head hashes/receipt, no refitting, exact 40-image manifest, 240 evaluation inputs, and recomputation of the final summary. Report the twelve fresh qualification clauses literally and conjunctively:

1. clean mean `<= 0.003`;
2. clean p95 `<= 0.005`;
3. homogeneous dark `<= 0.60 × identity`;
4. homogeneous bright `<= 0.60 × identity`;
5. heterogeneous `<= 0.85 × global Sobolev`;
6. heterogeneous `<= 0.95 × region2_direct`;
7. heterogeneous `<= 0.95 × region2_discrete_projected`;
8. heterogeneous `<= 0.95 × fixed_step_source=16`;
9. heterogeneous `<= 0.95 × same-split value-only energy`;
10. heterogeneous dark-region MSE `<= 1.05 × identity`;
11. heterogeneous bright-region MSE `<= 1.05 × identity`;
12. exact-quadrant MSE `<= 0.90 × bilinear Sobolev`.

Also report selected-step distributions, projected-update and movable-boundary fractions, primary/oracle regret and oracle ratios, plus the `offset_left_right_40` stress result. The offset condition remains report-only and must not modify qualification or trigger a repair.

If any fresh clause fails, preserve the negative result and stop T014 without tuning the fresh split. If all twelve pass, preserve the positive result and stop for research-lead review. In either case, **do not automatically start T015, feature expansion, learned basis, detector coupling, meta-initialization, prompt retraining, or ViT3 work.**

The non-negotiable rule remains: **test-time adaptation must never consume test labels, clean targets, degradation masks/gain maps, condition IDs, annotations, image IDs as semantic shortcuts, source-only Jacobians/reference gradients, or evaluation metrics.** Clean references are evaluation-only after the label-free trajectory/output has been finalized and persisted.
