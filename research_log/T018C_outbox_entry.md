

---

## T018-C — DONE — 2026-09-13T03:10:57.123802+00:00

**Fixed grouped-OOF direct hard-axis direction probe: 5/5, development-level feature sufficiency supported.** PR [#27](https://github.com/word-ky/TTIE/pull/27) is ready for research review and remains unmerged.

- Source commit: `8fc63deeb825c87257c46546bc34e2ab3b4bffd8`.
- Evidence commit: `68f4333bf582258e3daf27d4f7d98ba482693d8b`.
- Branch: `codex/T018C-direct-direction-oof`.
- Implementation: `ttie/direction_probe.py`, `ttie/direction_probe_run.py`; tests: `tests/test_direction_probe.py`; full report, independent verifier and artifacts: `research_log/T018C_analysis.md`, `T018C_verify.py`, `T018C_verification.json`, `T018C_run/`.

### Exact bounded execution

Reused the accepted T016-B saved 28-D features and T018-A training-fold choices. Original T016-C folds copied verbatim from `433683eccad24dc763450be6a546072a72e0910b:research_log/T016C_run/folds.json`, SHA256 `8fdb03cdac63af5d6e58b557c96679bc15c1e2e921e525916eb9d21ad22cd2c1` (also byte-identical to T016-F). Each of five folds uses 32/8 disjoint image IDs and 96/24 episodes. Exactly ten independent CPU heads; input `[f0, fminus-f0, fplus-f0]`, 84 dimensions, prescribed 84→64→64→3 SiLU architecture, ordinary CE, AdamW 1e-3 / 1e-4, seed 7, batch 256, 100 epochs, final epoch only. Train-only population-standard-deviation normalization clamped at 1e-12; float64 statistics stored as float32 buffers. Fixed class order `[0.5,0.4,0.6]`.

Before fitting, checked unique episode keys, target ordinals/coordinates, saved schema/candidate order, original fold partitions and accepted row-order provenance. Each fold fit accesses targets only for its training rows. Full reference-bearing identity/corner and actual image-ID-to-row checks occur after OOF freezing and before metrics; this keeps reference tables outside fitting. All ten heads, histories, train-only normalization and held-out logits/classes are saved. All 120 decisions frozen at `2026-09-13T03:04:44.810894+00:00` before the separate evaluator loads held-out diagnostics/reference metrics. Decision SHA256 `03a97e0c3fe34925c4de0aa878393766cc7203dab8315128ea49c14568c22893` remains unchanged.

Commands: `python -m ttie.direction_probe_run train --source-sha 8fc63deeb825c87257c46546bc34e2ab3b4bffd8 --output research_log/T018C_run`; separate `python -m ttie.direction_probe_run evaluate --output research_log/T018C_run`; independent `python research_log/T018C_verify.py`. Exact local commands/timestamps and exit codes in `T018C_commands.json`. One formal training run and one evaluation, both exit 0. No GPU experiment, image reads, rerendering or feature/model extraction rerun.

### Literal result

Vector `[true,true,true,true,true]`, **5/5**, no tolerance/fallback:

| Clause | Observed | Required |
|---|---:|---:|
| Pooled selected/H0 | 0.9463400050770442 | <=0.97 |
| Pooled selected/H* | 1.0184123344198242 | <=1.03 |
| Offset selected/H0 | 0.874913447259421 | <=0.95 |
| Left/right selected/H0 | 0.9837094741065749 | <=1.01 |
| Quadrants selected/H0 | 1.0 | <=1.01 |

Pooled selected MSE `0.03316313575875635`, H0 `0.03504357374816512`, H* `0.032563564518932255`. X/Y/joint target agreement `105/120`, `98/120`, `87/120`; movements no-move/x-only/y-only/both `52/2/39/27`; beneficial/equal/harmful `57/52/11`. All 11 harmful cases occur in left/right. Quadrants remains center on all 40 cases. Full group MSE, confusion matrices, agreement, movements and harmful examples are included in the report and summary JSON.

### Tests, failures, deviations and next step

Eight focused tests pass. Independent implementation reproduces all ten saved-head predictions exactly, including feature construction and normalization, verifies exact fold bytes and strict actual image-group exclusion, hashes/freeze order and all 120 reference calculations plus group diagnostics. An initial verifier-only assertion compared the historical path/hash dictionary with a newer dictionary containing an additional commit field. Corrected to compare the original fields while retaining independent Git-blob commit/hash checks; first failed log preserved. No changed training source, model, decision, metric or experimental rerun.

No scientific recipe change. Explicit normalization convention and post-freeze timing of reference-bearing identity checks are documented above. This supports development-level local direction information in the frozen representation; it is not fresh qualification or universal per-episode improvement. **Stop after T018-C.** Await research review and an explicitly issued next task; no all-ID final model, confidence gate, class weighting, feature redesign, larger model or fresh images in this cycle. Recovery artifacts are being mirrored to the existing home and F server project roots; final delivery receipt will be added to this PR. Existing 15-minute heartbeat continues to watch the research inbox without repeating this completed task.
