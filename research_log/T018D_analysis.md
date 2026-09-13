# T018-D — final all-development direction selector frozen

**DONE: all five engineering acceptance items pass.** Exactly two final CPU heads were trained once on all 120 accepted development rows with the unchanged T018-C recipe. This is an artifact freeze; no new MSE qualification or fresh cohort was run.

## Immutable artifact

- Source commit: `6c6eda3a2adb22a7921b787ef57dadae2aff0cfd`; branch `codex/T018D-final-direction-selector`.
- Pinned selector receipt SHA256: `db194f4caa897094655a36523fa074772d8cd7302ffda250a72e8b81cc2f9d94`.
- Frozen at `2026-09-13T04:23:51.384167+00:00`. Receipt: `research_log/T018D_run/selector_frozen.json`.
- Receipt binds nine source files, six input artifacts by commit/path/SHA256, the original feature schema and class order, all-development normalization, and eight final artifact files. The evidence commit binds the receipt itself.
- Head architecture and training loop are reused verbatim from accepted `ttie/direction_probe.py` (T018-C merge `b45bf9563e7c3f7e8aa27c923a35f1aabb95a5cb`). No model or objective changes.

| Final artifact | SHA256 |
|---|---|
| config.json | `7b516d0851ceff73c45051db8f1aeec64f7b47dc02946e3609e03652d4659291` |
| head_x.pt | `0efaab934acc7ca2988b427e04def6116e266ab02e0e87487aa07a75728b21b4` |
| head_y.pt | `212ce052f6c0409a7c02bd4fdcf56eb3105274b6ae4dbf2761ec4bd60c8e46b4` |
| normalization.json | `bb011113247239e6850bd3086e62609a9fe68e375234766ca44746cf62b2b305` |
| replay_features.json | `723c341802f529ea7d7b6320a470719a61e1958fad22c46de69f3296d9acf455` |
| replay_expected.json | `8f0005815c02c5088e3a0a4a9e0ab9e1812ccdba8dcf5a40d1d95f35565d89d6` |
| history_x.json | `1ee272ade1e95cf6ff1796a67f028ff3e1941776a91757c2032983a28aeb623f` |
| history_y.json | `7f04ef181da5f2413eb73fdebead2c63b187eb6c2b125c4755eadab7c73fcda5` |

## Fixed recipe and provenance

- Source features: T016-B commit `4062e01cb93de731c394015c5ac741d6c08e04d8`; training targets: T018-A merge `5ecf598763c499b2275994be53f9218a3757245c`; accepted T018-C config binds the identical score, schema and target bytes. Six exact inputs are listed in config/receipt. No new split or reference MSE table is opened.
- All 120 unique accepted episode keys in their original order. Axis inputs are 84-D `[f0, fminus-f0, fplus-f0]`; the five cross indices are `[4,1,7,3,5]`.
- Per head: 84→64→64→3, two SiLU layers, ordinary unweighted cross entropy, AdamW lr 1e-3 and weight decay 1e-4, batch 256, exactly 100 epochs, seed 7 reset, final epoch only. Population mean/std uses all and only the 120 development rows; std is clamped at 1e-12. Float64 statistics are stored as float32 buffers, unchanged from T018-C.
- Class order `[0.5,0.4,0.6]`; exact logit tie resolves center, lower, upper. Inference and training use one CPU thread for exact replay. No class weights, confidence, threshold, new features, or checkpoint selection.

## Reference-free API

```python
from ttie.direction_selector import load_selector
selector = load_selector("research_log/T018D_run", "db194f4caa897094655a36523fa074772d8cd7302ffda250a72e8b81cc2f9d94")
decisions = selector.predict(center, x_lower, x_upper, y_lower, y_upper)
```

Each argument is one 28-D vector or an N×28 batch. The function returns one record per row with x/y logits, classes, bx/by, score_index (the nine-hard index), and hard_index (the inherited 27-candidate table index, equal to 3×score_index). Normalization and both 84-D concatenations are internal. It has no target, reference, condition, family, image ID or oracle arguments. Loading requires the externally pinned receipt hash and verifies inference-code/head hashes; it opens no training artifact.

## Acceptance and verification

| Engineering acceptance item | Result |
|---|---|
| Exactly two heads, each fit once on the fixed 120 rows | PASS |
| All-development normalization recorded and reproduced exactly | PASS |
| Five-feature-only deterministic API; reference mutation/removal invariant | PASS |
| Source/input/schema/head/history/normalization hashes in pinned receipt | PASS |
| Independent replay without target/reference artifacts | PASS |

- Baseline: 8 tests passed. Final affected suite: 12 tests passed in 3.176 seconds. Compilation and real-input precheck passed. The new orchestration test mocks fitting, proving exactly two calls with all 120 features/targets without generating extra formal training runs.
- One formal train process and one separate save/reload replay process, both exit 0; exact commands/timestamps are in `T018D_commands.json`. The replay set is the already accepted 120 development crosses, fixed before fitting; it is a serialization check, not a performance estimate.
- Independent verification checks all source/input/final-file hashes and the exact accepted recipe. Its provenance phase hashes raw target bytes but never decodes target values. A separate subprocess runs in a temporary five-file bundle containing only the receipt, two heads, feature replay set and expected predictions. It imports no TTIE implementation and reconstructs both 84-D inputs, all-row normalization, network arithmetic, logits, class/tie rule and candidate indices. All 120 decisions match exactly. No target or reference artifact exists in that replay bundle.
- Replay and verification leave the pinned selector receipt and weights unchanged. No retraining, reference MSE, accuracy selection, or fresh qualification.

## Failures, deviations and stop

- Before formal fitting, a mocked orchestration test exposed last-bit replay differences when directly constructed and loaded selectors used different CPU thread counts. Moved the existing single-thread setting to shared selector initialization. Exact replay then passed; original failing test log preserved. No training recipe or model change.
- First source push failed with a transient GitHub connection timeout. Retried successfully. The locally committed source had been frozen before the single formal fit; source and artifacts were unchanged by the network retry.
- No formal training/replay/verifier failures. No scientific recipe deviations. Unrelated image/CLIP/GPU suites were not run because this task only adds a CPU artifact/inference adapter to the existing tested head.
- User requested GPU preference after these two small CPU heads had already trained. For later substantive training/inference, prefer the A6000 where practical; carry this preference to research planning. No repeat fit was needed in this cycle.
- Stop after T018-D. Await research-lead review; no T018-E, new fresh IDs, or qualification starts automatically.
