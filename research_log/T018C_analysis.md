# T018-C — grouped OOF direct hard-axis direction probe

**Result: 5/5, direct_direction_probe_viable (development only).** The fixed frozen 28-D candidate representation contains development-level local hard-geometry information under the prescribed direction-aligned supervision. No final all-ID model or fresh qualification was run.

## Bound implementation and execution

- Frozen source: `8fc63deeb825c87257c46546bc34e2ab3b4bffd8`; branch `codex/T018C-direct-direction-oof`.
- Exact original five grouped folds: T016-C commit `433683eccad24dc763450be6a546072a72e0910b`, `research_log/T016C_run/folds.json`, SHA256 `8fdb03cdac63af5d6e58b557c96679bc15c1e2e921e525916eb9d21ad22cd2c1`. T016-F has identical bytes.
- 120 development episodes; 40 image IDs; 96 training / 24 held-out episodes and 32 / 8 disjoint IDs per fold. Five x/y pairs, ten CPU heads total.
- Only input: `concat(f0, fminus-f0, fplus-f0)`, 84 float32 dimensions. X cross indices 4/1/7; y cross 4/3/5 in the nine-hard order. No reference, family, image identity, or coordinate appended as a feature.
- Exactly `84 -> 64 -> 64 -> 3`, two SiLU hidden activations, ordinary unweighted cross entropy, AdamW 1e-3 / weight decay 1e-4, batch 256, 100 epochs, seed 7 reset per head. Final epoch only. No sweep or second-stage calibration.
- Normalization: training-only per-dimension mean and population standard deviation, standard deviation clamped at 1e-12. Statistics computed in float64 and stored as float32 buffers. Exact ties resolve center 0.5, lower 0.4, upper 0.6.
- Baseline CPU head lifecycle and existing T018-B provenance/evaluator reused. No images read, rendering, feature recomputation, GPU work, or all-ID training.

## Information boundary and provenance

- Before fitting: exact saved-score/schema/candidate binding, 120 unique opaque episode keys, target ordinal/coordinate alignment, accepted row-order provenance, and verbatim grouped folds. Seven training-stage artifacts are bound by commit/path/hash in run/config.json.
- Training preparation loads the accepted target artifact; each fold fitting function accesses bx/by only at its training row indices. The fitting API never passes held-out labels to a head; a guarded-access test and a held-out-target mutation test cover that boundary. Metadata validation is separate from model input.
- Actual cross-artifact episode/corner and image-ID-to-row matches are rechecked after OOF freezing and before metrics, because the reference table also contains MSE/family information. This timing is explicit; it is not claimed that reference-bearing tables were opened before fitting.
- All ten saved heads, 100-epoch histories, normalization buffers, held-out logits/classes, and fold decisions are bound by fold receipts and the final OOF hash chain.
- OOF frozen: `2026-09-13T03:04:44.810894+00:00`; reference evaluator opened: `2026-09-13T03:04:49.305268+00:00`. Separate processes, both exit 0.
- Decision SHA256: `03a97e0c3fe34925c4de0aa878393766cc7203dab8315128ea49c14568c22893`. All 120 decisions remain byte-identical after evaluation.
- Eight post-freeze reference artifacts are bound in run/evaluation_receipt.json. Training-stage and evaluation-stage counts overlap for accepted target artifacts.

## Literal acceptance clauses

| Clause | Observed ratio | Limit | Pass |
|---|---:|---:|---|
| Pooled / H0 | 0.946340005077044 | 0.97 | true |
| Pooled / H* | 1.018412334419824 | 1.03 | true |
| Offset / H0 | 0.874913447259421 | 0.95 | true |
| Left/right / H0 | 0.983709474106575 | 1.01 | true |
| Quadrants / H0 | 1.000000000000000 | 1.01 | true |

Literal vector `[true, true, true, true, true]`, 5/5. No tolerance or fallback.

## Metrics and post-freeze diagnostics

| Group | N | H0 MSE | Selected MSE | H* MSE | Selected/H0 | Selected/H* |
|---|---:|---:|---:|---:|---:|---:|
| spatial_pool | 120 | 0.0350435737481651 | 0.0331631357587564 | 0.0325635645189323 | 0.946340005077 | 1.018412334420 |
| left_right | 40 | 0.0333970155101269 | 0.0328529605641961 | 0.0321709857787937 | 0.983709474107 | 1.021198442289 |
| quadrants | 40 | 0.0309838496497832 | 0.0309838496497832 | 0.0309821883565746 | 1.000000000000 | 1.000053620912 |
| offset_left_right_40 | 40 | 0.0407498560845852 | 0.0356525970622897 | 0.0345375194214284 | 0.874913447259 | 1.032285979409 |

| Group | x / y / joint exact agreement | no move / x only / y only / both | beneficial / equal / harmful |
|---|---|---|---|
| spatial_pool | 105 / 98 / 87 | 52 / 2 / 39 / 27 | 57 / 52 / 11 |
| left_right | 36 / 29 / 28 | 5 / 0 / 31 / 4 | 24 / 5 / 11 |
| quadrants | 39 / 40 / 39 | 40 / 0 / 0 / 0 | 0 / 40 / 0 |
| offset_left_right_40 | 30 / 29 / 20 | 7 / 2 / 8 / 23 | 33 / 7 / 0 |

Confusion matrices use truth rows and prediction columns in fixed order `[center 0.5, lower 0.4, upper 0.6]`.

**spatial_pool**

- x: `[[80, 4, 0], [9, 25, 0], [2, 0, 0]]`
- y: `[[52, 3, 2], [2, 32, 3], [0, 12, 14]]`

**left_right**

- x: `[[36, 4, 0], [0, 0, 0], [0, 0, 0]]`
- y: `[[5, 1, 1], [0, 18, 2], [0, 7, 6]]`

**quadrants**

- x: `[[39, 0, 0], [0, 0, 0], [1, 0, 0]]`
- y: `[[40, 0, 0], [0, 0, 0], [0, 0, 0]]`

**offset_left_right_40**

- x: `[[5, 0, 0], [9, 25, 0], [1, 0, 0]]`
- y: `[[7, 2, 1], [2, 14, 1], [0, 5, 8]]`

## Verification and limitations

- Eight focused tests passed. Formal training ran once (10.50 seconds including startup); separate evaluation ran once (4.86 seconds). Exact commands, timestamps and exit codes: T018C_commands.json.
- Independent verifier imports no TTIE implementation: it reconstructs feature contrasts and all saved-head predictions exactly, checks train-only statistics, original fold bytes and actual image-group exclusion, all source/input hashes, freeze order, every selected MSE and all group diagnostics. PASS: 10 heads / 120 decisions / 120 identities and reference rows. No formal retraining for verification.
- Initial verifier assertion compared the entire historical source-binding dictionary with a newer dictionary containing an additional commit field. The original dictionary has path and SHA256 only; both matched exactly. Corrected the verifier to compare those two original fields, retaining the independent git-blob commit/hash checks. First failed log is preserved; no model, decision, metric or training source changed.
- Pooled MSE improves 5.366% over H0; offset improves 12.509%. Joint target agreement is 87/120. All 11 harmful selections occur in left/right; quadrants remains at the center for all 40 rows. The x head never predicts upper, where only two target examples exist. These are descriptive diagnostics only; no class weighting, threshold, or retuning was applied.
- This is evidence of development-level feature sufficiency for the fixed probe, not fresh generalization or a deployable final learner. The 5/5 threshold result does not establish universal per-episode benefit.

## Stop and handoff

T018-C is complete. Research lead should review this bounded positive result and explicitly issue any next task. No final all-ID training, gating, feature redesign, new architecture, or fresh qualification starts automatically.
