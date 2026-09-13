# T018-B — frozen T014 energy local direction is insufficient (0/5)

**Literal result: frozen_energy_local_signal_insufficient.** All five clauses fail. Factorizing the already-saved frozen T014 hard-cross scores does not recover the viable T018-A reference target: pooled selected MSE is 8.6198% worse than canonical Region2; 58/120 outputs are harmful, and joint target agreement is 32/120. This is a label-free decision rule evaluated on the fixed development table, not fresh qualification.

## Exact clauses and metrics

| Clause | Observed ratio | Required | Pass |
|---|---:|---:|---|
| Pooled selected/H0 | 1.08619818325 | <= 0.97 | False |
| Pooled selected/H* | 1.16892197467 | <= 1.03 | False |
| Offset selected/H0 | 0.964061213085 | <= 0.95 | False |
| Left/right selected/H0 | 1.07525284318 | <= 1.01 | False |
| Quadrants selected/H0 | 1.25863013439 | <= 1.01 | False |

Vector `[false,false,false,false,false]`. No tolerance, fallback, calibration or alternative rule.

| Group | H0 | Selected | H* | Selected/H0 | Selected/H* |
|---|---:|---:|---:|---:|---:|
| spatial_pool | 0.0350435737482 | 0.0380642661398 | 0.0325635645189 | 1.08619818325 | 1.16892197467 |
| left_right | 0.0333970155101 | 0.0359102358809 | 0.0321709857788 | 1.07525284318 | 1.11623051055 |
| quadrants | 0.0309838496498 | 0.0389972068486 | 0.0309821883566 | 1.25863013439 | 1.25869762328 |
| offset_left_right_40 | 0.0407498560846 | 0.0392853556899 | 0.0345375194214 | 0.964061213085 | 1.13746894242 |

## Post-freeze target agreement and movement

| Group | x match | y match | Joint match | No move / x only / y only / both | Beneficial / equal / harmful |
|---|---|---|---|---|---|
| spatial_pool | 61/120 (50.8333%) | 55/120 (45.8333%) | 32/120 (26.6667%) | 38 / 23 / 30 / 29 | 24 / 38 / 58 |
| left_right | 24/40 (60.0000%) | 19/40 (47.5000%) | 14/40 (35.0000%) | 13 / 7 / 11 / 9 | 8 / 13 / 19 |
| quadrants | 20/40 (50.0000%) | 21/40 (52.5000%) | 11/40 (27.5000%) | 12 / 9 / 9 / 10 | 0 / 12 / 28 |
| offset_left_right_40 | 17/40 (42.5000%) | 15/40 (37.5000%) | 7/40 (17.5000%) | 13 / 7 / 10 / 10 | 16 / 13 / 11 |

Agreement is exact coordinate equality with the accepted T018-A center-first target, computed only after all energy decisions were frozen. It does not modify any decision. Quadrants have 28 harmful outputs and no beneficial output. Offset improves only 3.5939% versus H0 and therefore misses its required 5% gain.

## Harmful examples

All 58 examples are preserved in the per-family summary JSON with only row index, chosen/reference axes, H0/selected/H*. Rows below are illustrative; no example is used for retuning.

| Row index | Energy axes | T018-A axes | H0 | Selected | H* |
|---|---|---|---:|---:|---:|
| 1 | (0.5, 0.4) | (0.5, 0.5) | 0.0410976856947 | 0.0417934581637 | 0.0410976856947 |
| 3 | (0.6, 0.5) | (0.5, 0.6) | 0.0173140969127 | 0.0261382814497 | 0.0136866839603 |
| 4 | (0.6, 0.6) | (0.5, 0.5) | 0.0177506115288 | 0.0333853363991 | 0.0177506115288 |
| 5 | (0.6, 0.5) | (0.4, 0.6) | 0.0296823047101 | 0.0405887141824 | 0.0139650749043 |
| 6 | (0.5, 0.6) | (0.5, 0.4) | 0.0136059494689 | 0.0136833321303 | 0.0135645018891 |

## Decision/evaluation separation

Source freeze `e6ef5ab53823cbb07f8fb45e8ac22753a6d83d12`. The select process reads accepted T016-B scoring selection/config/receipt from `4062e01cb93de731c394015c5ac741d6c08e04d8`. The original JSON envelope necessarily contains opaque episode paths, features, gate and old global choices; these per-episode fields are ignored in selection. Only five raw energies at nine-hard indices [4,1,7,3,5] feed the selector. There is no score renormalization or model run. Exact ties use center0.5, lower0.4, upper0.6 independently for both axes.

The select process freezes all120 decisions at 2026-09-13T01:59:53.619083+00:00; decision SHA256 `9e1f4bc29629600760c249adb11f270833e6b4bc3f8c40bd00596020b1af0017`. A separate evaluate process first validates that hash/config, then first opens reference artifacts at 2026-09-13T01:59:54.410768+00:00. The decision bytes remain unchanged afterward.

Reference inputs are six accepted T018-A artifacts from merge `5ecf598763c499b2275994be53f9218a3757245c` (config, decisions/freeze, quantities/freeze, report receipt), plus original T016-A table/config. All 11 input blobs and six source files are hash-bound. Exact 120 episode identities and fixed-corner hashes and the nine hard coordinates agree; the score and reference row orders also match, but evaluation uses explicit identity joins. Identity precheck occurs after decision freeze and before metrics because inspecting identifiers before freeze would violate the stricter decision-stage constraint. No family/image ID/reference/target value enters selection.

Four focused tests PASS in0.048s; compile PASS. One select process and one evaluate process both exit0. Independent `T018B_verify.py` passes all120 energy decisions, identity joins, target/reference values, every group statistic/harmful example, strict clauses, source/input hashes and freeze-before-reference ordering. Tests include metadata and noncross-energy invariance, exact ties, blocked pre-freeze reference access and post-freeze example restrictions.

Commands: `python -m ttie.energy_local select --source-sha e6ef5ab53823cbb07f8fb45e8ac22753a6d83d12 --output research_log/T018B_run`; separately `python -m ttie.energy_local evaluate --output research_log/T018B_run`. Full interpreter paths, UTC times and exit codes are in the select/evaluate receipts. No formal-run failure or scientific deviation. A setup file search used unsupported Windows glob arguments, then succeeded with exact Git paths; no data or code was affected.

## Bounded conclusion and stop

T018-A target viability does not transfer through the existing frozen T014 scalar energy under this fixed factorized local rule. The result does not invalidate T014 canonical Region2 TTT; it limits reuse of that energy for boundary direction on this development table. Preserve the negative and stop after T018-B. No confidence gate, dedicated geometry predictor, training, rerendering, new images, model/CLIP/TTT run, GPU work or fresh qualification. Await research-lead review; no automatic T018-C.
