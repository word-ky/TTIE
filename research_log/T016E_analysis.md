# T016-E: fixed OOF confidence-abstention audit

Result: **rank30_development_safe_with_training_only_confidence_fallback**; **5/5** clauses.

Thresholds by outer fold: [0.75, 0.75, 0.75, 0.75, 0.75]. Infinity always selects canonical.

Literal clauses: {'spatial_improves_region2_3pct': True, 'spatial_within_hard_oracle_5pct': True, 'offset_improves_region2_5pct': True, 'left_right_no_more_than_1pct_worse': True, 'quadrants_no_more_than_1pct_worse': True}

| Group | MSE | /Region2 | /Hard oracle | /Ungated rank30 | /Pointwise probe30 | Adaptive / Canonical | Beneficial / Harmful / Zero among adapted |
|---|---:|---:|---:|---:|---:|---|---|
| spatial_pool | 0.0339046937918 | 0.967501032727 | 1.04118496524 | 0.990839152561 | 0.956381010474 | 32 / 88 | {'beneficial': 27, 'harmful': 4, 'zero': 1} |
| left_right | 0.03372707651 | 1.00988294896 | 1.04836938295 | 0.964671908714 | 0.98584808066 | 9 / 31 | {'beneficial': 4, 'harmful': 4, 'zero': 1} |
| quadrants | 0.0309838496498 | 1 | 1.00005362091 | 0.983995820478 | 0.928332475072 | 0 / 40 | {'beneficial': 0, 'harmful': 0, 'zero': 0} |
| offset_left_right_40 | 0.0370031552156 | 0.908056095679 | 1.07139006609 | 1.02206033524 | 0.954524692578 | 23 / 17 | {'beneficial': 23, 'harmful': 0, 'zero': 0} |

| Group | q summary (linear quantiles0/25/50/75/100) | Disagreement | Outside oracle ties | Selected / Oracle counts |
|---|---|---:|---:|---|
| spatial_pool | {'count': 120, 'mean': 0.26783018609886866, 'median': 0.2841843780692793, 'quantiles': {'0': -1.2891933917858664, '25': -0.2953141379989333, '50': 0.2841843780692793, '75': 0.858051015485551, '100': 2.6785315226655437}} | 0.5083333333333333 | 0.43333333333333335 | [13, 5, 7, 3, 88, 2, 1, 0, 1] / [27, 2, 12, 23, 41, 14, 0, 1, 0] |
| left_right | {'count': 40, 'mean': 0.4206974754985484, 'median': 0.3787803145730181, 'quantiles': {'0': -1.0805638129958552, '25': 0.07240065938100392, '50': 0.3787803145730181, '75': 0.6625348711319365, '100': 1.6627812289570538}} | 0.875 | 0.775 | [3, 0, 0, 2, 31, 2, 1, 0, 1] / [2, 0, 0, 23, 2, 13, 0, 0, 0] |
| quadrants | {'count': 40, 'mean': -0.5554363921581098, 'median': -0.661405003621444, 'quantiles': {'0': -1.2891933917858664, '25': -0.8420760858641135, '50': -0.661405003621444, '75': -0.23440179585129284, '100': 0.5125146666684557}} | 0.025 | 0.025 | [0, 0, 0, 0, 40, 0, 0, 0, 0] / [0, 0, 0, 0, 39, 0, 0, 1, 0] |
| offset_left_right_40 | {'count': 40, 'mean': 0.9382294749561674, 'median': 0.8933289597844493, 'quantiles': {'0': -0.4852272503878511, '25': 0.5332513795735243, '50': 0.8933289597844493, '75': 1.3625481363084821, '100': 2.6785315226655437}} | 0.625 | 0.5 | [10, 5, 7, 1, 17, 0, 0, 0, 0] / [25, 2, 12, 0, 0, 1, 0, 0, 0] |

| Fold | Threshold | Selected MSE | Adaptive / Canonical | Per-condition adaptive / canonical |
|---|---:|---:|---|---|
| 0 | 0.75 | 0.0422572983274 | 8 / 16 | {'left_right': (3, 5), 'quadrants': (0, 8), 'offset_left_right_40': (5, 3)} |
| 1 | 0.75 | 0.0209562610059 | 6 / 18 | {'left_right': (1, 7), 'quadrants': (0, 8), 'offset_left_right_40': (5, 3)} |
| 2 | 0.75 | 0.040737999642 | 4 / 20 | {'left_right': (0, 8), 'quadrants': (0, 8), 'offset_left_right_40': (4, 4)} |
| 3 | 0.75 | 0.0386544545181 | 8 / 16 | {'left_right': (2, 6), 'quadrants': (0, 8), 'offset_left_right_40': (6, 2)} |
| 4 | 0.75 | 0.0269174554657 | 6 / 18 | {'left_right': (3, 5), 'quadrants': (0, 8), 'offset_left_right_40': (3, 5)} |

All eight calibration means per fold, exact q/decision values and per-fold/condition diagnostics are in the accompanying JSON artifacts.
Canonical candidate4 versus accepted Region2 maximum MSE difference: 0.0.

Only the fixed q and eight thresholds were used. Strict q>t adapts; equal q falls back; equal calibration means choose the larger threshold. Calibration uses the other32IDs and their own pre-existing OOF scores; no model loaded or retrained. All five thresholds and120decisions were frozen before evaluation.

Protocol limitation: this is the requested reuse of OOF scores, not fully nested ranker training. A calibration image's frozen D ranker may have trained on IDs held out by the current outer threshold fold. Direct threshold fitting never reads those held-out reference rows, but inherited scorer-training dependence remains. Development-only; no independent fresh-generalization claim.

Stop after the literal result. No alternate threshold/grid, second gate, fresh data or follow-on experiment.

## Fixed calibration tables

| Threshold | Fold0 meanMSE | Fold1 | Fold2 | Fold3 | Fold4 |
|---|---:|---:|---:|---:|---:|
| 0.0 | 0.0321155768615426 | 0.0373653330558833 | 0.0326364154898329 | 0.0330212425081603 | 0.035952237813035 |
| 0.25 | 0.0319456409439833 | 0.0372432343768499 | 0.032323437272377 | 0.0327452062095593 | 0.0357123527792282 |
| 0.5 | 0.0320643586989415 | 0.0373087404926385 | 0.0324315614270745 | 0.0329271641239757 | 0.0356824899208732 |
| 0.75 | 0.0318165426579071 | 0.0371418019882791 | 0.0321963673292582 | 0.0327172536102201 | 0.0356515033733255 |
| 1.0 | 0.0322222617154087 | 0.0375525006723668 | 0.0324146367250554 | 0.0330746545272026 | 0.0359022100553072 |
| 1.5 | 0.0328024139404685 | 0.037686293386893 | 0.032937561525614 | 0.0335573839693097 | 0.0364749354290931 |
| 2.0 | 0.0330161663150648 | 0.0378443335309081 | 0.0332129545713542 | 0.0337363775421788 | 0.0367503284748333 |
| inf | 0.0330484863904227 | 0.0380087606075297 | 0.0333773816479758 | 0.0338684845434424 | 0.0369147555514549 |

## Provenance and validation

Issued inbox993c4813 / research state883a6b3d. Frozen scientific source **c98e6d119b89b8ffb755ed5b1e0d064973684675**, branch codex/T016E-confidence-abstention. Exact raw donor evaluator from Daa71d268294e35f5df67c76eada29f9bec117abe; no algorithm or newline changes. Six scientific runtime files bind to their actual Gitblob bytes. Eight immutable source artifacts from accepted B4062e01cb93de731c394015c5ac741d6c08e04d8, C433683eccad24dc763450be6a546072a72e0910b, Daa71d268294e35f5df67c76eada29f9bec117abe are identified with full paths/hashes in config.json. D's existing OOF freeze hash and C/B input provenance are checked before calibration. E has no model/torch import, loading, fitting, or scoring pass.

Command: `D:/anaconda3/python.exe -m ttie.boundary_confidence_run --source-sha c98e6d119b89b8ffb755ed5b1e0d064973684675 --output research_log/T016E_run`. Python3.12.7 CPU, standard-library implementation. One successful run; exact exit time in T016E_run.log. Baseline reproduces both accepted D probes' complete group/fold/count/rank/clauses exactly. Kernel4testsPASS0.030s; integrated6focusedtestsPASS0.103s; py_compilePASS. These are overlapping incremental checks, not10unique tests. Focused scope only; no unrelated full suite or training rerun.

Tests prove the exact confidence formula including near-constant denominator floor, constantq0, first noncanonical tie, strict q>t, exact8thresholdgrid, safer exact calibration ties, direct heldout-reference access guard and mutation invariance, use of prior OOF rows, freezing all thresholds/120decisions before evaluation, and gain/coverage/quantile arithmetic. No test/run failures, repair, or outcome-driven changes.

Independent audit:6scientificsource Gitblob/current-file hashes,8inputartifact hashes, exact groupedfold membership,120q/decisions (NumPy populationstd maximum absolute difference2.220446049250313e-16),40calibrationmeans,5thresholds,all MSEjoins/ratios/counts/linearquantiles and fiveclauses verified. No model or calibration output changed during audit. Canonical candidate4's accepted referenceMSE equals Region2 exactly in all120episodes.

All thresholds/decisions frozen2026-09-12T20:34:43.821950Z; evaluation began20:34:43.837677Z. Freeze SHA256a25b9ee3da0c7d4709758c8924cf1e60faca5b43f129e35d32525f4e187cdb5a. Decisions SHA256a5d6c4b864b5dcbdf28d354430d68d7b8f511004040ae6d4a7935bee51890f94; calibration SHA2564c5e289ecd67efcf92b0c7751f1c58e6c57a69200a58ca70d77d3580b0ca264f. Frozen hashes unchanged after evaluation.

## Interpretation boundaries

The literal prescribed audit passes5/5. Confidence fallback avoids all quadrant changes and retains23beneficial offset changes. Of32adapted episodes,27improve,4harm,and1ties canonical. All4harmful cases are left/right; that group's ratio1.00988294896 is only0.011705percentage points inside the1%allowed degradation limit. The result is development evidence with a narrow safety margin, not a robustness claim. No alternative threshold is tested or promoted after the outcome.

The allowed narrow reading is that this existing rank30 scorer with the specified training-only confidence fallback meets the development clauses; forced selection is a plausible diagnosed bottleneck under this audit. Do not promote0.75 as a deployable threshold: these are five separately calibrated thresholds that happen to coincide, on already-inspected development data.

The inherited dependence is concrete: in all480calibrationepisode appearances across five outer folds, the prior D ranker producing that calibration image's OOF score was trained on the current outer-held-out8IDs. This follows from using the exact requested original OOF scores. Direct threshold calibration never indexes its outer-held-out reference rows, but mutation tests on those rows after the rankers are frozen cannot remove the earlier training dependence. Therefore **5/5 is the literal result of the specified development audit, not a leakage-free fully nested cross-validation or independent generalization result**. No retraining was authorized; the implementation follows the requested recipe exactly and reports the limitation rather than silently replacing it.

No scientific deviations, newIDs, referenceMSE recomputation, model/head work, feature expansion, renderer/CLIP/TTT/A6000 experiment, grid change, condition-specific calibration, alternateq, secondgate, modelselection or fresh run. Stop for research review of both the controlled pass and its dependency limitation; no self-merge or follow-on task.

Artifacts: T016E_run/config.json, folds.json, calibrations.json, decisions.json, decisions_frozen.json, evaluation.json, evaluation_receipt.json, summary.json and generated report. All five tables/thresholds,120decisions/q values,source/inputhashes and calibration trace rows are retained; T016E_verification.json/log/script provide the independent audit. Final delivery commit/PR/main-mailbox and server recovery hashes are recorded separately in T016E_delivery.json.
