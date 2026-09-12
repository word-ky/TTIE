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
