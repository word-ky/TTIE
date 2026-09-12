# T016-F: fully nested rank30 + single confidence fallback

Result: **t016e_pass_does_not_survive_fully_nested_development_audit**, **4/5**.

Five outer32/8ID splits; each outertrain32 split into four24/8ID innerfits. Exactly25freshheads using the unchangedDrecipe; no oldhead/score used for primarynested decisions.
Thresholds: [0.5, 0.5, 0.5, 1.0, 0.75].

Literalclauses: {'spatial_improves_region2_3pct': True, 'spatial_within_hard_oracle_5pct': True, 'offset_improves_region2_5pct': True, 'left_right_no_more_than_1pct_worse': False, 'quadrants_no_more_than_1pct_worse': True}

| Group | Selected MSE | /Region2 | /Hard oracle | /Ungated nested | /Old E | Adaptive / Canonical | Adapted gains |
|---|---:|---:|---:|---:|---:|---|---|
| spatial_pool | 0.0339618805136 | 0.969132907439 | 1.04294112194 | 0.992510391456 | 1.00168669041 | 35 / 85 | {'beneficial': 27, 'harmful': 6, 'zero': 2} |
| left_right | 0.0340147255687 | 1.01849596586 | 1.05731064017 | 0.972899332945 | 1.00852872791 | 10 / 30 | {'beneficial': 3, 'harmful': 6, 'zero': 1} |
| quadrants | 0.0309838496498 | 1 | 1.00005362091 | 0.983995820478 | 1 | 0 / 40 | {'beneficial': 0, 'harmful': 0, 'zero': 0} |
| offset_left_right_40 | 0.0368870663224 | 0.905207278421 | 1.06802882605 | 1.01885385589 | 0.996862729878 | 25 / 15 | {'beneficial': 24, 'harmful': 0, 'zero': 1} |

| Group | q distribution | Disagreement / Outside oracle ties | Selected / Oracle counts |
|---|---|---|---|
| spatial_pool | {'count': 120, 'mean': 0.26783018609886866, 'median': 0.2841843780692793, 'quantiles': {'0': -1.2891933917858664, '25': -0.2953141379989333, '50': 0.2841843780692793, '75': 0.858051015485551, '100': 2.6785315226655437}} | 0.5166666666666667 / 0.44166666666666665 | [12, 5, 8, 3, 85, 4, 2, 0, 1] / [27, 2, 12, 23, 41, 14, 0, 1, 0] |
| left_right | {'count': 40, 'mean': 0.4206974754985484, 'median': 0.3787803145730181, 'quantiles': {'0': -1.0805638129958552, '25': 0.07240065938100392, '50': 0.3787803145730181, '75': 0.6625348711319365, '100': 1.6627812289570538}} | 0.9 / 0.8 | [3, 0, 0, 2, 30, 2, 2, 0, 1] / [2, 0, 0, 23, 2, 13, 0, 0, 0] |
| quadrants | {'count': 40, 'mean': -0.5554363921581098, 'median': -0.661405003621444, 'quantiles': {'0': -1.2891933917858664, '25': -0.8420760858641135, '50': -0.661405003621444, '75': -0.23440179585129284, '100': 0.5125146666684557}} | 0.025 / 0.025 | [0, 0, 0, 0, 40, 0, 0, 0, 0] / [0, 0, 0, 0, 39, 0, 0, 1, 0] |
| offset_left_right_40 | {'count': 40, 'mean': 0.9382294749561674, 'median': 0.8933289597844493, 'quantiles': {'0': -0.4852272503878511, '25': 0.5332513795735243, '50': 0.8933289597844493, '75': 1.3625481363084821, '100': 2.6785315226655437}} | 0.625 / 0.5 | [9, 5, 8, 1, 15, 2, 0, 0, 0] / [25, 2, 12, 0, 0, 1, 0, 0, 0] |

| Fold | Threshold | Selected MSE | Adaptive / Canonical | Per-condition adaptive/canonical |
|---|---:|---:|---|---|
| 0 | 0.5 | 0.0421568798677 | 9 / 15 | {'left_right': (3, 5), 'quadrants': (0, 8), 'offset_left_right_40': (6, 2)} |
| 1 | 0.5 | 0.0211793526929 | 8 / 16 | {'left_right': (2, 6), 'quadrants': (0, 8), 'offset_left_right_40': (6, 2)} |
| 2 | 0.5 | 0.0406880689552 | 6 / 18 | {'left_right': (1, 7), 'quadrants': (0, 8), 'offset_left_right_40': (5, 3)} |
| 3 | 1.0 | 0.0388676455865 | 6 / 18 | {'left_right': (1, 7), 'quadrants': (0, 8), 'offset_left_right_40': (5, 3)} |
| 4 | 0.75 | 0.0269174554657 | 6 / 18 | {'left_right': (3, 5), 'quadrants': (0, 8), 'offset_left_right_40': (3, 5)} |

Per-head histories, pair counts, normalization, calibration tables, innerOOFs and allfold hashes are persisted. q uses the unchanged Epopulationstd/floor formula andlinear0/25/50/75/100percentiles for description only.
Every fold excludes its outer-heldout IDs from allfiveheadtraining sets and thresholdcalibration. Each calibrationimage is also excluded from its own innerOOFranker training. All25heads andfivefolddecisions are frozen before combinedouterreferenceevaluation.
Canonical candidate4 versus Region2 maxMSEdifference: 0.0.

Development-only, on already-inspected40IDs. Historical E has inherited scorer dependence and is a comparator, not a qualified baseline. No fresh qualification or deployablethreshold claim. Stop after the literal result; no alternategrid,model,confidence orfollow-onwork.
