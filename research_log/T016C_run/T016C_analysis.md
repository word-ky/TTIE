# T016-C: grouped OOF feature-sufficiency probe

Result: **neither_probe_establishes_development_rankability**. Development-only supervised diagnostic; not fresh qualification or a deployable selector.

Forty fixed development IDs, sorted-ID modulo five folds; 32 training / 8 held-out IDs per fold. All conditions and candidates for an image stay together. Ten heads, CPU, exact fixed T014 value-head recipe. Train-only normalization; no model/epoch selection. All OOF predictions were frozen before reference evaluation.

## probe28

Fixed clauses: 0/5; {'spatial_improves_region2_3pct': False, 'spatial_within_hard_oracle_5pct': False, 'offset_improves_region2_5pct': False, 'left_right_no_more_than_1pct_worse': False, 'quadrants_no_more_than_1pct_worse': False}

| Group | Selected MSE | Region2 | Hard oracle | Frozen T016-B | /Region2 | /Hard oracle | /Frozen T016-B |
|---|---:|---:|---:|---:|---:|---:|---:|
| spatial_pool | 0.0358066896559 | 0.0350435737482 | 0.0325635645189 | 0.0378572608073 | 1.02177620106 | 1.09959367732 | 0.945834138349 |
| left_right | 0.0345864885952 | 0.0333970155101 | 0.0321709857788 | 0.0359704593895 | 1.03561614914 | 1.07508327016 | 0.96152479513 |
| quadrants | 0.0330392681877 | 0.0309838496498 | 0.0309821883566 | 0.0383517022361 | 1.06633838471 | 1.06639556275 | 0.861481140637 |
| offset_left_right_40 | 0.0397943121847 | 0.0407498560846 | 0.0345375194214 | 0.0392496207962 | 0.976550987127 | 1.15220527853 | 1.01387762168 |

| Group | Selected counts | Oracle counts | Disagreement | Outside oracle ties | Spearman summary |
|---|---|---|---:|---:|---|
| spatial_pool | [16, 9, 6, 16, 42, 11, 5, 14, 1] | [27, 2, 12, 23, 41, 14, 0, 1, 0] | 0.6083333333333333 | 0.6083333333333333 | {'count': 120, 'null_count': 9, 'mean': 0.41913425899437323, 'median': 0.5666666666666667, 'min': -0.9333333333333333, 'max': 1.0} |
| left_right | [7, 2, 0, 12, 10, 4, 1, 4, 0] | [2, 0, 0, 23, 2, 13, 0, 0, 0] | 0.725 | 0.725 | {'count': 40, 'null_count': 3, 'mean': 0.4968468468468468, 'median': 0.6, 'min': -0.4166666666666667, 'max': 1.0} |
| quadrants | [1, 4, 2, 0, 25, 2, 1, 4, 1] | [0, 0, 0, 0, 39, 0, 0, 1, 0] | 0.35 | 0.35 | {'count': 40, 'null_count': 0, 'mean': 0.5189309020427191, 'median': 0.6416666666666666, 'min': -0.38333333333333336, 'max': 0.9666666666666667} |
| offset_left_right_40 | [8, 3, 4, 4, 7, 5, 3, 6, 0] | [25, 2, 12, 0, 0, 1, 0, 0, 0] | 0.75 | 0.75 | {'count': 40, 'null_count': 6, 'mean': 0.21715686274509804, 'median': 0.275, 'min': -0.9333333333333333, 'max': 0.9666666666666667} |

| Fold | Held-out IDs | Selected MSE | LR / Quadrants / Offset MSE | Final train Huber |
|---|---|---:|---|---:|
| 0 | [55167, 56288, 57149, 57672, 58393, 59386, 60102, 60770] | 0.0432570583653 | {'left_right': 0.04247221723198891, 'quadrants': 0.04585543577559292, 'offset_left_right_40': 0.041443522088229656} | 0.0535377306795 |
| 1 | [55299, 56344, 57232, 58029, 58539, 59598, 60347, 60823] | 0.0247305707308 | {'left_right': 0.021285319118760526, 'quadrants': 0.023609573603607714, 'offset_left_right_40': 0.02929681946989149} | 0.0474748528666 |
| 2 | [55528, 56350, 57238, 58111, 58636, 59635, 60363, 60835] | 0.0421076838781 | {'left_right': 0.04103725589811802, 'quadrants': 0.03437544486951083, 'offset_left_right_40': 0.05091035086661577} | 0.066398843433 |
| 3 | [55950, 56545, 57244, 58350, 58705, 59920, 60449, 60855] | 0.0403848751448 | {'left_right': 0.039371089078485966, 'quadrants': 0.035608510952442884, 'offset_left_right_40': 0.0461750254034996} | 0.0547366887331 |
| 4 | [56127, 57027, 57597, 58384, 59044, 60052, 60507, 60886] | 0.0285532601605 | {'left_right': 0.02876656164880842, 'quadrants': 0.02574737573741004, 'offset_left_right_40': 0.03114584309514612} | 0.0528893273462 |

Tie/null details per condition: {'spatial_pool': {'prediction_tied_episodes': 14, 'oracle_tied_episodes': 12}, 'left_right': {'prediction_tied_episodes': 5, 'oracle_tied_episodes': 5}, 'quadrants': {'prediction_tied_episodes': 2, 'oracle_tied_episodes': 0}, 'offset_left_right_40': {'prediction_tied_episodes': 7, 'oracle_tied_episodes': 7}}.

## probe30

Fixed clauses: 0/5; {'spatial_improves_region2_3pct': False, 'spatial_within_hard_oracle_5pct': False, 'offset_improves_region2_5pct': False, 'left_right_no_more_than_1pct_worse': False, 'quadrants_no_more_than_1pct_worse': False}

| Group | Selected MSE | Region2 | Hard oracle | Frozen T016-B | /Region2 | /Hard oracle | /Frozen T016-B |
|---|---:|---:|---:|---:|---:|---:|---:|
| spatial_pool | 0.0354510319846 | 0.0350435737482 | 0.0325635645189 | 0.0378572608073 | 1.01162718846 | 1.08867172585 | 0.93643943668 |
| left_right | 0.0342112310929 | 0.0333970155101 | 0.0321709857788 | 0.0359704593895 | 1.02437989055 | 1.06341880004 | 0.951092415097 |
| quadrants | 0.0333758114488 | 0.0309838496498 | 0.0309821883566 | 0.0383517022361 | 1.07720027776 | 1.07725803822 | 0.870256325086 |
| offset_left_right_40 | 0.0387660534121 | 0.0407498560846 | 0.0345375194214 | 0.0392496207962 | 0.951317553899 | 1.1224330543 | 0.987679692841 |

| Group | Selected counts | Oracle counts | Disagreement | Outside oracle ties | Spearman summary |
|---|---|---|---:|---:|---|
| spatial_pool | [9, 5, 12, 16, 40, 25, 2, 7, 4] | [27, 2, 12, 23, 41, 14, 0, 1, 0] | 0.6333333333333333 | 0.5666666666666667 | {'count': 120, 'null_count': 7, 'mean': 0.5275345120253869, 'median': 0.5833333333333334, 'min': -0.7833333333333333, 'max': 1.0} |
| left_right | [4, 1, 4, 8, 10, 11, 1, 0, 1] | [2, 0, 0, 23, 2, 13, 0, 0, 0] | 0.725 | 0.65 | {'count': 40, 'null_count': 2, 'mean': 0.5548264357132587, 'median': 0.6166666666666667, 'min': -0.5, 'max': 1.0} |
| quadrants | [0, 1, 0, 2, 24, 4, 1, 6, 2] | [0, 0, 0, 0, 39, 0, 0, 1, 0] | 0.425 | 0.425 | {'count': 40, 'null_count': 0, 'mean': 0.5595833333333333, 'median': 0.625, 'min': -0.35, 'max': 0.95} |
| offset_left_right_40 | [5, 3, 8, 6, 6, 10, 0, 1, 1] | [25, 2, 12, 0, 0, 1, 0, 0, 0] | 0.75 | 0.625 | {'count': 40, 'null_count': 5, 'mean': 0.46127605624090157, 'median': 0.5, 'min': -0.7833333333333333, 'max': 1.0} |

| Fold | Held-out IDs | Selected MSE | LR / Quadrants / Offset MSE | Final train Huber |
|---|---|---:|---|---:|
| 0 | [55167, 56288, 57149, 57672, 58393, 59386, 60102, 60770] | 0.0431003960936 | {'left_right': 0.04263556282967329, 'quadrants': 0.04501480027101934, 'offset_left_right_40': 0.041650825180113316} | 0.0435444000694 |
| 1 | [55299, 56344, 57232, 58029, 58539, 59598, 60347, 60823] | 0.0239705079778 | {'left_right': 0.019814823172055185, 'quadrants': 0.02314280450809747, 'offset_left_right_40': 0.02895389625336975} | 0.0399213057977 |
| 2 | [55528, 56350, 57238, 58111, 58636, 59635, 60363, 60835] | 0.0423215969543 | {'left_right': 0.041563516831956804, 'quadrants': 0.037016590824350715, 'offset_left_right_40': 0.04838468320667744} | 0.0611170662774 |
| 3 | [55950, 56545, 57244, 58350, 58705, 59920, 60449, 60855] | 0.0406262602191 | {'left_right': 0.04043620638549328, 'quadrants': 0.035608510952442884, 'offset_left_right_40': 0.04583406331948936} | 0.0502439137134 |
| 4 | [56127, 57027, 57597, 58384, 59044, 60052, 60507, 60886] | 0.0272363986781 | {'left_right': 0.02660604624543339, 'quadrants': 0.02609635068802163, 'offset_left_right_40': 0.029006799100898206} | 0.0461615504766 |

Tie/null details per condition: {'spatial_pool': {'prediction_tied_episodes': 0, 'oracle_tied_episodes': 12}, 'left_right': {'prediction_tied_episodes': 0, 'oracle_tied_episodes': 5}, 'quadrants': {'prediction_tied_episodes': 0, 'oracle_tied_episodes': 0}, 'offset_left_right_40': {'prediction_tied_episodes': 0, 'oracle_tied_episodes': 7}}.

## Two-probe comparison

{'probe30_over_probe28': 0.990067284222979, 'median_spearman_difference': 0.01666666666666672}

Spearman uses average ranks for ties; constant ranks are null. Exact predicted-value and oracle ties use the first lexicographic boundary. Head inputs exclude IDs/condition/candidate IDs/reference MSE; probe30 appends only the prescribed gx/gy coordinates.

No threshold, fold, epoch, architecture or loss was changed. No new images, CLIP/A6000 scoring, rendering, TTT, deployable boundary model, or follow-on experiment. Stop for research-lead review.
