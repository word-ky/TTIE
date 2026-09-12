# T016-D: grouped OOF pairwise boundary-ranking probe

Result: **neither_rank_probe_establishes_safe_development_ranking**.

Development-only loss-alignment diagnostic. Same 40 IDs, five image-grouped folds, nine candidates and 28/30 features as T016-C. Ten fixed scalar heads; all non-tied within-episode pairs once per epoch; unweighted logistic loss, no target standardization. No fresh qualification or deployable model.

## rank28

Five clauses: 1/5. {'spatial_improves_region2_3pct': False, 'spatial_within_hard_oracle_5pct': False, 'offset_improves_region2_5pct': True, 'left_right_no_more_than_1pct_worse': False, 'quadrants_no_more_than_1pct_worse': False}

| Group | Selected MSE | Region2 | Hard oracle | Frozen T016-B | T016-C counterpart | /Region2 | /Hard oracle | /Frozen B | /Pointwise C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spatial_pool | 0.0354018605974 | 0.0350435737482 | 0.0325635645189 | 0.0378572608073 | 0.0358066896559 | 1.01022403856 | 1.08716171342 | 0.935140573895 | 0.988694038394 |
| left_right | 0.0359136292944 | 0.0333970155101 | 0.0321709857788 | 0.0359704593895 | 0.0345864885952 | 1.07535445146 | 1.11633599111 | 0.998420089816 | 1.03837165185 |
| quadrants | 0.0324687268003 | 0.0309838496498 | 0.0309821883566 | 0.0383517022361 | 0.0330392681877 | 1.04792423044 | 1.04798042109 | 0.846604580949 | 0.982731415715 |
| offset_left_right_40 | 0.0378232256975 | 0.0407498560846 | 0.0345375194214 | 0.0392496207962 | 0.0397943121847 | 0.928180595755 | 1.09513440256 | 0.963658372495 | 0.950468135295 |

| Group | Selected counts | Oracle counts | Disagreement | Outside oracle ties | Spearman |
|---|---|---|---:|---:|---|
| spatial_pool | [19, 15, 9, 15, 38, 12, 4, 6, 2] | [27, 2, 12, 23, 41, 14, 0, 1, 0] | 0.5333333333333333 | 0.5333333333333333 | {'count': 120, 'null_count': 9, 'mean': 0.4584735983337126, 'median': 0.55, 'min': -0.85, 'max': 1.0} |
| left_right | [8, 3, 1, 10, 6, 6, 3, 3, 0] | [2, 0, 0, 23, 2, 13, 0, 0, 0] | 0.75 | 0.75 | {'count': 40, 'null_count': 3, 'mean': 0.38333333333333336, 'median': 0.48333333333333334, 'min': -0.6166666666666667, 'max': 1.0} |
| quadrants | [2, 3, 0, 0, 31, 2, 0, 1, 1] | [0, 0, 0, 0, 39, 0, 0, 1, 0] | 0.25 | 0.25 | {'count': 40, 'null_count': 0, 'mean': 0.5697642353760524, 'median': 0.5833333333333334, 'min': -0.43333333333333335, 'max': 0.9833333333333333} |
| offset_left_right_40 | [9, 9, 8, 5, 1, 4, 1, 2, 1] | [25, 2, 12, 0, 0, 1, 0, 0, 0] | 0.6 | 0.6 | {'count': 40, 'null_count': 6, 'mean': 0.40931372549019607, 'median': 0.6833333333333333, 'min': -0.85, 'max': 1.0} |

| Fold | Held-out IDs | Selected MSE | LR / Quadrants / Offset | Non-tied train pairs | Final train pair loss |
|---|---|---:|---|---:|---:|
| 0 | [55167, 56288, 57149, 57672, 58393, 59386, 60102, 60770] | 0.0448084637367 | {'left_right': 0.045219917548820376, 'quadrants': 0.04565211688168347, 'offset_left_right_40': 0.04355335677973926} | 3249 | 0.13932502916 |
| 1 | [55299, 56344, 57232, 58029, 58539, 59598, 60347, 60823] | 0.0222227352594 | {'left_right': 0.017894462100230157, 'quadrants': 0.022907349630258977, 'offset_left_right_40': 0.025866394047625363} | 3195 | 0.199391562707 |
| 2 | [55528, 56350, 57238, 58111, 58636, 59635, 60363, 60835] | 0.0417215253692 | {'left_right': 0.0435947043588385, 'quadrants': 0.034383751335553825, 'offset_left_right_40': 0.04718612041324377} | 3159 | 0.205120937286 |
| 3 | [55950, 56545, 57244, 58350, 58705, 59920, 60449, 60855] | 0.0399121282777 | {'left_right': 0.040033655473962426, 'quadrants': 0.03473217599093914, 'offset_left_right_40': 0.04497055336833} | 3204 | 0.181085122956 |
| 4 | [56127, 57027, 57597, 58384, 59044, 60052, 60507, 60886] | 0.0283444503439 | {'left_right': 0.03282540699001402, 'quadrants': 0.024668240163009614, 'offset_left_right_40': 0.027539703878574073} | 3285 | 0.200213663963 |

Prediction/oracle minimum tie episodes: {'spatial_pool': (14, 12), 'left_right': (5, 5), 'quadrants': (2, 0), 'offset_left_right_40': (7, 7)}.

## rank30

Five clauses: 1/5. {'spatial_improves_region2_3pct': False, 'spatial_within_hard_oracle_5pct': False, 'offset_improves_region2_5pct': True, 'left_right_no_more_than_1pct_worse': False, 'quadrants_no_more_than_1pct_worse': False}

| Group | Selected MSE | Region2 | Hard oracle | Frozen T016-B | T016-C counterpart | /Region2 | /Hard oracle | /Frozen B | /Pointwise C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spatial_pool | 0.0342181611457 | 0.0350435737482 | 0.0325635645189 | 0.0378572608073 | 0.0354510319846 | 0.976446106541 | 1.05081128713 | 0.903873138627 | 0.965223273629 |
| left_right | 0.0349622251932 | 0.0333970155101 | 0.0321709857788 | 0.0359704593895 | 0.0342112310929 | 1.04686675319 | 1.08676263244 | 0.971970494304 | 1.02195168301 |
| quadrants | 0.0314877858269 | 0.0309838496498 | 0.0309821883566 | 0.0383517022361 | 0.0333758114488 | 1.01626447917 | 1.01631897219 | 0.821027072881 | 0.943431319272 |
| offset_left_right_40 | 0.036204472417 | 0.0407498560846 | 0.0345375194214 | 0.0392496207962 | 0.0387660534121 | 0.888456448579 | 1.04826498902 | 0.922415852245 | 0.933922058873 |

| Group | Selected counts | Oracle counts | Disagreement | Outside oracle ties | Spearman |
|---|---|---|---:|---:|---|
| spatial_pool | [20, 9, 12, 18, 46, 9, 2, 3, 1] | [27, 2, 12, 23, 41, 14, 0, 1, 0] | 0.45 | 0.39166666666666666 | {'count': 120, 'null_count': 7, 'mean': 0.6524167802729759, 'median': 0.75, 'min': -0.4666666666666667, 'max': 1.0} |
| left_right | [8, 0, 2, 12, 9, 6, 2, 0, 1] | [2, 0, 0, 23, 2, 13, 0, 0, 0] | 0.675 | 0.625 | {'count': 40, 'null_count': 2, 'mean': 0.5048379857303595, 'median': 0.5815421188487682, 'min': -0.4666666666666667, 'max': 0.9833333333333333} |
| quadrants | [0, 2, 0, 0, 36, 0, 0, 2, 0] | [0, 0, 0, 0, 39, 0, 0, 1, 0] | 0.125 | 0.125 | {'count': 40, 'null_count': 0, 'mean': 0.7291666666666666, 'median': 0.7583333333333333, 'min': 0.23333333333333334, 'max': 0.9833333333333333} |
| offset_left_right_40 | [12, 7, 10, 6, 1, 3, 0, 1, 0] | [25, 2, 12, 0, 0, 1, 0, 0, 0] | 0.55 | 0.425 | {'count': 40, 'null_count': 5, 'mean': 0.724931029897884, 'median': 0.8432740427115678, 'min': -0.2, 'max': 1.0} |

| Fold | Held-out IDs | Selected MSE | LR / Quadrants / Offset | Non-tied train pairs | Final train pair loss |
|---|---|---:|---|---:|---:|
| 0 | [55167, 56288, 57149, 57672, 58393, 59386, 60102, 60770] | 0.0426284982823 | {'left_right': 0.04391328920610249, 'quadrants': 0.04322901205159724, 'offset_left_right_40': 0.040743193589150906} | 3249 | 0.0695690337274 |
| 1 | [55299, 56344, 57232, 58029, 58539, 59598, 60347, 60823] | 0.0216294735049 | {'left_right': 0.01985968730878085, 'quadrants': 0.02106516424100846, 'offset_left_right_40': 0.023963568964973092} | 3195 | 0.0729809935827 |
| 2 | [55528, 56350, 57238, 58111, 58636, 59635, 60363, 60835] | 0.0405451437691 | {'left_right': 0.04147959453985095, 'quadrants': 0.034383751335553825, 'offset_left_right_40': 0.045772085431963205} | 3159 | 0.07271303974 |
| 3 | [55950, 56545, 57244, 58350, 58705, 59920, 60449, 60855] | 0.0390058356958 | {'left_right': 0.039306428050622344, 'quadrants': 0.03418327448889613, 'offset_left_right_40': 0.04352780454792082} | 3204 | 0.07471857552 |
| 4 | [56127, 57027, 57597, 58384, 59044, 60052, 60507, 60886] | 0.0272818544763 | {'left_right': 0.030252126860432327, 'quadrants': 0.02457772701745853, 'offset_left_right_40': 0.02701570955105126} | 3285 | 0.0765042268692 |

Prediction/oracle minimum tie episodes: {'spatial_pool': (0, 12), 'left_right': (0, 5), 'quadrants': (0, 0), 'offset_left_right_40': (0, 7)}.

## Comparisons

{'rank30_over_rank28': 0.9665639197566726, 'rank28_over_probe28': 0.9886940383940429, 'rank30_over_probe30': 0.965223273628518}

Exact score/oracle ties choose first lexicographic boundary. Spearman uses average ranks; constant cases are null. All OOF outputs were frozen before reference evaluation. No alternate threshold, weighting, abstention, epoch, fold or model selection.

Stop and report the literal result. No larger model, spatial/image features, continuous boundaries or fresh experiment is authorized by this diagnostic.
