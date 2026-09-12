# T011 offline projection diagnostics

Frozen convention: commit d2929e6; exact pre/post raw change; final physical boundary absolute tolerance 1e-6. Inactive fast coordinates are excluded from boundary occupancy. Overall includes the separately reported offset stress; qualification remains unchanged.

| Condition | Method | Hit updates / updates | EV update hit % | Gamma update hit % | Boundary active coords / total |
|---|---|---:|---:|---:|---:|
| clean | global_ttt_projected | 153/348 | 28.1609 | 27.0115 | 7/18 |
| clean | all_projected_methods | 719/1065 | 54.9296 | 53.8028 | 30/90 |
| clean | bilinear2_ttt_projected | 360/360 | 100.0000 | 100.0000 | 14/24 |
| clean | region2_ttt_projected_1step | 0/9 | 0.0000 | 0.0000 | 0/24 |
| clean | region2_ttt_projected | 206/348 | 36.4943 | 34.1954 | 9/24 |
| overall | global_ttt_projected | 4054/6529 | 24.9196 | 51.4014 | 189/404 |
| overall | all_projected_methods | 14601/19634 | 60.8332 | 63.9910 | 1458/3716 |
| overall | bilinear2_ttt_projected | 6526/6929 | 93.5489 | 88.9883 | 680/1104 |
| overall | region2_ttt_projected_1step | 7/202 | 3.4653 | 0.0000 | 7/1104 |
| overall | region2_ttt_projected | 4014/5974 | 64.0777 | 50.9207 | 582/1104 |
| homogeneous_dark | global_ttt_projected | 618/925 | 66.8108 | 50.5946 | 46/72 |
| homogeneous_dark | all_projected_methods | 2207/2955 | 74.6870 | 63.9932 | 328/702 |
| homogeneous_dark | bilinear2_ttt_projected | 968/1075 | 90.0465 | 86.2326 | 144/210 |
| homogeneous_dark | region2_ttt_projected_1step | 0/36 | 0.0000 | 0.0000 | 0/210 |
| homogeneous_dark | region2_ttt_projected | 621/919 | 67.5734 | 53.9717 | 138/210 |
| homogeneous_bright | global_ttt_projected | 596/1172 | 43.6860 | 21.6724 | 31/78 |
| homogeneous_bright | all_projected_methods | 2310/3571 | 60.3192 | 45.9255 | 221/774 |
| homogeneous_bright | bilinear2_ttt_projected | 991/1194 | 81.0720 | 75.0419 | 98/232 |
| homogeneous_bright | region2_ttt_projected_1step | 1/39 | 2.5641 | 0.0000 | 1/232 |
| homogeneous_bright | region2_ttt_projected | 722/1166 | 57.7187 | 42.0240 | 91/232 |
| left_right | global_ttt_projected | 915/1378 | 6.8215 | 65.6749 | 35/80 |
| left_right | all_projected_methods | 3069/3993 | 56.0731 | 70.0726 | 307/758 |
| left_right | bilinear2_ttt_projected | 1369/1412 | 96.3173 | 89.1643 | 149/226 |
| left_right | region2_ttt_projected_1step | 0/40 | 0.0000 | 0.0000 | 0/226 |
| left_right | region2_ttt_projected | 785/1163 | 67.4979 | 54.5142 | 123/226 |
| quadrants | global_ttt_projected | 973/1456 | 4.3956 | 64.9038 | 36/80 |
| quadrants | all_projected_methods | 3268/4202 | 55.8544 | 71.8229 | 317/764 |
| quadrants | bilinear2_ttt_projected | 1534/1577 | 96.5124 | 92.6443 | 163/228 |
| quadrants | region2_ttt_projected_1step | 0/40 | 0.0000 | 0.0000 | 0/228 |
| quadrants | region2_ttt_projected | 761/1129 | 67.4048 | 54.2073 | 118/228 |
| offset_left_right_40 | global_ttt_projected | 799/1250 | 19.2800 | 55.2000 | 34/76 |
| offset_left_right_40 | all_projected_methods | 3028/3848 | 62.6819 | 68.7110 | 255/628 |
| offset_left_right_40 | bilinear2_ttt_projected | 1304/1311 | 99.4661 | 96.3387 | 112/184 |
| offset_left_right_40 | region2_ttt_projected_1step | 6/38 | 15.7895 | 0.0000 | 6/184 |
| offset_left_right_40 | region2_ttt_projected | 919/1249 | 68.9351 | 55.3243 | 103/184 |

Coordinate-wise rates for all/active EV and gamma, with integer denominators, are in projection_diagnostics.json.

| Condition | Agree dark | Agree bright | Conflict | No active |
|---|---:|---:|---:|---:|
| clean | 2 | 7 | 0 | 31 |
| overall | 48 | 74 | 80 | 38 |
| homogeneous_dark | 36 | 0 | 0 | 4 |
| homogeneous_bright | 0 | 39 | 0 | 1 |
| left_right | 5 | 6 | 29 | 0 |
| quadrants | 2 | 6 | 32 | 0 |
| offset_left_right_40 | 3 | 16 | 19 | 2 |

| Condition | Mean one-step minus full MSE | One-step better / inputs | Equal |
|---|---:|---:|---:|
| clean | -0.0009287848 | 9/40 | 31 |
| homogeneous_dark | 0.0319969297 | 0/40 | 4 |
| homogeneous_bright | 0.0144044916 | 3/40 | 1 |
| left_right | 0.0248364126 | 0/40 | 0 |
| quadrants | 0.0253294375 | 1/40 | 0 |
| heterogeneous | 0.0250829251 | 1/80 | 0 |
| offset_left_right_40 | 0.0107353182 | 3/40 | 2 |

Negative MSE difference favors one-step. This is a diagnostic only; no output selection or regeneration. All 240 paired differences are in JSON.
