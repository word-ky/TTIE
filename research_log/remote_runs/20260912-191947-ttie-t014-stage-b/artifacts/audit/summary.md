# T014 Stage B

Pass: True; failed: []

| Value | Observed |
|---|---:|
| clean_mean | 0.000253260094905 |
| clean_p95 | 0.00012560087489 |
| dark_ratio | 0.570950280202 |
| bright_ratio | 0.420419523415 |
| global_ratio | 0.562495165146 |
| direct_ratio | 0.846958576027 |
| discrete_ratio | 0.882753937687 |
| dark_region_ratio | 0.5315698754 |
| bright_region_ratio | 0.447642584744 |
| quadrant_bilinear_ratio | 0.714367874507 |
| fixed_step_ratio | 0.916876394123 |
| value_only_ratio | 0.808299725731 |

Oracle diagnostics (reference-only): {'oracle_regret_ratio': 1.0280402978838314, 'oracle_over_discrete': 0.858676395763497, 'oracle_over_fixed16': 0.8918681456457281, 'oracle_beats_discrete_5pct': True, 'oracle_beats_fixed16_5pct': True}

| Condition | Method | Mean MSE | P95 MSE | Mean finite PSNR |
|---|---|---:|---:|---:|
| clean | identity | 0 | 0 | null |
| clean | region2_direct | 0.0005988126213 | 0.0001407128875 | 21.12783579 |
| clean | region2_discrete_projected | 0.0002689122368 | 6.460359436e-05 | 24.56309714 |
| clean | region2_ttt_projected | 0.0002302443369 | 2.891662007e-06 | 31.3815465 |
| clean | fixed_step_source | 0.0002847380088 | 9.53061317e-06 | 28.35336741 |
| clean | region2_ttt_energy_value_only | 7.388209034e-05 | 7.873716186e-08 | 41.6621128 |
| clean | global_ttt_energy_sobolev | 1.847811236e-05 | 6.080442472e-06 | 35.62193428 |
| clean | bilinear2_ttt_energy_sobolev | 0.0002789075312 | 0.0001305246376 | 23.25762176 |
| clean | region2_ttt_energy_sobolev | 0.0002532600949 | 0.0001256008749 | 23.59057175 |
| clean | oracle_best_sobolev_checkpoint | 9.664549996e-18 | 4.518312382e-19 | 162.335123 |
| homogeneous_dark | identity | 0.0916296836 | 0.1470077097 | 10.72484217 |
| homogeneous_dark | region2_direct | 0.06179229836 | 0.1302573815 | 12.67193529 |
| homogeneous_dark | region2_discrete_projected | 0.05496923706 | 0.1302573815 | 13.7529327 |
| homogeneous_dark | region2_ttt_projected | 0.05330164967 | 0.1302855872 | 13.98886588 |
| homogeneous_dark | fixed_step_source | 0.05328872568 | 0.1302855872 | 13.97585553 |
| homogeneous_dark | region2_ttt_energy_value_only | 0.0642172684 | 0.130475042 | 13.16470017 |
| homogeneous_dark | global_ttt_energy_sobolev | 0.03488411648 | 0.09546529874 | 16.01410275 |
| homogeneous_dark | bilinear2_ttt_energy_sobolev | 0.0516901859 | 0.1217044957 | 14.11229478 |
| homogeneous_dark | region2_ttt_energy_sobolev | 0.05231599353 | 0.11410546 | 14.03331116 |
| homogeneous_dark | oracle_best_sobolev_checkpoint | 0.05182109344 | 0.11410546 | 14.08387897 |
| homogeneous_bright | identity | 0.04331596943 | 0.07246358879 | 13.9666474 |
| homogeneous_bright | region2_direct | 0.01950047821 | 0.05271657966 | 18.53079842 |
| homogeneous_bright | region2_discrete_projected | 0.02121840902 | 0.05269591808 | 17.63538551 |
| homogeneous_bright | region2_ttt_projected | 0.02083811371 | 0.05512110032 | 17.90563668 |
| homogeneous_bright | fixed_step_source | 0.02126160135 | 0.05507532302 | 17.79125491 |
| homogeneous_bright | region2_ttt_energy_value_only | 0.02607486901 | 0.06091352329 | 16.99495168 |
| homogeneous_bright | global_ttt_energy_sobolev | 0.01256055192 | 0.03811164778 | 20.82684506 |
| homogeneous_bright | bilinear2_ttt_energy_sobolev | 0.0170138184 | 0.04890906923 | 19.12595714 |
| homogeneous_bright | region2_ttt_energy_sobolev | 0.01821087922 | 0.0514146354 | 18.76899111 |
| homogeneous_bright | oracle_best_sobolev_checkpoint | 0.01638630072 | 0.05124522876 | 19.58584036 |
| left_right | identity | 0.06753394138 | 0.1081845179 | 11.99804322 |
| left_right | region2_direct | 0.04093280572 | 0.08257512785 | 14.71775344 |
| left_right | region2_discrete_projected | 0.03931737586 | 0.0827045951 | 15.04883119 |
| left_right | region2_ttt_projected | 0.03750016952 | 0.0830162324 | 15.57346427 |
| left_right | fixed_step_source | 0.03811920226 | 0.0830162324 | 15.48204339 |
| left_right | region2_ttt_energy_value_only | 0.04502150846 | 0.08330984041 | 14.20941055 |
| left_right | global_ttt_energy_sobolev | 0.06225704015 | 0.1102552127 | 12.62950202 |
| left_right | bilinear2_ttt_energy_sobolev | 0.04143146838 | 0.08257947639 | 14.50599339 |
| left_right | region2_ttt_energy_sobolev | 0.03512309752 | 0.08257162161 | 15.66888027 |
| left_right | oracle_best_sobolev_checkpoint | 0.03440616763 | 0.08251282983 | 15.9210693 |
| quadrants | identity | 0.06672581183 | 0.09589747041 | 12.00943931 |
| quadrants | region2_direct | 0.03901922647 | 0.08095841445 | 14.64844402 |
| quadrants | region2_discrete_projected | 0.03739262954 | 0.08201898038 | 15.03866526 |
| quadrants | region2_ttt_projected | 0.03530827662 | 0.08159404583 | 15.44991209 |
| quadrants | fixed_step_source | 0.03573596488 | 0.08191372678 | 15.36593075 |
| quadrants | region2_ttt_energy_value_only | 0.03875441918 | 0.08230904341 | 14.98739601 |
| quadrants | global_ttt_energy_sobolev | 0.05812810007 | 0.09230485745 | 12.6928113 |
| quadrants | bilinear2_ttt_energy_sobolev | 0.04562489856 | 0.08552211672 | 13.81987306 |
| quadrants | region2_ttt_energy_sobolev | 0.03259296181 | 0.07764885277 | 15.76691005 |
| quadrants | oracle_best_sobolev_checkpoint | 0.03146290333 | 0.0772258848 | 15.92775809 |
| offset_left_right_40 | identity | 0.06283630771 | 0.1021154638 | 12.30128253 |
| offset_left_right_40 | region2_direct | 0.04692947036 | 0.0803801652 | 13.86285056 |
| offset_left_right_40 | region2_discrete_projected | 0.04583390239 | 0.08267526925 | 13.95658461 |
| offset_left_right_40 | region2_ttt_projected | 0.04498502119 | 0.08246439025 | 14.13922878 |
| offset_left_right_40 | fixed_step_source | 0.04559465428 | 0.08247267865 | 14.08160632 |
| offset_left_right_40 | region2_ttt_energy_value_only | 0.05262492071 | 0.09368670806 | 13.27976073 |
| offset_left_right_40 | global_ttt_energy_sobolev | 0.06021506626 | 0.1114207968 | 12.90042286 |
| offset_left_right_40 | bilinear2_ttt_energy_sobolev | 0.04450655326 | 0.07975632474 | 14.09774902 |
| offset_left_right_40 | region2_ttt_energy_sobolev | 0.04476872589 | 0.0793345537 | 14.05103551 |
| offset_left_right_40 | oracle_best_sobolev_checkpoint | 0.04294313989 | 0.07920592129 | 14.27831964 |
| heterogeneous | identity | 0.0671298766 | 0.1079897482 | 12.00374127 |
| heterogeneous | region2_direct | 0.03997601609 | 0.0819269184 | 14.68309873 |
| heterogeneous | region2_discrete_projected | 0.0383550027 | 0.08204649277 | 15.04374822 |
| heterogeneous | region2_ttt_projected | 0.03640422307 | 0.08199661002 | 15.51168818 |
| heterogeneous | fixed_step_source | 0.03692758357 | 0.08191372678 | 15.42398707 |
| heterogeneous | region2_ttt_energy_value_only | 0.04188796382 | 0.08231937028 | 14.59840328 |
| heterogeneous | global_ttt_energy_sobolev | 0.06019257011 | 0.1042927172 | 12.66115666 |
| heterogeneous | bilinear2_ttt_energy_sobolev | 0.04352818347 | 0.08552211672 | 14.16293323 |
| heterogeneous | region2_ttt_energy_sobolev | 0.03385802967 | 0.08189259768 | 15.71789516 |
| heterogeneous | oracle_best_sobolev_checkpoint | 0.03293453548 | 0.08164861649 | 15.9244137 |
