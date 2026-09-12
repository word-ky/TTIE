# T015 verified frozen cross-basis routing report

Qualification: 4/10 PASS; qualified=False; failed=['beyond_best_fixed', 'beyond_discrete', 'beyond_fixed16', 'offset_noninferiority', 'aligned_noninferiority', 'oracle_regret'].

40 fresh images x 6 primary conditions. spatial_pool includes left_right, quadrants and offset_left_right_40 (120 inputs). Aligned heterogeneous contains only left_right and quadrants (80 inputs). No source fitting or score calibration.

| Clause | Observed MSE | Required upper MSE | Pass |
|---|---:|---:|---|
| clean_mean | 0.0002601199375931174 | 0.003 | True |
| clean_p95 | 1.418400788679638e-05 | 0.005 | True |
| homogeneous_dark | 0.03103965624468401 | 0.04857287133112549 | True |
| homogeneous_bright | 0.01498076472198591 | 0.02407878587488085 | True |
| beyond_best_fixed | 0.0378063820884563 | 0.03399226653572017 | False |
| beyond_discrete | 0.0378063820884563 | 0.03479662535168851 | False |
| beyond_fixed16 | 0.0378063820884563 | 0.03474658093551019 | False |
| offset_noninferiority | 0.04196605016477406 | 0.04103178131813184 | False |
| aligned_noninferiority | 0.03572654805029742 | 0.03251233690575464 | False |
| oracle_regret | 0.0378063820884563 | 0.03608274542319123 | False |

| Ratio/value | Observed |
|---|---:|
| clean_mean | 0.00026011993759311736 |
| clean_p95 | 1.4184007886796383e-05 |
| dark_ratio | 0.3834196586784088 |
| bright_ratio | 0.37329369013445 |
| best_fixed_ratio | 1.0788392291307287 |
| discrete_ratio | 1.0321708677502732 |
| fixed16_ratio | 1.0336574712399433 |
| offset_bilinear_ratio | 1.0329970892024536 |
| aligned_region2_ratio | 1.1098498897633413 |
| oracle_regret_ratio | 1.1001574499751097 |

Evaluation-only best fixed spatial basis: region2_ttt_energy_sobolev

Oracle diagnostics: {'spatial_over_best_fixed': 0.980622572846402, 'beats_best_fixed_3pct': False, 'offset_over_bilinear': 0.9710560142767047}

The reference oracle selects among the three already-selected basis outputs. It does not select other checkpoints or change trajectories.

## Full method MSE table

| Method | clean | homogeneous_dark | homogeneous_bright | left_right | quadrants | offset_left_right_40 | heterogeneous | spatial_pool |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| identity | 0 | 0.0809547855519 | 0.0401313097915 | 0.0606666384265 | 0.0609545382205 | 0.056187466858 | 0.0608105883235 | 0.059269547835 |
| region2_direct | 0.000710774445906 | 0.0544074087404 | 0.0188588481396 | 0.0380024528829 | 0.0365370839252 | 0.0416490658186 | 0.0372697684041 | 0.0387295342089 |
| region2_discrete_projected | 0.000243814568967 | 0.0468419930432 | 0.0204168615397 | 0.0352110357722 | 0.033041987638 | 0.0416310566477 | 0.0341265117051 | 0.036628026686 |
| region2_ttt_projected | 0.0002479245828 | 0.0466090589296 | 0.0204619184951 | 0.034371916228 | 0.0328168975422 | 0.042184139695 | 0.0335944068851 | 0.0364576511551 |
| fixed_step_source | 0.000267344451277 | 0.0467032463057 | 0.0206056253635 | 0.0345093359472 | 0.0329585141386 | 0.0422581949737 | 0.0337339250429 | 0.0365753483532 |
| global_ttt_energy_sobolev | 0.000238258362515 | 0.0275996564073 | 0.0122257987678 | 0.0556364676915 | 0.0537501111627 | 0.0547769188415 | 0.0546932894271 | 0.0547211658986 |
| bilinear2_ttt_energy_sobolev | 0.000257908409549 | 0.0435962864198 | 0.0171191518719 | 0.0394196741749 | 0.0429767536698 | 0.0406255260576 | 0.0411982139223 | 0.0410073179674 |
| region2_ttt_energy_sobolev | 0.000288522831397 | 0.0452417620691 | 0.018624962325 | 0.0333970155101 | 0.0309838496498 | 0.0407498560846 | 0.03219043258 | 0.0350435737482 |
| routed_ttt_energy_sobolev | 0.000260119937593 | 0.0310396562447 | 0.014980764722 | 0.0377003192436 | 0.033752776857 | 0.0419660501648 | 0.0357265480503 | 0.0378063820885 |
| oracle_best_basis | 0.000225267509086 | 0.0275780358003 | 0.0117712200503 | 0.0330589110497 | 0.0305849858909 | 0.0394496614113 | 0.0318219484703 | 0.0343645194507 |

## Basis counts and oracle disagreement

Counts below are ordered global / bilinear2 / region2. Exact reference-MSE ties use the same order; count disagreement can include equal-output ties, so absolute MSE regret is also reported.

| Group | N | Routed counts | Oracle counts | Disagreements | Disagreement fraction |
|---|---:|---|---|---:|---:|
| all | 240 | [97, 60, 83] | [127, 27, 86] | 84 | 0.35 |
| clean | 40 | [38, 1, 1] | [39, 1, 0] | 2 | 0.05 |
| homogeneous_dark | 40 | [32, 5, 3] | [39, 1, 0] | 8 | 0.2 |
| homogeneous_bright | 40 | [12, 12, 16] | [33, 6, 1] | 23 | 0.575 |
| left_right | 40 | [6, 18, 16] | [5, 1, 34] | 25 | 0.625 |
| quadrants | 40 | [1, 10, 29] | [3, 0, 37] | 12 | 0.3 |
| offset_left_right_40 | 40 | [8, 14, 18] | [8, 18, 14] | 14 | 0.35 |
| spatial_pool | 120 | [15, 42, 63] | [16, 19, 85] | 51 | 0.425 |
| heterogeneous | 80 | [7, 28, 45] | [8, 1, 71] | 37 | 0.4625 |

## Winner-runner-up energy margins

All raw scores, selected bases and per-input margins are preserved in routing_diagnostics.json. These are unadjusted frozen energy scores; no offsets or normalization are applied.

| Group | N | Exact zero | Minimum | P10 | P25 | Median | P75 | P90 | P95 | Maximum | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 240 | 58 | 0 | 0 | 4.76837158203e-07 | 0.0358854532242 | 0.10357773304 | 0.212831068039 | 0.311548542976 | 0.72430229187 | 0.0787811676661 |
| clean | 40 | 38 | 0 | 0 | 0 | 0 | 0 | 0 | 0.000212252140045 | 0.015202999115 | 0.000486201047897 |
| homogeneous_dark | 40 | 9 | 0 | 0 | 4.76837158203e-07 | 0.0416015386581 | 0.206867456436 | 0.389429235458 | 0.457029521465 | 0.608545303345 | 0.125780928135 |
| homogeneous_bright | 40 | 4 | 0 | 0.000555324554443 | 0.0155860185623 | 0.0562121868134 | 0.108586668968 | 0.201965904236 | 0.32850549221 | 0.589151382446 | 0.0956467628479 |
| left_right | 40 | 2 | 0 | 0.00756986141205 | 0.0208687186241 | 0.0340242385864 | 0.0911980867386 | 0.14390847683 | 0.161295950413 | 0.311252593994 | 0.0630473196507 |
| quadrants | 40 | 0 | 0.00501585006714 | 0.0234809398651 | 0.0511330366135 | 0.0898152589798 | 0.149560570717 | 0.2137758255 | 0.262179303169 | 0.72430229187 | 0.117537945509 |
| offset_left_right_40 | 40 | 5 | 0 | 0 | 0.0110754370689 | 0.0406200885773 | 0.0792766809464 | 0.20606842041 | 0.259962546825 | 0.324862241745 | 0.0701878488064 |
| spatial_pool | 120 | 7 | 0 | 0.00728018283844 | 0.0234536528587 | 0.0555429458618 | 0.11175942421 | 0.202349948883 | 0.258766067028 | 0.72430229187 | 0.0835910379887 |
| heterogeneous | 80 | 2 | 0 | 0.0115681409836 | 0.0277714133263 | 0.0592249631882 | 0.126063704491 | 0.191235947609 | 0.21837157011 | 0.72430229187 | 0.0902926325798 |

## Regret conditioned on the routed basis

The MSE ratio is a ratio of group means, not the mean of per-input ratios. A zero oracle denominator remains null, with absolute regret and zero-oracle counts exposed.

| Group | Routed basis | N | Routed MSE | Oracle MSE | Mean absolute excess MSE | Ratio of means | Zero oracle | Positive excess with zero oracle |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | global_ttt_energy_sobolev | 97 | 0.0193512695193 | 0.0181408768649 | 0.0012103926544 | 1.06672183839 | 38 | 0 |
| all | bilinear2_ttt_energy_sobolev | 60 | 0.0320625860632 | 0.0246113848484 | 0.00745120121477 | 1.30275424405 | 0 | 0 |
| all | region2_ttt_energy_sobolev | 83 | 0.0311705925255 | 0.0297635556826 | 0.00140703684287 | 1.0472738156 | 0 | 0 |
| clean | global_ttt_energy_sobolev | 38 | 0 | 0 | 0 | null | 38 | 0 |
| clean | bilinear2_ttt_energy_sobolev | 1 | 0.010121117346 | 0.00881548132747 | 0.00130563601851 | 1.14810717305 | 0 | 0 |
| clean | region2_ttt_energy_sobolev | 1 | 0.000283680157736 | 0.000195219035959 | 8.84611217771e-05 | 1.45313778619 | 0 | 0 |
| homogeneous_dark | global_ttt_energy_sobolev | 32 | 0.0296719832841 | 0.0296719832841 | 0 | 1 | 0 | 0 |
| homogeneous_dark | bilinear2_ttt_energy_sobolev | 5 | 0.0497517114505 | 0.0221998682246 | 0.027551843226 | 2.24108138604 | 0 | 0 |
| homogeneous_dark | region2_ttt_energy_sobolev | 3 | 0.0144414091483 | 0.0142062085991 | 0.000235200549165 | 1.01655618017 | 0 | 0 |
| homogeneous_bright | global_ttt_energy_sobolev | 12 | 0.0178783852801 | 0.0178783852801 | 0 | 1 | 0 | 0 |
| homogeneous_bright | bilinear2_ttt_energy_sobolev | 12 | 0.0125115039991 | 0.00809179948798 | 0.00441970451114 | 1.54619550543 | 0 | 0 |
| homogeneous_bright | region2_ttt_energy_sobolev | 16 | 0.0146594948455 | 0.00995041154965 | 0.00470908329589 | 1.47325512843 | 0 | 0 |
| left_right | global_ttt_energy_sobolev | 6 | 0.0430129980668 | 0.0363550813248 | 0.00665791674207 | 1.18313579559 | 0 | 0 |
| left_right | bilinear2_ttt_energy_sobolev | 18 | 0.0303556626766 | 0.0228612081976 | 0.00749445447905 | 1.32782407711 | 0 | 0 |
| left_right | region2_ttt_energy_sobolev | 16 | 0.0439708033227 | 0.0432952629053 | 0.000675540417433 | 1.01560310233 | 0 | 0 |
| quadrants | global_ttt_energy_sobolev | 1 | 0.0391028113663 | 0.0391028113663 | 0 | 1 | 0 | 0 |
| quadrants | bilinear2_ttt_energy_sobolev | 10 | 0.0477059789002 | 0.0364307826385 | 0.0112751962617 | 1.3094964051 | 0 | 0 |
| quadrants | region2_ttt_energy_sobolev | 29 | 0.028756843928 | 0.0282754757891 | 0.000481368138872 | 1.01702422773 | 0 | 0 |
| offset_left_right_40 | global_ttt_energy_sobolev | 8 | 0.0519810318947 | 0.0422984585166 | 0.00968257337809 | 1.22891078582 | 0 | 0 |
| offset_left_right_40 | bilinear2_ttt_energy_sobolev | 14 | 0.0350911231445 | 0.0345683643328 | 0.000522758811712 | 1.01512246303 | 0 | 0 |
| offset_left_right_40 | region2_ttt_energy_sobolev | 18 | 0.0428621126339 | 0.041980093759 | 0.000882018874917 | 1.02101040746 | 0 | 0 |
| spatial_pool | global_ttt_energy_sobolev | 15 | 0.0475352703283 | 0.0397080644965 | 0.00782720583181 | 1.19711879516 | 0 | 0 |
| spatial_pool | bilinear2_ttt_energy_sobolev | 42 | 0.0360651771715 | 0.0299944446333 | 0.00607073253819 | 1.2023952306 | 0 | 0 |
| spatial_pool | region2_ttt_energy_sobolev | 63 | 0.0366507834045 | 0.0360056300322 | 0.000645153372297 | 1.0179181248 | 0 | 0 |
| heterogeneous | global_ttt_energy_sobolev | 7 | 0.0424543999668 | 0.0367476141879 | 0.00570678577891 | 1.15529676974 | 0 | 0 |
| heterogeneous | bilinear2_ttt_energy_sobolev | 28 | 0.036552204185 | 0.0277074847836 | 0.00884471940143 | 1.31921769408 | 0 | 0 |
| heterogeneous | region2_ttt_energy_sobolev | 45 | 0.0341662517128 | 0.0336158445415 | 0.000550407171249 | 1.01637344469 | 0 | 0 |

## Frozen per-basis trajectory diagnostics

| Basis | Group | Selected-step counts | No-active | Projected / updates | Movable boundary / coordinates |
|---|---|---|---:|---|---|
| global_ttt_energy_sobolev | all | {'0': 54, '3': 1, '5': 1, '6': 2, '7': 2, '8': 1, '9': 1, '10': 2, '11': 1, '12': 37, '13': 6, '15': 1, '16': 1, '17': 3, '18': 2, '19': 6, '20': 1, '21': 1, '22': 4, '23': 2, '24': 2, '25': 6, '26': 7, '27': 6, '28': 8, '29': 2, '30': 5, '31': 3, '32': 7, '33': 5, '34': 6, '35': 10, '36': 5, '37': 5, '38': 10, '39': 6, '40': 18} | 54 | 4281/7440 (0.575403225806) | 199/372 (0.534946236559) |
| global_ttt_energy_sobolev | clean | {'0': 38, '5': 1, '19': 1} | 38 | 40/80 (0.5) | 1/4 (0.25) |
| global_ttt_energy_sobolev | homogeneous_dark | {'0': 5, '10': 1, '12': 25, '13': 5, '26': 1, '27': 1, '38': 1, '40': 1} | 5 | 1085/1400 (0.775) | 66/70 (0.942857142857) |
| global_ttt_energy_sobolev | homogeneous_bright | {'0': 4, '6': 2, '7': 2, '8': 1, '9': 1, '12': 1, '18': 1, '20': 1, '22': 2, '23': 1, '25': 1, '26': 1, '27': 1, '28': 2, '30': 2, '32': 2, '35': 2, '36': 1, '37': 1, '38': 6, '40': 5} | 4 | 439/1440 (0.304861111111) | 19/72 (0.263888888889) |
| global_ttt_energy_sobolev | left_right | {'0': 2, '10': 1, '11': 1, '12': 5, '16': 1, '17': 1, '19': 1, '22': 2, '25': 2, '26': 1, '27': 2, '28': 1, '29': 2, '30': 2, '31': 1, '32': 1, '33': 1, '34': 3, '35': 2, '37': 1, '38': 1, '39': 3, '40': 3} | 2 | 1044/1520 (0.686842105263) | 42/76 (0.552631578947) |
| global_ttt_energy_sobolev | quadrants | {'12': 2, '13': 1, '15': 1, '17': 1, '19': 2, '21': 1, '24': 2, '25': 1, '27': 1, '28': 3, '30': 1, '31': 1, '32': 3, '33': 1, '34': 2, '35': 5, '36': 3, '38': 2, '39': 1, '40': 6} | 0 | 927/1600 (0.579375) | 38/80 (0.475) |
| global_ttt_energy_sobolev | offset_left_right_40 | {'0': 5, '3': 1, '12': 4, '17': 1, '18': 1, '19': 2, '23': 1, '25': 2, '26': 4, '27': 1, '28': 2, '31': 1, '32': 1, '33': 3, '34': 1, '35': 1, '36': 1, '37': 3, '39': 2, '40': 3} | 5 | 746/1400 (0.532857142857) | 33/70 (0.471428571429) |
| bilinear2_ttt_energy_sobolev | all | {'0': 54, '12': 34, '13': 10, '15': 2, '16': 1, '17': 1, '19': 3, '20': 2, '22': 4, '23': 1, '25': 5, '26': 1, '27': 4, '28': 2, '29': 10, '30': 5, '31': 3, '32': 7, '33': 8, '34': 5, '35': 13, '36': 2, '37': 6, '38': 8, '39': 7, '40': 42} | 54 | 6932/7440 (0.931720430108) | 619/982 (0.630346232179) |
| bilinear2_ttt_energy_sobolev | clean | {'0': 38, '12': 1, '39': 1} | 38 | 80/80 (1) | 2/6 (0.333333333333) |
| bilinear2_ttt_energy_sobolev | homogeneous_dark | {'0': 5, '12': 18, '13': 5, '15': 1, '16': 1, '17': 1, '19': 1, '29': 1, '31': 1, '33': 3, '35': 1, '40': 2} | 5 | 1328/1400 (0.948571428571) | 183/192 (0.953125) |
| bilinear2_ttt_energy_sobolev | homogeneous_bright | {'0': 4, '15': 1, '20': 1, '25': 1, '27': 1, '30': 1, '32': 1, '33': 1, '34': 1, '35': 4, '37': 1, '38': 3, '39': 6, '40': 14} | 4 | 1168/1440 (0.811111111111) | 70/214 (0.327102803738) |
| bilinear2_ttt_energy_sobolev | left_right | {'0': 2, '12': 6, '13': 1, '20': 1, '22': 2, '28': 1, '29': 4, '30': 2, '32': 1, '33': 3, '34': 2, '35': 4, '38': 2, '40': 9} | 2 | 1448/1520 (0.952631578947) | 126/198 (0.636363636364) |
| bilinear2_ttt_energy_sobolev | quadrants | {'12': 6, '13': 3, '19': 1, '25': 3, '27': 2, '28': 1, '29': 3, '30': 1, '32': 3, '33': 1, '35': 2, '36': 1, '37': 2, '38': 1, '40': 10} | 0 | 1529/1600 (0.955625) | 139/204 (0.68137254902) |
| bilinear2_ttt_energy_sobolev | offset_left_right_40 | {'0': 5, '12': 3, '13': 1, '19': 1, '22': 2, '23': 1, '25': 1, '26': 1, '27': 1, '29': 2, '30': 1, '31': 2, '32': 2, '34': 2, '35': 2, '36': 1, '37': 3, '38': 2, '40': 7} | 5 | 1379/1400 (0.985) | 99/168 (0.589285714286) |
| region2_ttt_energy_sobolev | all | {'0': 54, '12': 32, '13': 11, '14': 2, '15': 1, '18': 1, '19': 1, '20': 2, '21': 3, '22': 1, '23': 1, '24': 4, '25': 2, '27': 4, '28': 3, '29': 5, '30': 3, '31': 4, '32': 6, '33': 6, '34': 2, '35': 10, '36': 15, '37': 3, '38': 10, '39': 4, '40': 50} | 54 | 6972/7440 (0.937096774194) | 605/982 (0.616089613035) |
| region2_ttt_energy_sobolev | clean | {'0': 38, '12': 1, '38': 1} | 38 | 80/80 (1) | 2/6 (0.333333333333) |
| region2_ttt_energy_sobolev | homogeneous_dark | {'0': 5, '12': 15, '13': 7, '14': 2, '15': 1, '18': 1, '21': 1, '31': 1, '32': 1, '33': 1, '35': 1, '36': 1, '38': 1, '40': 2} | 5 | 1328/1400 (0.948571428571) | 181/192 (0.942708333333) |
| region2_ttt_energy_sobolev | homogeneous_bright | {'0': 4, '20': 1, '24': 1, '25': 1, '27': 2, '28': 1, '29': 1, '31': 2, '32': 2, '33': 2, '35': 2, '36': 2, '37': 1, '38': 2, '40': 16} | 4 | 1195/1440 (0.829861111111) | 65/214 (0.303738317757) |
| region2_ttt_energy_sobolev | left_right | {'0': 2, '12': 7, '13': 1, '23': 1, '24': 1, '28': 1, '29': 1, '30': 1, '31': 1, '32': 1, '33': 1, '34': 2, '35': 2, '36': 1, '37': 1, '38': 2, '39': 2, '40': 12} | 2 | 1449/1520 (0.953289473684) | 129/198 (0.651515151515) |
| region2_ttt_energy_sobolev | quadrants | {'12': 5, '13': 2, '20': 1, '22': 1, '24': 1, '25': 1, '27': 2, '29': 2, '30': 1, '32': 1, '33': 1, '35': 2, '36': 6, '37': 1, '39': 1, '40': 12} | 0 | 1528/1600 (0.955) | 134/204 (0.656862745098) |
| region2_ttt_energy_sobolev | offset_left_right_40 | {'0': 5, '12': 4, '13': 1, '19': 1, '21': 2, '24': 1, '28': 1, '29': 1, '30': 1, '32': 1, '33': 1, '35': 3, '36': 5, '38': 4, '39': 1, '40': 8} | 5 | 1392/1400 (0.994285714286) | 94/168 (0.559523809524) |

## Verification receipts

```json
{
  "remote": {
    "task": "T015",
    "all_hashes_verified": true,
    "all_stored_pixel_mse_exact": true,
    "frozen_scores_checkpoints_routes_oracles_exact": true,
    "features_projections_summary_exact": true,
    "frozen_receipt_assets_manifest_verified": true,
    "no_refitting": true,
    "offset_is_primary": true,
    "qualified": false,
    "failed": [
      "beyond_best_fixed",
      "beyond_discrete",
      "beyond_fixed16",
      "offset_noninferiority",
      "aligned_noninferiority",
      "oracle_regret"
    ],
    "hashed_files": 4560,
    "large_image_bytes": 54981354720,
    "evaluation_inputs": 240,
    "energy_checkpoints": 23040,
    "energy_updates": 22320,
    "semantic_checkpoints": 6002,
    "inactive_region_checks": 10589,
    "no_active_inputs": 54
  },
  "local": {
    "task": "T015",
    "local_small_hashes_verified": 2400,
    "evaluation_rows": 2400,
    "summary_max_abs_difference": 6.938893903907228e-18,
    "frozen_head_manifest_match": true,
    "all_routes_margin_counts_regret_recomputed": true,
    "qualified": false,
    "failed": [
      "beyond_best_fixed",
      "beyond_discrete",
      "beyond_fixed16",
      "offset_noninferiority",
      "aligned_noninferiority",
      "oracle_regret"
    ],
    "large_image_bytes_preserved_remotely": 54981354720,
    "archive_sha256": "7b2ff9d718f946300c24872c60d132a34e9f0c3b80be2b308a28b4c3a159d9a6"
  }
}
```
