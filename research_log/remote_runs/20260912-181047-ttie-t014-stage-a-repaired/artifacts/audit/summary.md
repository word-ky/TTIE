# T014 Stage A

Pass: True; failed: []

| Value | Observed |
|---|---:|
| positive_cosine_fraction | 0.986486486486 |
| median_cosine | 0.936059070995 |
| clean_p95 | 0.000239434512332 |
| dark_ratio | 0.559133746831 |
| bright_ratio | 0.410865238191 |
| discrete_ratio | 0.926347033029 |
| fixed_step_ratio | 0.918656222033 |
| value_only_ratio | 0.834248302006 |

Oracle diagnostics (reference-only): {'oracle_regret_ratio': 1.044737586056045, 'oracle_over_discrete': 0.8866791483267122, 'oracle_over_fixed16': 0.8793176720108989, 'oracle_beats_discrete_5pct': True, 'oracle_beats_fixed16_5pct': True}

| Condition | Method | Mean MSE | P95 MSE | Mean finite PSNR |
|---|---|---:|---:|---:|
| clean | identity | 0 | 0 | null |
| clean | region2_direct | 8.986031171e-05 | 8.986031171e-05 | 27.45402084 |
| clean | region2_discrete_projected | 0.0002394345123 | 0.0002394345123 | 23.19783254 |
| clean | region2_ttt_projected | 0.0002049619565 | 0.0002049619565 | 23.87296746 |
| clean | fixed_step_source | 0.0002394345123 | 0.0002394345123 | 23.19783254 |
| clean | region2_ttt_energy_value_only | 0.0002394345123 | 0.0002394345123 | 23.19783254 |
| clean | global_ttt_energy_sobolev | 0.001447631512 | 0.001447631512 | 15.38311976 |
| clean | bilinear2_ttt_energy_sobolev | 0.0001469618641 | 0.0001469618641 | 25.31765352 |
| clean | region2_ttt_energy_sobolev | 0.0002394345123 | 0.0002394345123 | 23.19783254 |
| clean | oracle_best_sobolev_checkpoint | 9.198086284e-19 | 9.198086284e-19 | 167.3527253 |
| homogeneous_dark | identity | 0.08669466879 | 0.1340045065 | 11.0420049 |
| homogeneous_dark | region2_direct | 0.05929090157 | 0.1022316795 | 13.00772463 |
| homogeneous_dark | region2_discrete_projected | 0.05305672586 | 0.1022316795 | 14.18390696 |
| homogeneous_dark | region2_ttt_projected | 0.05448204884 | 0.1186956216 | 14.29946248 |
| homogeneous_dark | fixed_step_source | 0.05394582949 | 0.1133336883 | 14.32204601 |
| homogeneous_dark | region2_ttt_energy_value_only | 0.05537502004 | 0.1022316795 | 14.06540779 |
| homogeneous_dark | global_ttt_energy_sobolev | 0.03893020966 | 0.1022316795 | 15.99494586 |
| homogeneous_dark | bilinear2_ttt_energy_sobolev | 0.04769865396 | 0.1022316795 | 14.77394455 |
| homogeneous_dark | region2_ttt_energy_sobolev | 0.04847391499 | 0.1022316795 | 14.70057939 |
| homogeneous_dark | oracle_best_sobolev_checkpoint | 0.04832725869 | 0.1022316795 | 14.72158213 |
| homogeneous_bright | identity | 0.04533175612 | 0.07879954204 | 13.84191285 |
| homogeneous_bright | region2_direct | 0.01709948705 | 0.03810844421 | 19.0663937 |
| homogeneous_bright | region2_discrete_projected | 0.01983428255 | 0.03640404586 | 17.57067551 |
| homogeneous_bright | region2_ttt_projected | 0.01964650042 | 0.04039275441 | 17.89776747 |
| homogeneous_bright | fixed_step_source | 0.01930036573 | 0.04039275441 | 17.88509413 |
| homogeneous_bright | region2_ttt_energy_value_only | 0.02465986665 | 0.05528182592 | 17.38458798 |
| homogeneous_bright | global_ttt_energy_sobolev | 0.01329901883 | 0.03698109109 | 19.93492178 |
| homogeneous_bright | bilinear2_ttt_energy_sobolev | 0.01743614005 | 0.03663987592 | 18.47594469 |
| homogeneous_bright | region2_ttt_energy_sobolev | 0.01862524278 | 0.03486666474 | 18.12455121 |
| homogeneous_bright | oracle_best_sobolev_checkpoint | 0.01561174362 | 0.03395543285 | 19.72179282 |
| left_right | identity | 0.06436894229 | 0.106271299 | 12.24332961 |
| left_right | region2_direct | 0.03771977886 | 0.06303542592 | 14.68151061 |
| left_right | region2_discrete_projected | 0.03552068518 | 0.06792654134 | 15.17795674 |
| left_right | region2_ttt_projected | 0.03632055887 | 0.07353245392 | 15.17426107 |
| left_right | fixed_step_source | 0.03625753985 | 0.07364665195 | 15.19015374 |
| left_right | region2_ttt_energy_value_only | 0.04305589059 | 0.07748783343 | 14.29540697 |
| left_right | global_ttt_energy_sobolev | 0.06251936005 | 0.0926173076 | 12.31258003 |
| left_right | bilinear2_ttt_energy_sobolev | 0.0405556092 | 0.0660409119 | 14.33369501 |
| left_right | region2_ttt_energy_sobolev | 0.03364706715 | 0.05985518489 | 15.48179226 |
| left_right | oracle_best_sobolev_checkpoint | 0.03218347752 | 0.05872994456 | 15.69469094 |
| quadrants | identity | 0.0646023469 | 0.1018382162 | 12.25392423 |
| quadrants | region2_direct | 0.03885982339 | 0.06454078406 | 14.51851381 |
| quadrants | region2_discrete_projected | 0.03745278628 | 0.06551027745 | 14.78197854 |
| quadrants | region2_ttt_projected | 0.03793484406 | 0.07173230015 | 14.92151081 |
| quadrants | fixed_step_source | 0.0373268513 | 0.07016830929 | 15.01122824 |
| quadrants | region2_ttt_energy_value_only | 0.03797365255 | 0.07768846042 | 14.90229301 |
| quadrants | global_ttt_energy_sobolev | 0.06124047572 | 0.08563368507 | 12.42512322 |
| quadrants | bilinear2_ttt_energy_sobolev | 0.04732324835 | 0.07243423276 | 13.60268378 |
| quadrants | region2_ttt_energy_sobolev | 0.03395169163 | 0.06502220035 | 15.38998681 |
| quadrants | oracle_best_sobolev_checkpoint | 0.03252057801 | 0.06223291382 | 15.54816134 |
| heterogeneous | identity | 0.0644856446 | 0.106271299 | 12.24862692 |
| heterogeneous | region2_direct | 0.03828980112 | 0.06454078406 | 14.60001221 |
| heterogeneous | region2_discrete_projected | 0.03648673573 | 0.06792654134 | 14.97996764 |
| heterogeneous | region2_ttt_projected | 0.03712770146 | 0.07353245392 | 15.04788594 |
| heterogeneous | fixed_step_source | 0.03679219558 | 0.07364665195 | 15.10069099 |
| heterogeneous | region2_ttt_energy_value_only | 0.04051477157 | 0.07768310271 | 14.59884999 |
| heterogeneous | global_ttt_energy_sobolev | 0.06187991789 | 0.0926173076 | 12.36885162 |
| heterogeneous | bilinear2_ttt_energy_sobolev | 0.04393942878 | 0.07094431929 | 13.9681894 |
| heterogeneous | region2_ttt_energy_sobolev | 0.03379937939 | 0.06491436027 | 15.43588953 |
| heterogeneous | oracle_best_sobolev_checkpoint | 0.03235202776 | 0.06223291382 | 15.62142614 |
