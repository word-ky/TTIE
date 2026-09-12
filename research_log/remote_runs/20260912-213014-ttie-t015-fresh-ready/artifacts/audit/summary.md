# T015 frozen cross-basis routing

Qualified: False; failed: ['beyond_best_fixed', 'beyond_discrete', 'beyond_fixed16', 'offset_noninferiority', 'aligned_noninferiority', 'oracle_regret']

| Clause | Observed MSE | Upper bound MSE | Pass |
|---|---:|---:|---|
| clean_mean | 0.000260119937593 | 0.003 | True |
| clean_p95 | 1.41840078868e-05 | 0.005 | True |
| homogeneous_dark | 0.0310396562447 | 0.0485728713311 | True |
| homogeneous_bright | 0.014980764722 | 0.0240787858749 | True |
| beyond_best_fixed | 0.0378063820885 | 0.0339922665357 | False |
| beyond_discrete | 0.0378063820885 | 0.0347966253517 | False |
| beyond_fixed16 | 0.0378063820885 | 0.0347465809355 | False |
| offset_noninferiority | 0.0419660501648 | 0.0410317813181 | False |
| aligned_noninferiority | 0.0357265480503 | 0.0325123369058 | False |
| oracle_regret | 0.0378063820885 | 0.0360827454232 | False |

Best fixed spatial basis (reference-only): region2_ttt_energy_sobolev

{'spatial_over_best_fixed': 0.980622572846402, 'beats_best_fixed_3pct': False, 'offset_over_bilinear': 0.9710560142767047}

| Group | Method | Mean MSE | P95 MSE | Mean finite PSNR |
|---|---|---:|---:|---:|
| clean | identity | 0 | 0 | null |
| clean | region2_direct | 0.000710774445906 | 0.000600093137473 | 18.5256858608 |
| clean | region2_discrete_projected | 0.000243814568967 | 0.000242396001704 | 23.1191770225 |
| clean | region2_ttt_projected | 0.0002479245828 | 0.000178759230766 | 23.222445374 |
| clean | fixed_step_source | 0.000267344451277 | 0.000178759230766 | 22.971536687 |
| clean | global_ttt_energy_sobolev | 0.000238258362515 | 3.57426586561e-05 | 26.0026856042 |
| clean | bilinear2_ttt_energy_sobolev | 0.000257908409549 | 9.76095179794e-06 | 28.5212468802 |
| clean | region2_ttt_energy_sobolev | 0.000288522831397 | 1.41840078868e-05 | 27.4786969067 |
| clean | routed_ttt_energy_sobolev | 0.000260119937593 | 1.41840078868e-05 | 27.7097129004 |
| clean | oracle_best_basis | 0.000225267509086 | 9.76095179794e-06 | 28.8211590317 |
| homogeneous_dark | identity | 0.0809547855519 | 0.129595614225 | 11.2842330925 |
| homogeneous_dark | region2_direct | 0.0544074087404 | 0.0927228957415 | 13.2328206256 |
| homogeneous_dark | region2_discrete_projected | 0.0468419930432 | 0.0877911798656 | 14.2770325088 |
| homogeneous_dark | region2_ttt_projected | 0.0466090589296 | 0.0909839127213 | 14.4170699548 |
| homogeneous_dark | fixed_step_source | 0.0467032463057 | 0.0909839127213 | 14.3872860638 |
| homogeneous_dark | global_ttt_energy_sobolev | 0.0275996564073 | 0.0790830176324 | 16.9310241023 |
| homogeneous_dark | bilinear2_ttt_energy_sobolev | 0.0435962864198 | 0.0878995969892 | 14.7915245986 |
| homogeneous_dark | region2_ttt_energy_sobolev | 0.0452417620691 | 0.0898503579199 | 14.6313418767 |
| homogeneous_dark | routed_ttt_energy_sobolev | 0.0310396562447 | 0.0877040915191 | 16.6998313043 |
| homogeneous_dark | oracle_best_basis | 0.0275780358003 | 0.0790830176324 | 16.9355579569 |
| homogeneous_bright | identity | 0.0401313097915 | 0.0626201782376 | 14.2467367315 |
| homogeneous_bright | region2_direct | 0.0188588481396 | 0.0407458260655 | 18.4120352204 |
| homogeneous_bright | region2_discrete_projected | 0.0204168615397 | 0.0411549547687 | 17.7113003988 |
| homogeneous_bright | region2_ttt_projected | 0.0204619184951 | 0.0430386960506 | 17.7810493149 |
| homogeneous_bright | fixed_step_source | 0.0206056253635 | 0.0430386960506 | 17.8047281643 |
| homogeneous_bright | global_ttt_energy_sobolev | 0.0122257987678 | 0.0402539866045 | 20.7820872151 |
| homogeneous_bright | bilinear2_ttt_energy_sobolev | 0.0171191518719 | 0.0406468093395 | 18.8419610275 |
| homogeneous_bright | region2_ttt_energy_sobolev | 0.018624962325 | 0.0406468093395 | 18.4098900039 |
| homogeneous_bright | routed_ttt_energy_sobolev | 0.014980764722 | 0.0367740541697 | 19.3952949527 |
| homogeneous_bright | oracle_best_basis | 0.0117712200503 | 0.035953435488 | 20.8949266479 |
| left_right | identity | 0.0606666384265 | 0.0822105493397 | 12.4011548336 |
| left_right | region2_direct | 0.0380024528829 | 0.0610697852448 | 14.6628635702 |
| left_right | region2_discrete_projected | 0.0352110357722 | 0.0608214797452 | 15.1316373222 |
| left_right | region2_ttt_projected | 0.034371916228 | 0.0627734731883 | 15.379092281 |
| left_right | fixed_step_source | 0.0345093359472 | 0.0627734731883 | 15.3413708524 |
| left_right | global_ttt_energy_sobolev | 0.0556364676915 | 0.0857819229364 | 12.8606363931 |
| left_right | bilinear2_ttt_energy_sobolev | 0.0394196741749 | 0.0600484654307 | 14.5156602655 |
| left_right | region2_ttt_energy_sobolev | 0.0333970155101 | 0.0613654250279 | 15.5539868368 |
| left_right | routed_ttt_energy_sobolev | 0.0377003192436 | 0.0613654250279 | 14.7901755587 |
| left_right | oracle_best_basis | 0.0330589110497 | 0.0583138681948 | 15.5843291282 |
| quadrants | identity | 0.0609545382205 | 0.0889400105923 | 12.3780447277 |
| quadrants | region2_direct | 0.0365370839252 | 0.0611912881956 | 14.8351051603 |
| quadrants | region2_discrete_projected | 0.033041987638 | 0.0639809656888 | 15.4497210139 |
| quadrants | region2_ttt_projected | 0.0328168975422 | 0.0655890837312 | 15.6078701605 |
| quadrants | fixed_step_source | 0.0329585141386 | 0.0656498782337 | 15.5964720663 |
| quadrants | global_ttt_energy_sobolev | 0.0537501111627 | 0.0847222652286 | 12.9482624174 |
| quadrants | bilinear2_ttt_energy_sobolev | 0.0429767536698 | 0.070764105767 | 13.9707701995 |
| quadrants | region2_ttt_energy_sobolev | 0.0309838496498 | 0.0612065115944 | 15.8957991454 |
| quadrants | routed_ttt_energy_sobolev | 0.033752776857 | 0.0687245864421 | 15.4810599316 |
| quadrants | oracle_best_basis | 0.0305849858909 | 0.060031495057 | 15.9266937346 |
| offset_left_right_40 | identity | 0.056187466858 | 0.0771882377565 | 12.7081043001 |
| offset_left_right_40 | region2_direct | 0.0416490658186 | 0.0612959658727 | 14.0162698218 |
| offset_left_right_40 | region2_discrete_projected | 0.0416310566477 | 0.0626325426623 | 14.0563240785 |
| offset_left_right_40 | region2_ttt_projected | 0.042184139695 | 0.0683056663722 | 14.0154491343 |
| offset_left_right_40 | fixed_step_source | 0.0422581949737 | 0.0672305293381 | 14.0077614578 |
| offset_left_right_40 | global_ttt_energy_sobolev | 0.0547769188415 | 0.0910937372595 | 12.9632003778 |
| offset_left_right_40 | bilinear2_ttt_energy_sobolev | 0.0406255260576 | 0.0627843245864 | 14.1656730542 |
| offset_left_right_40 | region2_ttt_energy_sobolev | 0.0407498560846 | 0.0630505468696 | 14.1232947449 |
| offset_left_right_40 | routed_ttt_energy_sobolev | 0.0419660501648 | 0.0698409486562 | 14.1086408965 |
| offset_left_right_40 | oracle_best_basis | 0.0394496614113 | 0.0627387728542 | 14.2925197703 |
| heterogeneous | identity | 0.0608105883235 | 0.0889400105923 | 12.3895997806 |
| heterogeneous | region2_direct | 0.0372697684041 | 0.0610697852448 | 14.7489843653 |
| heterogeneous | region2_discrete_projected | 0.0341265117051 | 0.0621740099043 | 15.2906791681 |
| heterogeneous | region2_ttt_projected | 0.0335944068851 | 0.0650365930051 | 15.4934812207 |
| heterogeneous | fixed_step_source | 0.0337339250429 | 0.0650365930051 | 15.4689214594 |
| heterogeneous | global_ttt_energy_sobolev | 0.0546932894271 | 0.0854446865618 | 12.9044494052 |
| heterogeneous | bilinear2_ttt_energy_sobolev | 0.0411982139223 | 0.070764105767 | 14.2432152325 |
| heterogeneous | region2_ttt_energy_sobolev | 0.03219043258 | 0.0612296355888 | 15.7248929911 |
| heterogeneous | routed_ttt_energy_sobolev | 0.0357265480503 | 0.0686784785241 | 15.1356177452 |
| heterogeneous | oracle_best_basis | 0.0318219484703 | 0.060031495057 | 15.7555114314 |
| spatial_pool | identity | 0.059269547835 | 0.0884717490524 | 12.4957679538 |
| spatial_pool | region2_direct | 0.0387295342089 | 0.061189234443 | 14.5047461841 |
| spatial_pool | region2_discrete_projected | 0.036628026686 | 0.0622125817463 | 14.8792274715 |
| spatial_pool | region2_ttt_projected | 0.0364576511551 | 0.0655113894492 | 15.0008038586 |
| spatial_pool | fixed_step_source | 0.0365753483532 | 0.0654567785561 | 14.9818681255 |
| spatial_pool | global_ttt_energy_sobolev | 0.0547211658986 | 0.0881395936012 | 12.9240330627 |
| spatial_pool | bilinear2_ttt_energy_sobolev | 0.0410073179674 | 0.0706918813288 | 14.2173678397 |
| spatial_pool | region2_ttt_energy_sobolev | 0.0350435737482 | 0.0629864256829 | 15.191026909 |
| spatial_pool | routed_ttt_energy_sobolev | 0.0378063820885 | 0.0698409486562 | 14.7932921289 |
| spatial_pool | oracle_best_basis | 0.0343645194507 | 0.0608976675197 | 15.2678475444 |

All selection counts, full per-input energy margins, oracle disagreements and selected-basis regret are in routing_diagnostics.json. Offset is primary. No new task or retuning.
