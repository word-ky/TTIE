# T013 Stage A

Pass: False; failed: ['gradient_positive', 'gradient_median', 'beyond_discrete', 'beyond_fixed_step']

| Value | Observed |
|---|---:|
| positive_cosine_fraction | 0.692307692308 |
| median_cosine | 0.247648800633 |
| clean_p95 | 0.00396800745511 |
| dark_ratio | 0.572854301476 |
| bright_ratio | 0.584402447032 |
| discrete_ratio | 1.21313491791 |
| fixed_step_ratio | 1.24061404489 |

Oracle diagnostics (reference-only): {'oracle_regret_ratio': 1.0645632855672291, 'oracle_over_discrete': 1.1395611086291644, 'oracle_over_fixed16': 1.1653736905152772, 'oracle_beats_discrete_5pct': False, 'oracle_beats_fixed16_5pct': False}

| Condition | Method | Mean MSE | P95 MSE | Mean finite PSNR |
|---|---|---:|---:|---:|
| clean | identity | 0 | 0 | null |
| clean | region2_direct | 0.0003580142293 | 0.002195431397 | 28.57565245 |
| clean | region2_discrete_projected | 0.0002250165126 | 0.001926385047 | 28.80795669 |
| clean | region2_ttt_projected | 0.0002433263464 | 0.001614168432 | 27.94537003 |
| clean | fixed_step_source | 0.0002258391061 | 0.001609582611 | 28.3244443 |
| clean | global_ttt_energy | 0.001887541986 | 0.01155345528 | 19.74976868 |
| clean | bilinear2_ttt_energy | 0.0004267005861 | 0.002108949661 | 27.00134929 |
| clean | region2_ttt_energy | 0.0005608959938 | 0.003968007455 | 25.13594895 |
| clean | oracle_best_energy_checkpoint | 5.05107493e-18 | 3.697868742e-17 | 164.7854597 |
| homogeneous_dark | identity | 0.07090148674 | 0.132149522 | 12.12108681 |
| homogeneous_dark | region2_direct | 0.04619118213 | 0.106111626 | 14.30080103 |
| homogeneous_dark | region2_discrete_projected | 0.03850329822 | 0.09909385331 | 15.71808501 |
| homogeneous_dark | region2_ttt_projected | 0.03771898496 | 0.0997146409 | 15.86463284 |
| homogeneous_dark | fixed_step_source | 0.03795582632 | 0.1012275513 | 15.81921213 |
| homogeneous_dark | global_ttt_energy | 0.03810952195 | 0.1244887225 | 17.50219376 |
| homogeneous_dark | bilinear2_ttt_energy | 0.04155886973 | 0.1157834943 | 15.34958618 |
| homogeneous_dark | region2_ttt_energy | 0.04061622166 | 0.1196011547 | 15.47901394 |
| homogeneous_dark | oracle_best_energy_checkpoint | 0.03977664447 | 0.1189835329 | 15.56193638 |
| homogeneous_bright | identity | 0.03550040533 | 0.05834755581 | 14.90496107 |
| homogeneous_bright | region2_direct | 0.01495253687 | 0.03256110288 | 19.50069772 |
| homogeneous_bright | region2_discrete_projected | 0.0170983312 | 0.03600338232 | 18.33112847 |
| homogeneous_bright | region2_ttt_projected | 0.01695556743 | 0.03564288151 | 18.51216779 |
| homogeneous_bright | fixed_step_source | 0.01693880806 | 0.03550726473 | 18.55764201 |
| homogeneous_bright | global_ttt_energy | 0.0228124499 | 0.06175006889 | 18.70895144 |
| homogeneous_bright | bilinear2_ttt_energy | 0.01807357124 | 0.0424429113 | 18.67801139 |
| homogeneous_bright | region2_ttt_energy | 0.02074652375 | 0.04264850412 | 17.91032431 |
| homogeneous_bright | oracle_best_energy_checkpoint | 0.01756313323 | 0.04169091135 | 18.91721199 |
| left_right | identity | 0.05227854734 | 0.08539818972 | 13.2552404 |
| left_right | region2_direct | 0.02881411259 | 0.05660199523 | 16.41130105 |
| left_right | region2_discrete_projected | 0.02438164538 | 0.04893830232 | 17.16974819 |
| left_right | region2_ttt_projected | 0.02333281875 | 0.05041686613 | 17.44884827 |
| left_right | fixed_step_source | 0.02360390201 | 0.05040584709 | 17.37125963 |
| left_right | global_ttt_energy | 0.05670712339 | 0.08582976423 | 12.63244864 |
| left_right | bilinear2_ttt_energy | 0.03558840689 | 0.06731367409 | 15.09652034 |
| left_right | region2_ttt_energy | 0.02977938186 | 0.06909580566 | 16.31530968 |
| left_right | oracle_best_energy_checkpoint | 0.02798979995 | 0.06647267975 | 16.66663023 |
| quadrants | identity | 0.05250937708 | 0.0839388337 | 13.25109203 |
| quadrants | region2_direct | 0.02974616759 | 0.05613643825 | 16.19419973 |
| quadrants | region2_discrete_projected | 0.02644708742 | 0.0487708332 | 16.65310343 |
| quadrants | region2_ttt_projected | 0.02572387347 | 0.05025923941 | 16.85301516 |
| quadrants | fixed_step_source | 0.02609899378 | 0.05033350959 | 16.78068588 |
| quadrants | global_ttt_energy | 0.05649861023 | 0.08852559216 | 12.87754628 |
| quadrants | bilinear2_ttt_energy | 0.03835107409 | 0.0635885898 | 14.71413453 |
| quadrants | region2_ttt_energy | 0.03188272873 | 0.06054327209 | 15.72977173 |
| quadrants | oracle_best_energy_checkpoint | 0.02993264715 | 0.05454674326 | 16.04017603 |
| heterogeneous | identity | 0.05239396221 | 0.08539818972 | 13.25316621 |
| heterogeneous | region2_direct | 0.02928014009 | 0.05613643825 | 16.30275039 |
| heterogeneous | region2_discrete_projected | 0.0254143664 | 0.0487708332 | 16.91142581 |
| heterogeneous | region2_ttt_projected | 0.02452834611 | 0.05025923941 | 17.15093172 |
| heterogeneous | fixed_step_source | 0.0248514479 | 0.05033350959 | 17.07597275 |
| heterogeneous | global_ttt_energy | 0.05660286681 | 0.08830453418 | 12.75499746 |
| heterogeneous | bilinear2_ttt_energy | 0.03696974049 | 0.06731367409 | 14.90532744 |
| heterogeneous | region2_ttt_energy | 0.0308310553 | 0.06141782701 | 16.02254071 |
| heterogeneous | oracle_best_energy_checkpoint | 0.02896122355 | 0.06127976775 | 16.35340313 |
