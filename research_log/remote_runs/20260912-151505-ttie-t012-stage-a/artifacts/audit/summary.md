# T012 Stage A

Pass: False; failed: ['clean_p95', 'homogeneous_dark', 'beyond_discrete', 'beyond_fixed_step']

| Metric / ratio | Value |
|---|---:|
| clean_p95 | 0.0051114665 |
| dark_ratio | 0.6574603091 |
| bright_ratio | 0.4033348411 |
| discrete_ratio | 1.0977345391 |
| fixed_step_ratio | 1.1384553552 |
| oracle_regret_ratio | 1.1543505056 |

| Condition | Method | Mean MSE | P95 MSE | Mean finite PSNR |
|---|---|---:|---:|---:|
| clean | identity | 0.000000000 | 0.000000000 | null |
| clean | region2_direct | 0.002744979 | 0.016906760 | 23.189041583 |
| clean | region2_discrete_projected | 0.001059633 | 0.004942643 | 25.022412365 |
| clean | region2_ttt_projected | 0.000792457 | 0.004028250 | 26.464579539 |
| clean | region2_ttt_projected_1step | 0.000062106 | 0.000412377 | 39.687687038 |
| clean | region2_ttt_learned_stop | 0.000734536 | 0.005111467 | 95.412355744 |
| clean | fixed_step_source | 0.001070099 | 0.004781726 | 25.672382218 |
| clean | oracle_best_checkpoint | 0.000000000 | 0.000000000 | 168.424480602 |
| homogeneous_dark | identity | 0.073152555 | 0.117559322 | 11.921526673 |
| homogeneous_dark | region2_direct | 0.049792175 | 0.090154435 | 13.990074608 |
| homogeneous_dark | region2_discrete_projected | 0.041348616 | 0.083794983 | 15.930671031 |
| homogeneous_dark | region2_ttt_projected | 0.041399476 | 0.083794983 | 15.766473929 |
| homogeneous_dark | region2_ttt_projected_1step | 0.068604552 | 0.111302167 | 12.239114181 |
| homogeneous_dark | region2_ttt_learned_stop | 0.048094901 | 0.116103517 | 15.489298634 |
| homogeneous_dark | fixed_step_source | 0.040829941 | 0.083794983 | 15.890281527 |
| homogeneous_dark | oracle_best_checkpoint | 0.040627462 | 0.083794983 | 15.926382977 |
| homogeneous_bright | identity | 0.045504754 | 0.083316941 | 13.866978634 |
| homogeneous_bright | region2_direct | 0.013654373 | 0.032431572 | 20.518600467 |
| homogeneous_bright | region2_discrete_projected | 0.018221054 | 0.032504247 | 18.074648141 |
| homogeneous_bright | region2_ttt_projected | 0.017751741 | 0.032420514 | 18.262536834 |
| homogeneous_bright | region2_ttt_projected_1step | 0.037643648 | 0.065026961 | 14.611270298 |
| homogeneous_bright | region2_ttt_learned_stop | 0.018353653 | 0.032425628 | 18.053709812 |
| homogeneous_bright | fixed_step_source | 0.016392433 | 0.032414267 | 18.644478614 |
| homogeneous_bright | oracle_best_checkpoint | 0.015324100 | 0.032413257 | 19.064112739 |
| left_right | identity | 0.060372320 | 0.099433232 | 12.676783448 |
| left_right | region2_direct | 0.031370185 | 0.053591870 | 15.644184919 |
| left_right | region2_discrete_projected | 0.028700330 | 0.056192993 | 16.307396820 |
| left_right | region2_ttt_projected | 0.029220814 | 0.059762079 | 16.112516604 |
| left_right | region2_ttt_projected_1step | 0.053981172 | 0.089497140 | 13.149899976 |
| left_right | region2_ttt_learned_stop | 0.030948609 | 0.075653172 | 16.202337014 |
| left_right | fixed_step_source | 0.028123102 | 0.058918130 | 16.387540219 |
| left_right | oracle_best_checkpoint | 0.027689075 | 0.057793718 | 16.477708540 |
| quadrants | identity | 0.059279274 | 0.097901288 | 12.760159555 |
| quadrants | region2_direct | 0.031383478 | 0.054169004 | 15.637671198 |
| quadrants | region2_discrete_projected | 0.029023383 | 0.053963270 | 16.226801803 |
| quadrants | region2_ttt_projected | 0.028747621 | 0.053735648 | 16.227751174 |
| quadrants | region2_ttt_projected_1step | 0.053000484 | 0.084024204 | 13.231531123 |
| quadrants | region2_ttt_learned_stop | 0.032416704 | 0.065522005 | 15.949908630 |
| quadrants | fixed_step_source | 0.027535921 | 0.053243302 | 16.564513025 |
| quadrants | oracle_best_checkpoint | 0.027203536 | 0.052778480 | 16.624710732 |
| heterogeneous | identity | 0.059825797 | 0.098811030 | 12.718471501 |
| heterogeneous | region2_direct | 0.031376832 | 0.053789149 | 15.640928059 |
| heterogeneous | region2_discrete_projected | 0.028861856 | 0.055851760 | 16.267099311 |
| heterogeneous | region2_ttt_projected | 0.028984217 | 0.055449386 | 16.170133889 |
| heterogeneous | region2_ttt_projected_1step | 0.053490828 | 0.088836218 | 13.190715549 |
| heterogeneous | region2_ttt_learned_stop | 0.031682656 | 0.075454790 | 16.076122822 |
| heterogeneous | fixed_step_source | 0.027829512 | 0.053911701 | 16.476026622 |
| heterogeneous | oracle_best_checkpoint | 0.027446305 | 0.053292351 | 16.551209636 |

Oracle uses references only after label-free outputs are persisted. Offset stress is report-only. No tuning or automatic next task.
