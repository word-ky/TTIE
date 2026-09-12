# T009 development geometry diagnosis

Fired rules: ['overcorrection_stopping_failure', 'renderer_failure']

| Pool | State | Coordinate | Cosine positive | Cosine median | Step1 improves | Earlier better | Final MSE | Oracle-step MSE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| pooled_nonclean | 1 | ev_only | 0.941176 | 1.000000 | 0.915033 | 0.790850 | 0.046741 | 0.039638 |
| pooled_nonclean | 1 | gamma_only | 0.745098 | 1.000000 | 0.738562 | 0.601307 | 0.055679 | 0.047511 |
| pooled_nonclean | 1 | ev_gamma | 0.934641 | 0.978401 | 0.934641 | 0.633987 | 0.038519 | 0.032088 |
| pooled_nonclean | 2 | ev_only | 0.947712 | 0.816323 | 0.921569 | 0.666667 | 0.040289 | 0.033629 |
| pooled_nonclean | 2 | gamma_only | 0.738562 | 0.566447 | 0.764706 | 0.588235 | 0.056667 | 0.050203 |
| pooled_nonclean | 2 | ev_gamma | 0.947712 | 0.797269 | 0.928105 | 0.692810 | 0.040405 | 0.032150 |
| homogeneous_dark | 1 | ev_only | 1.000000 | 1.000000 | 1.000000 | 0.257143 | 0.027029 | 0.026662 |
| homogeneous_dark | 1 | gamma_only | 0.971429 | 1.000000 | 0.971429 | 0.028571 | 0.039647 | 0.039120 |
| homogeneous_dark | 1 | ev_gamma | 0.971429 | 0.982898 | 1.000000 | 0.371429 | 0.024296 | 0.022798 |
| homogeneous_dark | 2 | ev_only | 1.000000 | 0.871956 | 0.971429 | 0.371429 | 0.031675 | 0.030921 |
| homogeneous_dark | 2 | gamma_only | 0.971429 | 0.802518 | 0.914286 | 0.200000 | 0.052884 | 0.051557 |
| homogeneous_dark | 2 | ev_gamma | 1.000000 | 0.847777 | 0.942857 | 0.428571 | 0.032011 | 0.030395 |
| homogeneous_bright | 1 | ev_only | 0.974359 | 1.000000 | 0.974359 | 0.846154 | 0.012656 | 0.007725 |
| homogeneous_bright | 1 | gamma_only | 0.153846 | -1.000000 | 0.153846 | 0.974359 | 0.068891 | 0.043847 |
| homogeneous_bright | 1 | ev_gamma | 0.974359 | 0.992672 | 0.974359 | 0.820513 | 0.014904 | 0.009629 |
| homogeneous_bright | 2 | ev_only | 0.974359 | 0.916109 | 0.974359 | 0.820513 | 0.012916 | 0.007960 |
| homogeneous_bright | 2 | gamma_only | 0.153846 | -0.511953 | 0.307692 | 0.923077 | 0.060644 | 0.044010 |
| homogeneous_bright | 2 | ev_gamma | 0.974359 | 0.913311 | 0.974359 | 0.794872 | 0.016121 | 0.009384 |
| left_right | 1 | ev_only | 0.871795 | 1.000000 | 0.820513 | 1.000000 | 0.073070 | 0.062055 |
| left_right | 1 | gamma_only | 0.948718 | 1.000000 | 0.923077 | 0.615385 | 0.055143 | 0.052605 |
| left_right | 1 | ev_gamma | 0.871795 | 0.895324 | 0.871795 | 0.666667 | 0.057053 | 0.047816 |
| left_right | 2 | ev_only | 0.948718 | 0.808515 | 0.871795 | 0.641026 | 0.047925 | 0.040868 |
| left_right | 2 | gamma_only | 0.923077 | 0.695978 | 0.923077 | 0.512821 | 0.054962 | 0.051194 |
| left_right | 2 | ev_gamma | 0.948718 | 0.775984 | 0.948718 | 0.743590 | 0.047697 | 0.038865 |
| quadrants | 1 | ev_only | 0.925000 | 1.000000 | 0.875000 | 1.000000 | 0.074211 | 0.062110 |
| quadrants | 1 | gamma_only | 0.925000 | 1.000000 | 0.925000 | 0.725000 | 0.059033 | 0.054473 |
| quadrants | 1 | ev_gamma | 0.925000 | 0.910183 | 0.900000 | 0.650000 | 0.057824 | 0.048109 |
| quadrants | 2 | ev_only | 0.875000 | 0.662086 | 0.875000 | 0.800000 | 0.068640 | 0.054765 |
| quadrants | 2 | gamma_only | 0.925000 | 0.584960 | 0.925000 | 0.675000 | 0.058180 | 0.054050 |
| quadrants | 2 | ev_gamma | 0.875000 | 0.634262 | 0.850000 | 0.775000 | 0.065793 | 0.049957 |
| clean | 1 | ev_only | null | null | 0.000000 | 1.000000 | 0.001721 | 0.000000 |
| clean | 1 | gamma_only | null | null | 0.000000 | 1.000000 | 0.002549 | 0.000000 |
| clean | 1 | ev_gamma | null | null | 0.000000 | 1.000000 | 0.002646 | 0.000000 |
| clean | 2 | ev_only | null | null | 0.000000 | 1.000000 | 0.007175 | 0.000000 |
| clean | 2 | gamma_only | null | null | 0.000000 | 1.000000 | 0.001716 | 0.000000 |
| clean | 2 | ev_gamma | null | null | 0.000000 | 1.000000 | 0.007467 | 0.000000 |

Development/oracle results only. No T010 or fresh held-out qualification is implied. Full renderer, surface, coordinate and correlation summaries are in summary.json.
