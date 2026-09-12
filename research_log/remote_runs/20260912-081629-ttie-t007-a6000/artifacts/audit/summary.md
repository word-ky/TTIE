# T007 fixed joint calibration audit

Qualifies later pilot: True; failed: []

| Scope | Gate | Clean view FPR | Clean image-any | Dark AUC | Bright AUC | Dark correct TPR | Bright correct TPR | Active precision |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | baseline | 0.215000 | 0.450000 | 0.983650 | 0.930125 | 0.915000 | 0.860000 | 0.983380 |
| all | joint | 0.060000 | 0.200000 | 0.983650 | 0.930125 | 0.625000 | 0.745000 | 0.992754 |
| full | baseline | 0.275000 | 0.275000 | 0.984375 | 0.930625 | 0.900000 | 0.875000 | 0.972603 |
| full | joint | 0.075000 | 0.075000 | 0.984375 | 0.930625 | 0.675000 | 0.775000 | 0.983051 |
| quadrants | baseline | 0.200000 | 0.375000 | 0.983359 | 0.931211 | 0.918750 | 0.856250 | 0.986111 |
| quadrants | joint | 0.056250 | 0.175000 | 0.983359 | 0.931211 | 0.612500 | 0.737500 | 0.995392 |

| Mixed scope | Gate | True type | N | Correct recall | Wrong activation | Active precision |
|---|---|---|---:|---:|---:|---:|
| all | baseline | dark | 160 | 0.906250 | 0.012500 | 0.9863945578231292 |
| all | baseline | bright | 160 | 0.875000 | 0.012500 | 0.9859154929577465 |
| all | joint | dark | 160 | 0.618750 | 0.012500 | 0.9801980198019802 |
| all | joint | bright | 160 | 0.756250 | 0.000000 | 1.0 |
| left_right | baseline | dark | 80 | 0.925000 | 0.012500 | 0.9866666666666667 |
| left_right | baseline | bright | 80 | 0.862500 | 0.012500 | 0.9857142857142858 |
| left_right | joint | dark | 80 | 0.612500 | 0.012500 | 0.98 |
| left_right | joint | bright | 80 | 0.737500 | 0.000000 | 1.0 |
| quadrants | baseline | dark | 80 | 0.887500 | 0.012500 | 0.9861111111111112 |
| quadrants | baseline | bright | 80 | 0.887500 | 0.012500 | 0.9861111111111112 |
| quadrants | joint | dark | 80 | 0.625000 | 0.012500 | 0.9803921568627451 |
| quadrants | joint | bright | 80 | 0.775000 | 0.000000 | 1.0 |

## Activation tradeoff counts

```json
{
  "all": {
    "clean_activations_removed": 31,
    "clean_activations_added": 0,
    "homogeneous_dark_correct_activations_lost": 58,
    "mixed_dark_correct_activations_lost": 46,
    "homogeneous_bright_correct_activations_lost": 23,
    "mixed_bright_correct_activations_lost": 19
  },
  "full": {
    "clean_activations_removed": 8,
    "clean_activations_added": 0,
    "homogeneous_dark_correct_activations_lost": 9,
    "mixed_dark_correct_activations_lost": 0,
    "homogeneous_bright_correct_activations_lost": 4,
    "mixed_bright_correct_activations_lost": 0
  },
  "quadrants": {
    "clean_activations_removed": 23,
    "clean_activations_added": 0,
    "homogeneous_dark_correct_activations_lost": 49,
    "mixed_dark_correct_activations_lost": 46,
    "homogeneous_bright_correct_activations_lost": 19,
    "mixed_bright_correct_activations_lost": 19
  }
}
```

All six AUC comparisons are identical: True. No retraining, threshold sweep or ISP adaptation.
