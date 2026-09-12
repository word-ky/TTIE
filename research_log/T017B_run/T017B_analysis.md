# T017-B: frozen-choice soft-to-hard failure attribution

Result: **transfer_dominant**. Reference-only development attribution, not a qualification or deployable selector.

| Group | Harmful hard moves | Transfer / Interaction / Zero-tie | Fractions | Group dominance |
|---|---:|---|---|---|
| spatial_pool | 20 | {'transfer_flip': 19, 'soft_interaction_failure': 1, 'zero/tie': 0} | {'transfer_flip': 0.95, 'soft_interaction_failure': 0.05, 'zero/tie': 0.0} | transfer_dominant |
| left_right | 9 | {'transfer_flip': 9, 'soft_interaction_failure': 0, 'zero/tie': 0} | {'transfer_flip': 1.0, 'soft_interaction_failure': 0.0, 'zero/tie': 0.0} | transfer_dominant |
| quadrants | 11 | {'transfer_flip': 10, 'soft_interaction_failure': 1, 'zero/tie': 0} | {'transfer_flip': 0.9090909090909091, 'soft_interaction_failure': 0.09090909090909091, 'zero/tie': 0.0} | transfer_dominant |
| offset_left_right_40 | 0 | {'transfer_flip': 0, 'soft_interaction_failure': 0, 'zero/tie': 0} | {'transfer_flip': None, 'soft_interaction_failure': None, 'zero/tie': None} | no_harmful_moves |

Overall attribution requires the same category to reach at least 2/3 of harmful moves both pooled and in quadrants. Integer comparison 3*count>=2*total, no tolerance. No harmful moves gives undefined fractions, not dominance.

| Group | Soft separability regret: mean / median / p95 / max / zero count | First oracle equality | In any oracle tie set | Tied oracle episodes / selected in tied set |
|---|---|---|---|---|
| spatial_pool | {'mean': 3.224369914581378e-05, 'median': 0.0, 'p95': 0.00016416171565651887, 'max': 0.00128931924700737, 'zero_count': 104} | 92 (0.7666666666666667) | 104 (0.8666666666666667) | 12 / 12 |
| left_right | {'mean': 1.5571340918540953e-05, 'median': 0.0, 'p95': 5.402667447924581e-05, 'max': 0.0003486126661300659, 'zero_count': 34} | 29 (0.725) | 34 (0.85) | 5 / 5 |
| quadrants | {'mean': 1.2148963287472725e-05, 'median': 0.0, 'p95': 3.247549757361388e-05, 'max': 0.00034543126821517944, 'zero_count': 37} | 37 (0.925) | 37 (0.925) | 0 / 0 |
| offset_left_right_40 | {'mean': 6.901079323142766e-05, 'median': 0.0, 'p95': 0.00023978343233465942, 'max': 0.00128931924700737, 'zero_count': 33} | 26 (0.65) | 33 (0.825) | 7 / 7 |

## spatial_pool gain signs

Rows=soft_joint_gain sign; columns=hard_gain sign.

| Soft sign | Hard negative | Hard zero | Hard positive |
|---|---:|---:|---:|
| negative | 1 | 0 | 0 |
| zero | 0 | 39 | 0 |
| positive | 19 | 0 | 61 |

## left_right gain signs

Rows=soft_joint_gain sign; columns=hard_gain sign.

| Soft sign | Hard negative | Hard zero | Hard positive |
|---|---:|---:|---:|
| negative | 0 | 0 | 0 |
| zero | 0 | 5 | 0 |
| positive | 9 | 0 | 26 |

## quadrants gain signs

Rows=soft_joint_gain sign; columns=hard_gain sign.

| Soft sign | Hard negative | Hard zero | Hard positive |
|---|---:|---:|---:|
| negative | 1 | 0 | 0 |
| zero | 0 | 29 | 0 |
| positive | 10 | 0 | 0 |

## offset_left_right_40 gain signs

Rows=soft_joint_gain sign; columns=hard_gain sign.

| Soft sign | Hard negative | Hard zero | Hard positive |
|---|---:|---:|---:|
| negative | 0 | 0 | 0 |
| zero | 0 | 5 | 0 |
| positive | 0 | 0 | 35 |

## Bounded interpretation

The tau=.05 neighborhood is not a faithful surrogate for hard-boundary deployment; do not train a hard-deployment geometry objective from it in the next cycle.

The nine-soft oracle is diagnostic only and never changes any of the 120 frozen A choices. Oracle equality includes exact tie-aware and first lexicographic variants separately. p95 uses unchanged linear quantiles. No clipping, tolerance, threshold, rescue rule or new selector. Stop after T017-B; no geometry objective or fresh training.
