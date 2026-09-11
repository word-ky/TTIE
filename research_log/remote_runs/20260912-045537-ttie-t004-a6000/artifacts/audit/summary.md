# T004 Stage-A frozen CLIP audit

**Gate failed: bright-vs-clean AUC < 0.75. Stage B was not run.**

| Views | Clean FPR | Dark AUC | Bright AUC | Dark paired increase | Bright paired increase |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 13.333% | 0.836389 | 0.598611 | 100.000% | 90.000% |
| full | 8.333% | 0.881944 | 0.604167 | 100.000% | 100.000% |
| quadrants | 14.583% | 0.825955 | 0.600694 | 100.000% | 87.500% |

| Views / condition | Any-activation TPR | Correct-type TPR | Correct type among active |
| --- | ---: | ---: | ---: |
| all / dark | 40.000% | 36.667% | 91.667% |
| all / bright | 18.333% | 16.667% | 90.909% |
| full / dark | 50.000% | 41.667% | 83.333% |
| full / bright | 25.000% | 25.000% | 100.000% |
| quadrants / dark | 37.500% | 35.417% | 94.444% |
| quadrants / bright | 16.667% | 14.583% | 87.500% |

Aggregate correct type among active homogeneous-degraded views: 32/35 = 91.429%. Clean false activations:8/60. The five gate criteria are evaluated on all views; subgroup breakdowns do not replace the aggregate gate.

Paired brightness scores can increase while cross-image discrimination remains weak: bright paired increase is54/60 but AUC is0.598611. This is descriptive evidence of overlapping clean/degraded score distributions, not a causal isolation of content bias. Low thresholded recall is also retained, even though it is not part of the gate.

Only12 held-out source images; crops and synthetic variants are correlated. The18-image metadata-selected subset comes from an existing200-image COCO cache of incompletely documented selection provenance. No claim of representative COCO performance, restoration quality, or spatial-vs-global TTT benefit follows from this audit. All390 scores, clipping values, frozen constants and failed-run receipt remain available.
