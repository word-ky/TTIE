# T005 fresh relative-versus-absolute CLIP audit

**Stage A fails; Stage B was not run.**

| Gate criterion (all views) | Value | Requirement | Pass |
| --- | ---: | ---: | --- |
| Clean relative FPR | 7/100 = 7% | <=15% | Yes |
| Dark response AUC | 0.580600 | >=0.75 | No |
| Bright response AUC | 0.294500 | >=0.75 | No |
| Dark correct-type TPR | 10/100 = 10% | >=30% | No |
| Bright correct-type TPR | 5/100 = 5% | >=30% | No |
| Combined active type precision | 15/21 = 71.429% | >=80% | No |

| Signal / views | Clean FPR | Dark AUC | Bright AUC | Paired dark increase | Paired bright increase |
| --- | ---: | ---: | ---: | ---: | ---: |
| relative / all | 7.000% | 0.580600 | 0.294500 | 56.000% | 29.000% |
| relative / full | 5.000% | 0.520000 | 0.400000 | 40.000% | 45.000% |
| relative / quadrants | 7.500% | 0.597344 | 0.270625 | 60.000% | 25.000% |
| absolute / all | 8.000% | 0.833900 | 0.604000 | 100.000% | 85.000% |
| absolute / full | 0.000% | 0.837500 | 0.632500 | 100.000% | 90.000% |
| absolute / quadrants | 10.000% | 0.831719 | 0.600313 | 100.000% | 83.750% |

| Signal / views / condition | Any-activation TPR | Correct-type TPR | Correct type among active |
| --- | ---: | ---: | ---: |
| relative / all / dark | 10.000% | 10.000% | 100.000% |
| relative / all / bright | 11.000% | 5.000% | 45.455% |
| relative / full / dark | 0.000% | 0.000% | undefined (no active views) |
| relative / full / bright | 10.000% | 10.000% | 100.000% |
| relative / quadrants / dark | 12.500% | 12.500% | 100.000% |
| relative / quadrants / bright | 11.250% | 3.750% | 33.333% |
| absolute / all / dark | 29.000% | 28.000% | 96.552% |
| absolute / all / bright | 9.000% | 7.000% | 77.778% |
| absolute / full / dark | 20.000% | 15.000% | 75.000% |
| absolute / full / bright | 15.000% | 15.000% | 100.000% |
| absolute / quadrants / dark | 31.250% | 31.250% | 100.000% |
| absolute / quadrants / bright | 7.500% | 5.000% | 66.667% |

Absolute reference uses the unchanged T004 model/prompts/thresholds on the SAME fresh T005 images. The relative signal is worse in both AUCs; comparing only with previous T004 images would confound the split. Paired absolute score sensitivity (100% dark,85% bright) does not transfer to the correction-response ranking (56% dark,29% bright). A finite local response is not automatically an exposure-severity score. This audit does not isolate the contribution of saturation/clipping or prove a universal failure of within-image signals.

All650 score rows (50calibration/600heldout), original/plus/minus score terms, fixed targets and clipping remain in scores.csv/json. No response-sign inversion, new probe magnitude, threshold/prompt/model tuning or adaptation was attempted after seeing the result. Fresh30-image manifest excludes every T004 ID; the existing200-image source-pool provenance and correlated20-image held-out views limit population claims.
