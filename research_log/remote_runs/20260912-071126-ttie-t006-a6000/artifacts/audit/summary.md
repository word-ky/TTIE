# T006 learned exposure prototypes: fixed audit

**Overall gate fails only clean-view false activation:21/100=21% >15%. No ISP adaptation is run.**

| Criterion | Learned result | Requirement | Pass |
| --- | ---: | ---: | --- |
| Clean all-view FPR | 21% | <=15% | No |
| Dark / bright AUC | .9870 / .9366 | each>=.80 | Yes |
| Homogeneous correct-type TPR, dark / bright | 93% /92% | each>=40% | Yes |
| Homogeneous active-type precision | 100% | >=85% | Yes |
| Mixed correct activation recall, dark / bright | 91.25% /95% | each>=40% | Yes |
| Mixed wrong-type activation, dark / bright | 0% /0% | each<=15% | Yes |

| Readout / views | Clean FPR | Dark AUC | Bright AUC | Dark paired increase | Bright paired increase |
| --- | ---: | ---: | ---: | ---: | ---: |
| learned / all | 21.000% | 0.987000 | 0.936600 | 100.000% | 100.000% |
| learned / full | 25.000% | 0.990000 | 0.922500 | 100.000% | 100.000% |
| learned / quadrants | 20.000% | 0.986875 | 0.941094 | 100.000% | 100.000% |
| zero_shot / all | 4.000% | 0.875500 | 0.613800 | 100.000% | 81.000% |
| zero_shot / full | 5.000% | 0.895000 | 0.607500 | 100.000% | 85.000% |
| zero_shot / quadrants | 3.750% | 0.871719 | 0.618125 | 100.000% | 80.000% |

| Readout / views / exposure | Any TPR | Correct-type TPR | Correct type among active |
| --- | ---: | ---: | ---: |
| learned / all / dark | 93.000% | 93.000% | 100.000% |
| learned / all / bright | 92.000% | 92.000% | 100.000% |
| learned / full / dark | 90.000% | 90.000% | 100.000% |
| learned / full / bright | 85.000% | 85.000% | 100.000% |
| learned / quadrants / dark | 93.750% | 93.750% | 100.000% |
| learned / quadrants / bright | 93.750% | 93.750% | 100.000% |
| zero_shot / all / dark | 11.000% | 8.000% | 72.727% |
| zero_shot / all / bright | 10.000% | 10.000% | 100.000% |
| zero_shot / full / dark | 5.000% | 0.000% | 0.000% |
| zero_shot / full / bright | 20.000% | 20.000% | 100.000% |
| zero_shot / quadrants / dark | 12.500% | 10.000% | 80.000% |
| zero_shot / quadrants / bright | 7.500% | 7.500% | 100.000% |

| Mixed readout / condition / true type | Views | Correct recall | Wrong-type rate | Active precision | Margin mean | Margin min /p25 /median /p75 /max |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| learned / all / dark | 80 | 91.250% | 0.000% | 100.000% | 0.211277 | 0.026545 / 0.160090 / 0.206634 / 0.265354 / 0.390412 |
| learned / all / bright | 80 | 95.000% | 0.000% | 100.000% | 0.203356 | -0.021549 / 0.114105 / 0.211364 / 0.269729 / 0.417014 |
| learned / left_right / dark | 40 | 92.500% | 0.000% | 100.000% | 0.210874 | 0.026545 / 0.163605 / 0.206634 / 0.262453 / 0.390412 |
| learned / left_right / bright | 40 | 95.000% | 0.000% | 100.000% | 0.204312 | -0.021549 / 0.129462 / 0.208623 / 0.269729 / 0.417014 |
| learned / quadrants / dark | 40 | 90.000% | 0.000% | 100.000% | 0.211680 | 0.046644 / 0.160090 / 0.216475 / 0.274575 / 0.376905 |
| learned / quadrants / bright | 40 | 95.000% | 0.000% | 100.000% | 0.202399 | -0.021549 / 0.110025 / 0.211447 / 0.267970 / 0.386778 |
| zero_shot / all / dark | 80 | 15.000% | 2.500% | 85.714% | 0.019103 | -0.038418 / -0.000054 / 0.016814 / 0.033747 / 0.070757 |
| zero_shot / all / bright | 80 | 7.500% | 0.000% | 100.000% | 0.029052 | -0.033507 / 0.012587 / 0.028680 / 0.043323 / 0.095283 |
| zero_shot / left_right / dark | 40 | 12.500% | 2.500% | 83.333% | 0.018542 | -0.038418 / -0.001377 / 0.016160 / 0.032606 / 0.070757 |
| zero_shot / left_right / bright | 40 | 10.000% | 0.000% | 100.000% | 0.029547 | -0.021109 / 0.011939 / 0.028680 / 0.046638 / 0.095283 |
| zero_shot / quadrants / dark | 40 | 17.500% | 2.500% | 87.500% | 0.019664 | -0.038418 / 0.003544 / 0.016814 / 0.038451 / 0.070757 |
| zero_shot / quadrants / bright | 40 | 5.000% | 0.000% | 100.000% | 0.028556 | -0.033507 / 0.012587 / 0.028996 / 0.039768 / 0.095283 |

Clean-quadrant false activation: learned20%, zero-shot3.75%. Full-view mixed scores are excluded from localization. Both readouts use the same fresh clean-calibration IDs and95th-percentile/population-std procedure fixed before training; zero-shot text prototypes/prompts are unchanged.

The learned signal substantially improves discrimination and correct localization at the frozen thresholds, while failing the declared clean-content activation ceiling. High AUC or zero mixed wrong-type activation does not establish identity safety. No threshold sweep, early stopping, optimizer/temperature/prototype/split tuning or adaptation followed this result.

Training:900source features (300perclass),1536prototype scalars,500full-batch AdamW updates; source CE .995049775 -> .353821874.501loss samples and500gradient norms, initial/final cosines, learned tensors/hash, calibration and all700score rows are retained. Views/synthetic conditions are correlated;20held-out images from an incompletely documented200-image pool do not establish natural-image population or restoration performance.
