The prescribed T016-E confidence fallback passes all5fixed development clauses. Each outer fold independently selects0.75 from the unchanged8thresholdgrid using the other32IDs and their frozen D OOF scores. No head is loaded or retrained.

| Group | Selected MSE | / Region2 | / hard oracle | / ungated rank30 |
|---|---:|---:|---:|---:|
| Spatial | .0339046937918 | .967501032727 | 1.04118496524 | .990839152561 |
| Left/right | .03372707651 | 1.00988294896 | 1.04836938295 | .964671908714 |
| Quadrants | .0309838496498 | 1.0 | 1.00005362091 | .983995820478 |
| Offset | .0370031552156 | .908056095679 | 1.07139006609 | 1.02206033524 |

32adaptive/88canonical;27beneficial,4harmful(allleft/right),1zero. Left/right passes with a narrow margin. Five thresholds and120decisions were frozen before attaching evaluation references.

**Interpretation limitation:** this follows the requested reuse of existing OOF scores, not fully nested ranker training. For all480calibrationepisode appearances, the frozen ranker producing that score had trained on the current outer-held-out8IDs. Direct threshold calibration excludes their reference rows, but earlier scorer-training dependence remains. The5/5 result is the literal prescribed development audit, not leakage-free fully nested CV or independent fresh generalization. No deployable threshold or fresh evaluation is claimed.

Validation: original evaluator reproduces both accepted D probes exactly; kernel4testsPASS0.030s; integrated6focusedtestsPASS0.103s; py_compilePASS. Independent audit verifies6sourcefiles,8inputartifacts,120q/decisions,40calibrationmeans,5thresholds,allgroups/folds/gaincounts/quantiles/ratios/clauses and freeze hashes. NumPy q agrees within2.22e-16. Formal runexit0; no failures or tuning. No GPU, rendering, CLIP, TTT, new IDs, referenceMSE recomputation or model changes.

Frozen source c98e6d119b89b8ffb755ed5b1e0d064973684675; evidence66b597cef47d483dea2027d9ad925d8961031f98. Full report: https://github.com/word-ky/TTIE/blob/66b597cef47d483dea2027d9ad925d8961031f98/research_log/T016E_analysis.md

Research review only; no self-merge, PR19 topology repair or follow-on experiment.