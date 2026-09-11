# T003 complete fixed Pareto diagnostic

936 rows; all finite: True.

One setting must reduce worst clean drift by at least 5x AND retain at least 70% of baseline improvement. These are diagnostic thresholds, not a per-image selection policy.

Baseline worst drift: 0.0628858656; safety ceiling: 0.0125771731; baseline mean MSE improvement: 0.00600123236.

| Setting (anchor, TV) | Mean clean drift | Worst clean drift | Worst family mean | Drift reduction | Utility retained | Meets both |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| (0, 0) | 0.0279259 | 0.0628859 | 0.0628824 | 1.000x | 100.000% | False |
| (0.01, 0) | 0.0267567 | 0.0622966 | 0.0622932 | 1.009x | 104.931% | False |
| (0.1, 0) | 0.0233593 | 0.0599426 | 0.0599393 | 1.049x | 78.810% | False |
| (1, 0) | 0.0153735 | 0.0430007 | 0.0429983 | 1.462x | -54.356% | False |
| (10, 0) | 0.0030384 | 0.0089490 | 0.0089483 | 7.027x | -516.828% | False |
| (0, 0.01) | 0.0273664 | 0.0628816 | 0.0628802 | 1.000x | 92.302% | False |
| (0, 0.1) | 0.0257677 | 0.0628020 | 0.0627932 | 1.001x | 72.599% | False |
| (0, 1) | 0.0226825 | 0.0629759 | 0.0629259 | 0.999x | -192.740% | False |
| (0.1, 0.1) | 0.0221615 | 0.0600628 | 0.0600088 | 1.047x | 53.862% | False |
| (1, 0.1) | 0.0150028 | 0.0429802 | 0.0429731 | 1.463x | -106.000% | False |
| (1, 1) | 0.0149865 | 0.0432249 | 0.0431449 | 1.455x | -403.836% | False |

Qualifying fixed settings: NONE

Safety uses the worst individual input among the nine clean cases; the alternative worst family-mean is displayed without changing the decision rule. Utility is a ratio of mean improvements, not a mean of per-image ratios. Negative utility is retained, not clipped.

All input-level errors, region errors, clipping and grid diagnostics remain in metrics.csv/metrics.json. All prior/anchor/TV/total and gradient trajectories are in trajectories/. Seeds vary synthetic noise only. Passing does not establish natural-image identity safety; failing rules out only this finite diagnostic weight sweep and optimization budget.
