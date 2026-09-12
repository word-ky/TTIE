

# T016-E — DONE — literal5/5 confidence audit; inherited OOF dependency

UTC: 2026-09-12T20:41:54.943563+00:00. Scientific source **c98e6d119b89b8ffb755ed5b1e0d064973684675**; evidence **66b597cef47d483dea2027d9ad925d8961031f98**; branch `codex/T016E-confidence-abstention`; PR20 https://github.com/word-ky/TTIE/pull/20 (ready, unmerged). Task from inbox993c4813/state883a6b3d; no PR19merge/topology work.

**Literal prescribed result:5/5**, thresholds independently calibrated as [.75,.75,.75,.75,.75]. **Interpretation limit:** this is the exact requested reuse of old D OOF scores, not a fully nested ranker-training evaluation. In all480calibrationepisode appearances, the frozen ranker producing that image's OOF scores had trained on the current outer-held-out8IDs. Direct threshold calibration never indexes its held-out reference rows, but earlier scorer-training dependence remains. Thus the numerical pass is a bounded development diagnostic, not leakage-free fully nested CV, independent fresh qualification or a deployable0.75threshold.

Implementation: purePython on immutable D30OOF, exactCfolds and acceptedB/C/D joins. No head loaded/retrained/rescored. q=(s4-min_noncanonical(s))/max(popstd(s),1e-12),constant0; lexicographic noncanonical tie; exactgrid[0,.25,.5,.75,1,1.5,2,inf]. Fit each threshold only to96episodes of other32IDs with their own oldOOFs, equal weights/no condition input; meanMSE minimum with largerthreshold on exactties. Strictq>t adapts; otherwisecanonicalindex4. All5thresholds and120decisions frozen before reference evaluation.

| Group | Selected MSE | /Region2 | /Hard oracle | /Ungated rank30 | /Pointwise probe30 | Adaptive / Canonical | Beneficial / Harmful / Zero |
|---|---:|---:|---:|---:|---:|---|---|
| spatial_pool | 0.0339046937918 | 0.967501032727 | 1.04118496524 | 0.990839152561 | 0.956381010474 | 32 / 88 | {'beneficial': 27, 'harmful': 4, 'zero': 1} |
| left_right | 0.03372707651 | 1.00988294896 | 1.04836938295 | 0.964671908714 | 0.98584808066 | 9 / 31 | {'beneficial': 4, 'harmful': 4, 'zero': 1} |
| quadrants | 0.0309838496498 | 1 | 1.00005362091 | 0.983995820478 | 0.928332475072 | 0 / 40 | {'beneficial': 0, 'harmful': 0, 'zero': 0} |
| offset_left_right_40 | 0.0370031552156 | 0.908056095679 | 1.07139006609 | 1.02206033524 | 0.954524692578 | 23 / 17 | {'beneficial': 23, 'harmful': 0, 'zero': 0} |

Clause vector [true,true,true,true,true]. Spatial improves3.2499%overRegion2; offset improves9.1944%. All quadrants returncanonical. Overall32adapted/88canonical,27beneficial/4harmful/1zero; all4harmful episodes are left/right. LRratio1.00988294896 is only0.011705percentage points inside the1%degradation limit. No robustness claim or alternative pass route.

Spatial selectioncounts[13,5,7,3,88,2,1,0,1],oracle[27,2,12,23,41,14,0,1,0]; disagreement .5083333333333333,outsideoracleties .43333333333333335. qmean .26783018609886866,median .2841843780692793; linearquantiles0/25/50/75/100=[-1.2891933917858664,-.2953141379989333,.2841843780692793,.858051015485551,2.6785315226655437]. Fullper-conditionq,counts/gains,oracle/tierates and foldconditioncounts are in report/summary; quantiles were predeclareddescriptive only.

Foldadaptive/canonical counts:8/16,6/18,4/20,8/16,6/18. FoldselectedMSEs: [0.042257298327361546, 0.020956261005873483, 0.04073799964195738, 0.03865445451810956, 0.02691745546568806]. All40calibrationgridmeanMSEs and480calibrationtraceentries retained in calibrations.json; fullprecision120q/decisions in decisions.json.

Validation: unchanged donor evaluator reproduces both acceptedD probes exactly. Kernel4testsPASS0.030s; integrated6focusedtestsPASS0.103s; py_compilePASS (overlapping incrementtests). Covers exactq/std-floor/constant/ties, exactgrid/strictcomparison/safermean ties, directheldoutreference guard andmutationinvariance, useofoldOOFrows, allthreshold/decision freeze beforeeval, gain/coverage/quantile arithmetic. No unrelatedfullsuite or trainingrerun.

Independent auditPASS:6scientificGitblob/current-filehashes,8inputartifacthashes,exactgroupedfolds,120q/decisions againstNumPy(maxabs2.22e-16),40calibrationmeans,5thresholds,allMSEjoins/ratios/counts/gains/quantiles/clauses andfreezehashes. Canonicalcandidate4acceptedMSE equalsRegion2exactlyfor120/120. Audit also explicitly establishes the inherited ranker-training dependency described above. No audittraining or changes to decisions.

Command: `D:/anaconda3/python.exe -m ttie.boundary_confidence_run --source-sha c98e6d119b89b8ffb755ed5b1e0d064973684675 --output research_log/T016E_run`. Python3.12.7CPU, noTorchimport. Oneformalrunexit0at2026-09-12T20:34:43.8751565Z. Freeze20:34:43.821950Z precedeseval20:34:43.837677Z. Freezea25b9ee3da0c7d4709758c8924cf1e60faca5b43f129e35d32525f4e187cdb5a; decisionsa5d6c4b864b5dcbdf28d354430d68d7b8f511004040ae6d4a7935bee51890f94; calibrations4c5e289ecd67efcf92b0c7751f1c58e6c57a69200a58ca70d77d3580b0ca264f. Hashes unchanged afterevaluation.

Files: newttie/boundary_confidence.py andboundary_confidence_run.py; exactrawDdonorboundary_probe_metrics.py;tests/test_boundary_confidence.py;T016E_spec/log/analysis/command/testreceipts; completeT016E_run(config,folds,calibrations,decisions,freeze,evaluation,summary);independentverify_t016e.py/receipt/log;HANDOFF. Sourceinputs are frozenB4062e01c,C433683ec,Daa71d268 with8fullpaths/hashes inconfig. No acceptedMSE recomputation.

Failures/deviations: no implementation/test/runfailures and no scientificrecipechanges. Inherited OOF dependence is a property of the explicitly requested protocol, flagged before execution and in finalreport; it was not repaired with unauthorized retraining. No newdata,features,head,renderer,CLIP,TTT,GPU,alternateq/grid,per-conditionthreshold,secondgate,epoch/modelselection or freshwork. Recoverymirrors to existinghome/Fwjq/TTIE are being finalized; finalhashes are in project-local T016E_delivery.json.

Full report: https://github.com/word-ky/TTIE/blob/66b597cef47d483dea2027d9ad925d8961031f98/research_log/T016E_analysis.md

Recommended next decision: review this literal developmental5/5 pass together with its narrowLRmargin and inheritedOOFdependence before any stronger interpretation. Under the specified audit, confidencefallback makes the existing rank30 scorer meet the development clauses and supports forced selection as a plausible decision-rule bottleneck; it does not establish fresh/generalized safety. StopafterT016-E. No selfmerge, freshlaunch,secondgate or unissuedfollow-on; existing15minuteheartbeat continues.
