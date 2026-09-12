The T016-E confidence pass does not survive proper nesting: T016-F passes4/5 fixed development clauses. Left/right selected MSE is1.018496xRegion2, exceeding the1.01 limit. This closes the prescribed fixed scalar-head/confidence cycle without retuning.

Exactly25freshCPU rank30 heads were trained: five outer32-ID heads and four24-ID inner heads per outer fold. Every outer-held-out ID is excluded from allfive heads and calibration for its fold; each calibration image receives a fresh inner-OOF score from a head that excluded it. D's training recipe and E's confidence/grid/ties are unchanged. Allheads/innerOOFs/thresholds/decisions are frozen before combined evaluation.

| Group | MSE | /Region2 | /Hard oracle | /Ungated nested | /Old E |
|---|---:|---:|---:|---:|---:|
| Spatial | .0339618805136 | .969132907439 | 1.04294112194 | .992510391456 | 1.00168669041 |
| Left/right | .0340147255687 | 1.01849596586 | 1.05731064017 | .972899332945 | 1.00852872791 |
| Quadrants | .0309838496498 | 1 | 1.00005362091 | .983995820478 | 1 |
| Offset | .0368870663224 | .905207278421 | 1.06802882605 | 1.01885385589 | .996862729878 |

Thresholds [.5,.5,.5,1,.75].35adaptive/85canonical;27beneficial,6harmful(allleft/right),2zero. Clausevector [true,true,true,false,true]. This is development-only on40already-inspectedIDs, with no fresh or deployable claim.

Validation: D/Ebaseline12testsPASS5.930s; nestedkernel2PASS10.130s; integrated3PASS7.684s; py_compilePASS. Five real heads and allscores/thresholds/decisions are invariant to outer-held-out reference mutation. Independent audit verifies67sourcefiles,14inputartifacts,25headnormalizations,5400exact saved-head scores,2500pairpermutations,40calibrationmeans,allfoldexclusions andmetrics/clauses. Formal runexit0; no failures/tuning. All25heads/histories andfold/combinedfreeze evidence committed.

The freshly trained outerheads reproduce historical D's1080scores exactly, verified after formal evaluation, as expected from the identical outerdata/seed/recipe. No historicalhead/score is reused in the primary nested pipeline. No newdata,renderer,CLIP,TTT,GPU,feature orreferenceMSE recomputation.

Frozen sourcefb9d33e9c10e50cd88b389411adf56601ea3cc68; evidence45bc8f6dfa536a5adc351929b335f750ab82487e.
Full report: https://github.com/word-ky/TTIE/blob/45bc8f6dfa536a5adc351929b335f750ab82487e/research_log/T016F_analysis.md

Research review only. No self-merge, oldPRtopology repair or follow-on experiment.