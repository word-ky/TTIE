# T016-F: fully nested rank30 + single confidence fallback

Result: **t016e_pass_does_not_survive_fully_nested_development_audit**, **4/5**.

Five outer32/8ID splits; each outertrain32 split into four24/8ID innerfits. Exactly25freshheads using the unchangedDrecipe; no oldhead/score used for primarynested decisions.
Thresholds: [0.5, 0.5, 0.5, 1.0, 0.75].

Literalclauses: {'spatial_improves_region2_3pct': True, 'spatial_within_hard_oracle_5pct': True, 'offset_improves_region2_5pct': True, 'left_right_no_more_than_1pct_worse': False, 'quadrants_no_more_than_1pct_worse': True}

| Group | Selected MSE | /Region2 | /Hard oracle | /Ungated nested | /Old E | Adaptive / Canonical | Adapted gains |
|---|---:|---:|---:|---:|---:|---|---|
| spatial_pool | 0.0339618805136 | 0.969132907439 | 1.04294112194 | 0.992510391456 | 1.00168669041 | 35 / 85 | {'beneficial': 27, 'harmful': 6, 'zero': 2} |
| left_right | 0.0340147255687 | 1.01849596586 | 1.05731064017 | 0.972899332945 | 1.00852872791 | 10 / 30 | {'beneficial': 3, 'harmful': 6, 'zero': 1} |
| quadrants | 0.0309838496498 | 1 | 1.00005362091 | 0.983995820478 | 1 | 0 / 40 | {'beneficial': 0, 'harmful': 0, 'zero': 0} |
| offset_left_right_40 | 0.0368870663224 | 0.905207278421 | 1.06802882605 | 1.01885385589 | 0.996862729878 | 25 / 15 | {'beneficial': 24, 'harmful': 0, 'zero': 1} |

| Group | q distribution | Disagreement / Outside oracle ties | Selected / Oracle counts |
|---|---|---|---|
| spatial_pool | {'count': 120, 'mean': 0.26783018609886866, 'median': 0.2841843780692793, 'quantiles': {'0': -1.2891933917858664, '25': -0.2953141379989333, '50': 0.2841843780692793, '75': 0.858051015485551, '100': 2.6785315226655437}} | 0.5166666666666667 / 0.44166666666666665 | [12, 5, 8, 3, 85, 4, 2, 0, 1] / [27, 2, 12, 23, 41, 14, 0, 1, 0] |
| left_right | {'count': 40, 'mean': 0.4206974754985484, 'median': 0.3787803145730181, 'quantiles': {'0': -1.0805638129958552, '25': 0.07240065938100392, '50': 0.3787803145730181, '75': 0.6625348711319365, '100': 1.6627812289570538}} | 0.9 / 0.8 | [3, 0, 0, 2, 30, 2, 2, 0, 1] / [2, 0, 0, 23, 2, 13, 0, 0, 0] |
| quadrants | {'count': 40, 'mean': -0.5554363921581098, 'median': -0.661405003621444, 'quantiles': {'0': -1.2891933917858664, '25': -0.8420760858641135, '50': -0.661405003621444, '75': -0.23440179585129284, '100': 0.5125146666684557}} | 0.025 / 0.025 | [0, 0, 0, 0, 40, 0, 0, 0, 0] / [0, 0, 0, 0, 39, 0, 0, 1, 0] |
| offset_left_right_40 | {'count': 40, 'mean': 0.9382294749561674, 'median': 0.8933289597844493, 'quantiles': {'0': -0.4852272503878511, '25': 0.5332513795735243, '50': 0.8933289597844493, '75': 1.3625481363084821, '100': 2.6785315226655437}} | 0.625 / 0.5 | [9, 5, 8, 1, 15, 2, 0, 0, 0] / [25, 2, 12, 0, 0, 1, 0, 0, 0] |

| Fold | Threshold | Selected MSE | Adaptive / Canonical | Per-condition adaptive/canonical |
|---|---:|---:|---|---|
| 0 | 0.5 | 0.0421568798677 | 9 / 15 | {'left_right': (3, 5), 'quadrants': (0, 8), 'offset_left_right_40': (6, 2)} |
| 1 | 0.5 | 0.0211793526929 | 8 / 16 | {'left_right': (2, 6), 'quadrants': (0, 8), 'offset_left_right_40': (6, 2)} |
| 2 | 0.5 | 0.0406880689552 | 6 / 18 | {'left_right': (1, 7), 'quadrants': (0, 8), 'offset_left_right_40': (5, 3)} |
| 3 | 1.0 | 0.0388676455865 | 6 / 18 | {'left_right': (1, 7), 'quadrants': (0, 8), 'offset_left_right_40': (5, 3)} |
| 4 | 0.75 | 0.0269174554657 | 6 / 18 | {'left_right': (3, 5), 'quadrants': (0, 8), 'offset_left_right_40': (3, 5)} |

Per-head histories, pair counts, normalization, calibration tables, innerOOFs and allfold hashes are persisted. q uses the unchanged Epopulationstd/floor formula andlinear0/25/50/75/100percentiles for description only.
Every fold excludes its outer-heldout IDs from allfiveheadtraining sets and thresholdcalibration. Each calibrationimage is also excluded from its own innerOOFranker training. All25heads andfivefolddecisions are frozen before combinedouterreferenceevaluation.
Canonical candidate4 versus Region2 maxMSEdifference: 0.0.

Development-only, on already-inspected40IDs. Historical E has inherited scorer dependence and is a comparator, not a qualified baseline. No fresh qualification or deployablethreshold claim. Stop after the literal result; no alternategrid,model,confidence orfollow-onwork.

## Fixed calibration tables

| Threshold | Fold0 meanMSE | Fold1 | Fold2 | Fold3 | Fold4 |
|---|---:|---:|---:|---:|---:|
| 0.0 | 0.0323887827253202 | 0.0380277652244937 | 0.0328150021377951 | 0.0334106058047231 | 0.0365314145359055 |
| 0.25 | 0.0323281964568499 | 0.0377851319741846 | 0.0327071790961782 | 0.0332363555159342 | 0.0362931735483774 |
| 0.5 | 0.0322371654377397 | 0.0377346462191781 | 0.0324694425653433 | 0.033220194488725 | 0.0361235061718617 |
| 0.75 | 0.0323269200986639 | 0.0378555956267519 | 0.032668480615636 | 0.0332003363437252 | 0.036104501215353 |
| 1.0 | 0.0327955882191115 | 0.0380036858502232 | 0.0328663537463096 | 0.0331776268479492 | 0.0363026453026881 |
| 1.5 | 0.0328703531364833 | 0.0378671580159183 | 0.0330705755477538 | 0.0337594973534578 | 0.0365963632357307 |
| 2.0 | 0.0329581531065439 | 0.0379764405321718 | 0.0333976573165273 | 0.0340968264451173 | 0.036882435476097 |
| inf | 0.0330484863904227 | 0.0380087606075297 | 0.0333773816479758 | 0.0338684845434424 | 0.0369147555514549 |

## All25heads

| Outer fold | Role | Train IDs | Non-tied pairs | Final epoch train pair loss |
|---|---|---:|---:|---:|
| 0 | outer_head | 32 | 3249 | 0.0695690337273846 |
| 0 | inner0 | 24 | 2421 | 0.0684757322897748 |
| 0 | inner1 | 24 | 2385 | 0.0627531054447282 |
| 0 | inner2 | 24 | 2430 | 0.073116075241394 |
| 0 | inner3 | 24 | 2511 | 0.0710158040891696 |
| 1 | outer_head | 32 | 3195 | 0.0729809935827024 |
| 1 | inner0 | 24 | 2421 | 0.0684757322897748 |
| 1 | inner1 | 24 | 2331 | 0.0826742033118958 |
| 1 | inner2 | 24 | 2376 | 0.0754919859144824 |
| 1 | inner3 | 24 | 2457 | 0.0846893539182966 |
| 2 | outer_head | 32 | 3159 | 0.0727130397399531 |
| 2 | inner0 | 24 | 2385 | 0.0627531054447282 |
| 2 | inner1 | 24 | 2331 | 0.0826742033118958 |
| 2 | inner2 | 24 | 2340 | 0.0837724750240644 |
| 2 | inner3 | 24 | 2421 | 0.0665913154761538 |
| 3 | outer_head | 32 | 3204 | 0.0747185755200154 |
| 3 | inner0 | 24 | 2430 | 0.073116075241394 |
| 3 | inner1 | 24 | 2376 | 0.0754919859144824 |
| 3 | inner2 | 24 | 2340 | 0.0837724750240644 |
| 3 | inner3 | 24 | 2466 | 0.0779438847275594 |
| 4 | outer_head | 32 | 3285 | 0.0765042268692358 |
| 4 | inner0 | 24 | 2511 | 0.0710158040891696 |
| 4 | inner1 | 24 | 2457 | 0.0846893539182966 |
| 4 | inner2 | 24 | 2421 | 0.0665913154761538 |
| 4 | inner3 | 24 | 2466 | 0.0779438847275594 |

## Implementation, provenance and validation

Frozen scientific source **fb9d33e9c10e50cd88b389411adf56601ea3cc68**; branch codex/T016F-fully-nested-confidence. Issued inboxd6904f7e/researchstate77384bb0. Daa71d268294e35f5df67c76eada29f9bec117abe training, feature/fold, fit and metric code is reused raw-byte unchanged; E66b597cef47d483dea2027d9ad925d8961031f98 confidence/calibration/diagnostic code is also raw-byte unchanged. New logic is confined to inner-fold construction, isolated five-head outer pipelines, freezing and combined report. Every head uses the fixedD30SiLU64/64 logistic recipe100epochs/seed7/AdamW1e-3/1e-4, every non-tied pair once per epoch, train-only x normalization, yidentity, finalepoch. Final train loss is the epoch100sample-weighted minibatch average, not a second final-weight sweep.

The outer pipeline's only reference access is the outer-train32ID subset. The outer ranker sees all32IDs; each inner ranker sees24 and predicts the other8; calibration uses exactly96innerOOFepisodes from those32IDs. No outer-heldout reference reaches any of the five heads, innerOOF or threshold. ID metadata controls partitions only; modelinput is exactly saved28features+gx/gy. Neither condition nor ID nor reference is a model feature. The confidence function receives scores only. Allfolds are finalized before any combined heldout reference evaluation; this is stronger ordering than per-fold evaluation after its own freeze.

Baseline12D/EtestsPASS5.930s; nestedkernel2PASS10.130s; final3nestedfocusedtestsPASS7.684s; py_compilePASS. Incrementtests overlap and are not17distincttests. The realfiveheadfixture is trained twice under outerheldout reference mutation; allfiveweights/normalizations,innerOOFs,thresholds,outer scores/decisions remain unchanged. Guardedreference lookup rejects forbidden directreads. The originalDhelper reproduces the same outerprediction. Structural tests use40IDs/3conditions and exactouter32/8inner24/8 partitions; end-to-end mutation uses a small10IDfixture with the unchanged100epochrecipe. Freeze-before-evaluation and ungated/E comparator arithmetic are tested. No unrelated fullsuite or acceptedexperiment rerun.

Independent verificationPASS:67scientific Gitblob/current-file hashes,14sourceartifact hashes,25headreceipts/normalizations,5outer/20inner group exclusions,all5400saved-head score values reproduced exactly,independently enumerated pair/sign hashes,2500epochpermutation/coverage records,40calibrationmeans,5thresholds,120decisions,allMSEjoins/ratios/gaincounts/coverage/qquantiles/clauses andallfoldfreezehashes. NumPyqmaximumdifference2.220446049250313e-16. No fitting occurred in the audit. Canonical candidate4MSE equals acceptedRegion2exactlyfor120/120.

The five newly trained outer heads reproduce all1080historicalD30OOF score values exactly, checked only after formal evaluation. This is expected from the same outertrainIDs, features, deterministicrecipe and seed. Consequently outer q distributions match E; the decisive new information is truly inner-cross-fitted calibration. No historicalhead or score is substituted for any primaryF fit/prediction. Separate post-run receipt T016F_historical_comparison.json binds this comparison to the original D OOF artifact.

Command: `D:/anaconda3/python.exe -m ttie.boundary_nested_run --source-sha fb9d33e9c10e50cd88b389411adf56601ea3cc68 --output research_log/T016F_run`. Python3.12.7/Torch2.13.0+cpu,oneCPUthread. Oneformalrun endedexit0at2026-09-12T21:08:22.3651008Z. No unit/integration/formalrun failures or fixes; no outcome-driven changes.

AllfoldsfreezeSHA2560109cefdc2a644e9e22c0214d0d9ac5bf65b4d39ad26197735e0181e22a2b7db; combineddecisionsc933f395e6316e128e5edcbbdcc8cd47396ff1a25cce43d75274ee1bd5ec51a6. Per-foldfreezehashes andtimes are in evaluation_receipt.json/all_folds_frozen.json; everyfoldfreeze precedes combinedreferenceevaluation at2026-09-12T21:08:21.545240Z. All hashes are unchanged after evaluation. Eachfoldfreezereceipt includes25files:fiveheads/histories/predictions/receipts plusinnerfolds/innerOOF/outerscores/calibration/decisions.

## Literal interpretation and stop

The fixed clause vector is **[true,true,true,false,true]**. PooledMSE improves3.0867%overRegion2 and remains4.2941%abovehardoracle; offset improves9.4793%; quadrantsremaincanonical. However,left/right is1.8496%worse thanRegion2, exceeding its1%limit. Of35adaptedepisodes,27arebeneficial,6harmfuland2zero; all6harmfulchanges occuronleft/right. Pooling does not override this failure.

NestedFspatialMSE is0.1687%higher than oldE; nested calibration chooses[.5,.5,.5,1,.75] rather than E's[.75]*5. These are measured contrasts, not proof of a general statistical bias magnitude. The requested binary decision is negative: **the T016-E5/5 pass does not survive properly nested development evaluation**. Stop this fixedscalar-head/singleconfidencecycle. Do not tune anothergrid,confidence,fold split,featureorhead. T014's previouslyfreshvalidatedresult and the broaderSpatialISPquestion are not overturned by this boundednegative.

No scientificdeviation,newIDs,referenceMSE/feature recomputation,rendering,CLIP,TTT,GPU/A6000experiment,rank28,alternateobjective,conditionweighting,modelselection,freshsplit orfollow-onwork. Development-only because these40IDs were already inspected; nesting removes the specified trainingdependency but does not make this a fresh evaluation. No self-merge or oldPRtopologyrepair. Awaitresearchreview/newtask.

Artifacts: complete T016F_run with25heads/histories/predictions/receipts,5innerOOFs/calibrationtables/thresholds/decisions/foldfreezes, combineddecisions/allfoldfreeze, gated/ungatedevaluation+summary, source/inputconfig andoriginalouterfolds; focusedtests/runlogs/command/spec; independentaudit andpost-runhistoricalcomparison; HANDOFF andfinaldeliveryreceipt. Sourceartifacts retain14fullGitcommit/path/hashentries inconfig. FinalPR/evidence/mainmailbox andserverrecoveryhashes are inT016F_delivery.json.
