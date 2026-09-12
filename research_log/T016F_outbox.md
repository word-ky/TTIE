

# T016-F — DONE — fully nested development negative (4/5)

UTC: 2026-09-12T21:14:00.649975+00:00. Inboxd6904f7e/state77384bb0. Branch `codex/T016F-fully-nested-confidence`; PR21 https://github.com/word-ky/TTIE/pull/21 ready/unmerged. Scientific source **fb9d33e9c10e50cd88b389411adf56601ea3cc68**, evidence **45bc8f6dfa536a5adc351929b335f750ab82487e**.

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

Implemented direct outertrainreference subset before allfivefits/calibration. Allinnerheads are trainedfromscratch; sortedoutertrain numericIDs positionmod4 defines24/8innergroups. Inputnormalization uses eachhead's owntrainingcandidate rows only; noystandardization; exactD30SiLU64/64,100epochs,seed7,AdamW1e-3/1e-4,betas.9/.999,eps1e-8,pairbatch256,allnon-tiedpairs once/epoch. SameE q/std-floor/constant0,8thresholdgrid,strictq>t andsaferexactmean ties. No condition-specificthreshold/weight,heldoutreferenceinput,identifierfeature orparametersearch.

spatial_pool: q={'count': 120, 'mean': 0.26783018609886866, 'median': 0.2841843780692793, 'quantiles': {'0': -1.2891933917858664, '25': -0.2953141379989333, '50': 0.2841843780692793, '75': 0.858051015485551, '100': 2.6785315226655437}}; selectedcounts=[12, 5, 8, 3, 85, 4, 2, 0, 1]; oraclecounts=[27, 2, 12, 23, 41, 14, 0, 1, 0]; disagreement=0.5166666666666667; outsideoracleties=0.44166666666666665.

left_right: q={'count': 40, 'mean': 0.4206974754985484, 'median': 0.3787803145730181, 'quantiles': {'0': -1.0805638129958552, '25': 0.07240065938100392, '50': 0.3787803145730181, '75': 0.6625348711319365, '100': 1.6627812289570538}}; selectedcounts=[3, 0, 0, 2, 30, 2, 2, 0, 1]; oraclecounts=[2, 0, 0, 23, 2, 13, 0, 0, 0]; disagreement=0.9; outsideoracleties=0.8.

quadrants: q={'count': 40, 'mean': -0.5554363921581098, 'median': -0.661405003621444, 'quantiles': {'0': -1.2891933917858664, '25': -0.8420760858641135, '50': -0.661405003621444, '75': -0.23440179585129284, '100': 0.5125146666684557}}; selectedcounts=[0, 0, 0, 0, 40, 0, 0, 0, 0]; oraclecounts=[0, 0, 0, 0, 39, 0, 0, 1, 0]; disagreement=0.025; outsideoracleties=0.025.

offset_left_right_40: q={'count': 40, 'mean': 0.9382294749561674, 'median': 0.8933289597844493, 'quantiles': {'0': -0.4852272503878511, '25': 0.5332513795735243, '50': 0.8933289597844493, '75': 1.3625481363084821, '100': 2.6785315226655437}}; selectedcounts=[9, 5, 8, 1, 15, 2, 0, 0, 0]; oraclecounts=[25, 2, 12, 0, 0, 1, 0, 0, 0]; disagreement=0.625; outsideoracleties=0.5.

FoldselectedMSEs: [0.042156879867737494, 0.021179352692949276, 0.04068806895520538, 0.038867645586530365, 0.02691745546568806]. Foldadaptive/canonical:9/15,8/16,6/18,6/18,6/18. Fullperconditionfoldcounts/q/gains,40calibrationmeans and25finalepochlosses/paircounts are in committedreport/JSON. qquantiles are descriptive0/25/50/75/100linear, unchangedfromE.

Testingdetail: baseline12D/EtestsPASS5.930s; kernel2PASS10.130s; final3nestedtestsPASS7.684s;compilePASS. Theseoverlap; not17distincttests. The realfiveheadtest trains the prescribed recipe on a small10IDfixture twice, withallouterheldoutMSEs mutated; fiveheadstates,norms,innerOOFs,threshold/decisions are identical. Aguardedlookuprejectsouterheldoutreferenceaccess. Structuraltests use40IDs/3conditions to verifyexactouter32/8inner24/8groups. OriginalDhelpermatchesouterfit. Allfoldfreezes precedeevaluationandcomparatorarithmetictested. No unrelatedfullsuiteoracceptedexperiment rerun.

Independentverification: all67scientificGitblob/runtime hashes,14inputartifacthashes,25trainingnorms,allouter/innerexclusions,5400exactforwardvalues,2500epochpermutation/coverage receipts,pairindices/signs,40calibrationmeans,5thresholds,120decisions,gaincounts/coverage/qquantiles/allMSEratios/clauses andfreezetimingPASS. NumPyqmaxabsdifference2.22e-16. Canonicalcandidate4 equalsRegion2referenceMSE exactly120/120. No fitting duringaudit. ThehistoricalDcomparisonwasperformedafterformalresult, notusedforprimarytraining/selection.

Runcommand: `D:/anaconda3/python.exe -m ttie.boundary_nested_run --source-sha fb9d33e9c10e50cd88b389411adf56601ea3cc68 --output research_log/T016F_run`. Python3.12.7/Torch2.13.0+cpu,oneCPUthread,exactly25formalheads. Exit0at2026-09-12T21:08:22.3651008Z; combinedreferenceevalstarted21:08:21.545240Z afterallfivefolds froze. AllfoldsfreezeSHA 0109cefdc2a644e9e22c0214d0d9ac5bf65b4d39ad26197735e0181e22a2b7db; combineddecisionsSHA c933f395e6316e128e5edcbbdcc8cd47396ff1a25cce43d75274ee1bd5ec51a6. Perfoldfreezes: {'0': '0997874f734f8275180027601bd40b4dc7a245f40d33273e6e5b9dcea9edf18d', '1': 'f254510e707aefe9b70dc5b8f72fdd4035a09351ec8854d0250b5ec316ac3503', '2': 'b70c9aa57819c5d44d2fcb5f6654c05384bd81c946824746867281d9ea9a90c0', '3': 'e9f5935fa8852f94fcb89b113725c5375516e68a141aa00072e790911fab77ba', '4': '8dc91de82d9a331f74819ce16c2c4e07d95c9109e043c3bb64bab18fedabc681'}. Hashesunchangedafterevaluation. Eachfoldbinds25files plus5headreceipts.

Files:newttie/boundary_nested.py andboundary_nested_run.py, tests/test_boundary_nested.py; rawunchangedD/Etraining/features/metrics/confidencedonors;completeT016F_run with25heads/histories/predictions/receipts,5innerOOFs/calibrationtables/decisions/freezes,combinedgated/ungatedevaluation/config/summary;spec/log/testreceipts/analysis,verify_t016f.py andverification/historicalcomparison,HANDOFF. InputsourceartifactsB4062e01c,C433683ec,E66b597ce; all14fullpaths/hashes inconfig. No feature/referenceMSE recomputation.

Failures/deviations: no implementation,test orformalrunfailure and no scientificdeviation. Literal scientificnegative preserved; no tuning afterresults. Pooled3.0867%gain doesnotcompensate forLR1.8496%degradation. FpooledMSE is0.1687%higherthanoldE; this is a measured contrast, not a general bias estimate. Propernesting removes theidentifiedEtrainingdependency but these40IDs remain alreadyinspecteddevelopmentdata, notfreshqualification. No GPU/A6000experiment,rendering,CLIP,TTT,newdata,rank28,alternateq/grid/loss/secondgate/model oroldPRtopologywork.

Recommendednextdecision: **preserve4/5negative; T016-E5/5doesnotsurvivepropernesting; stopthisfixedscalar-head/singleconfidencecycle**. Nofreshlaunch,alternatethreshold,confidence,foldsplit ormodelstarted. Awaitresearchreview/newissuedtask. Existing15minuteheartbeat continues; completedFstillOPENdoesnotauthorizearun. Engineeringevidencepushed; finalhome/Frecoverymirrors beingfinalized, withhashesinT016F_delivery.json.
