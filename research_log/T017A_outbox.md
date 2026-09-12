

# T017-A — DONE — reference-only local geometry negative (4/5)

UTC: 2026-09-12T22:35:02.645429+00:00. Issuedinboxc49b06a7/state61dd70c2. Scientific source **5247690887078540fd5bfe74c90052efe0f4fb93**; evidence **bb45bd71dbaf512ce957f041cbcee1bcde6c9138**; branch `codex/T017A-local-geometry-landscape`; PR22 https://github.com/word-ky/TTIE/pull/22 ready/unmerged.

T017-A passes4/5 fixed reference-only viability clauses. The five-point tau=.05 local rule captures69.58%ofpoolednine-hardoracleheadroom, but quadrants MSE is1.03303xRegion2 and violates the1.01 safety limit. This coarse local-axis diagnostic does not yet justify continuous geometry optimization.

| Group | Selected MSE | /Region2 | /Hard oracle | Oracle gain captured |
|---|---:|---:|---:|---:|
| Spatial | .0333179102473 | .950756634776 | 1.02316533032 | .695829467285 |
| Left/right | .0329902476864 | .987820234308 | 1.02546586273 | .331776476 |
| Quadrants | .0320072673145 | 1.03303068135 | 1.03308607339 | -616.036747393 |
| Offset | .0349562157411 | .85782427473 | 1.01212294127 | .932602442143 |

The quadrant capture ratio has a tiny positive denominator:39/40episodes have zerooracleheadroom. AbsoluteMSE and11harmful/29zero/no-beneficial quadrant cases make the failure clear. Acrossall120episodes:61beneficial,20harmful,39zero;51zerooracleheadroom ratios are null;12tiedoracles remain explicit. No clipping or alternative pass route.

The rule reads only the fixed5softreferenceMSEs, chooses x/y independently with center-first/lower-coordinate ties, and selects an existing hardcandidate. All120decisions are frozen/hash-bound before familylabels are attached. gx/gy are diagnostic finite differences only. **Reference-only, not a label-free or deployable selector.** No model,OOF,threshold,image,CLIP,TTT,rendering,GPU orfreshdata work.

Validation: exact120x27candidate grid,120canonicalequalities and1080hardvalue matches reproduce acceptedBoracle beforeanalysis. Kernel3testsPASS; final6testsPASS0.062s; py_compilePASS. Independent arrayaudit verifies4sourcefiles,6inputartifacts,120choices,240differences,allcounts/ties/gains/quantiles/clauses andfreezetiming. Runexit0, no failures or outcome-driven changes.

Frozen source5247690887078540fd5bfe74c90052efe0f4fb93; evidencebb45bd71dbaf512ce957f041cbcee1bcde6c9138.
Report: https://github.com/word-ky/TTIE/blob/bb45bd71dbaf512ce957f041cbcee1bcde6c9138/research_log/T017A_analysis.md

Stop after this boundednegative; it does not rule out all learnedspatialbases. No self-merge, derivative-objective training or fresh evaluation.

Literalvector **[true,true,true,true,false]**. Pooled4.9243%improvement and69.5829%oracleheadroomcapture do notoverrideQ3.3031%degradation. Left/rightimproves1.2180%,offset14.2176%;offsetcaptures93.2602%ofitshardoracleheadroom. Qhas11harmful/29zero episodes, no beneficialcase. Its-616.04aggregatecapture ratio is finite because meanoracleheadroom isonly1.6612932085990905e-6;39Qepisodes havezeroheadroom. Per-episodeundefinedratiosareexplicitnull;negativevaluesnotclipped. Do notinterpretthatlargefiniteaggregate as a robustpercentage.

spatial_pool: localcounts=[18, 5, 10, 18, 39, 13, 7, 4, 6]; oraclecounts=[27, 2, 12, 23, 41, 14, 0, 1, 0]; disagreement=0.4166666666666667; outside/insideoracleties=(0.325, 0.675); zerooracleheadroom/tiedoracles=(51, 12).

gx={'negative': 51, 'zero': 7, 'positive': 62, 'distribution': {'count': 120, 'null_count': 0, 'mean': 0.005938434818138679, 'median': 0.0008760159835219383, 'quantiles': {'0': -0.058259498327970505, '5': -0.0369613841176033, '25': -0.009978124871850014, '50': 0.0008760159835219383, '75': 0.01983996480703354, '95': 0.056567099411040545, '100': 0.11242068372666836}}}; gy={'negative': 50, 'zero': 12, 'positive': 58, 'distribution': {'count': 120, 'null_count': 0, 'mean': 0.0004091294249519706, 'median': 0.0, 'quantiles': {'0': -0.046944282948970795, '5': -0.037081437185406685, '25': -0.006427820771932602, '50': 0.0, '75': 0.008098329417407513, '95': 0.031969896517693996, '100': 0.045635709539055824}}}.

left_right: localcounts=[2, 1, 0, 15, 5, 10, 4, 0, 3]; oraclecounts=[2, 0, 0, 23, 2, 13, 0, 0, 0]; disagreement=0.45; outside/insideoracleties=(0.35, 0.65); zerooracleheadroom/tiedoracles=(7, 5).

gx={'negative': 23, 'zero': 2, 'positive': 15, 'distribution': {'count': 40, 'null_count': 0, 'mean': -0.006741668679751456, 'median': -0.0071463268250226974, 'quantiles': {'0': -0.04297932609915733, '5': -0.04092911072075367, '25': -0.01976239262148738, '50': -0.0071463268250226974, '75': 0.004248428158462048, '95': 0.02923724520951504, '100': 0.05049875006079674}}}; gy={'negative': 13, 'zero': 5, 'positive': 22, 'distribution': {'count': 40, 'null_count': 0, 'mean': 0.003097150009125471, 'median': 0.0003345124423503876, 'quantiles': {'0': -0.04214947111904621, '5': -0.03170506842434406, '25': -0.0003226008266210556, '50': 0.0003345124423503876, '75': 0.010898734908550978, '95': 0.031978415325284004, '100': 0.045635709539055824}}}.

quadrants: localcounts=[0, 2, 1, 0, 29, 0, 2, 4, 2]; oraclecounts=[0, 0, 0, 0, 39, 0, 0, 1, 0]; disagreement=0.275; outside/insideoracleties=(0.275, 0.725); zerooracleheadroom/tiedoracles=(39, 0).

gx={'negative': 24, 'zero': 0, 'positive': 16, 'distribution': {'count': 40, 'null_count': 0, 'mean': -0.002653270959854126, 'median': -0.0028307829052209854, 'quantiles': {'0': -0.058259498327970505, '5': -0.036992503330111504, '25': -0.016148178838193417, '50': -0.0028307829052209854, '75': 0.012116036377847195, '95': 0.030023199971765266, '100': 0.06906064227223396}}}; gy={'negative': 24, 'zero': 0, 'positive': 16, 'distribution': {'count': 40, 'null_count': 0, 'mean': -0.0044512980384752154, 'median': -0.005525751039385796, 'quantiles': {'0': -0.046936264261603355, '5': -0.03722250368446112, '25': -0.012578433379530907, '50': -0.005525751039385796, '75': 0.005167587660253048, '95': 0.01979864295572041, '100': 0.03145109862089157}}}.

offset_left_right_40: localcounts=[16, 2, 9, 3, 5, 3, 1, 0, 1]; oraclecounts=[25, 2, 12, 0, 0, 1, 0, 0, 0]; disagreement=0.525; outside/insideoracleties=(0.35, 0.65); zerooracleheadroom/tiedoracles=(5, 7).

gx={'negative': 4, 'zero': 5, 'positive': 31, 'distribution': {'count': 40, 'null_count': 0, 'mean': 0.02721024409402162, 'median': 0.020044599659740925, 'quantiles': {'0': -0.011749565601348877, '5': -0.003911491483449936, '25': 0.0017051491886377335, '50': 0.020044599659740925, '75': 0.04835523199290037, '95': 0.07551572844386095, '100': 0.11242068372666836}}}; gy={'negative': 13, 'zero': 7, 'positive': 20, 'distribution': {'count': 40, 'null_count': 0, 'mean': 0.002581536304205656, 'median': 2.9713846743106842e-05, 'quantiles': {'0': -0.046944282948970795, '5': -0.03277507983148098, '25': -0.00126721803098917, '50': 2.9713846743106842e-05, '75': 0.009322864934802055, '95': 0.03950638929381966, '100': 0.04292067140340805}}}.

Per-conditionabsolute/relativegain,oracle-gap/available-gain/captured-gain distributions are preserved in fullreport/summary andall120episodevalues inevaluation.json. Linearquantiles0/5/25/50/75/95/100predeclaredfordescriptiononly. Aggregatecapturedgain is ratioofaggregate gains; per-episodecaptureratiodistribution is separate. Zero-gain,tiedoracle andzero-denominatorcases remainexplicit.

Baselinebinding: Aee5d8fdaf3ab48ee7ad3654d45bdc65419be8367 candidate_metrics/config/summary/sanity/local_verification; B4062e01cb93de731c394015c5ac741d6c08e04d8 evaluation used only to reproduce accepted9hardvalues/oracle. Candidate-tableSHA2566091a0c928f115940997a647693b6571d8608131e7f9567a75882f0233b9c41e. Exact120x27grid;canonical120equal;hard1080exactmatches;pooled9hardoracle.032563564518932255reproducedbeforelocaldiagnostic. FullA27oracle andRegion2alsoagree. Precheckdoesnotusefamilylabels; localruleonlyindices[13,4,22,10,16]. No source_directory/imageID/condition/corners orothercandidateMSEs enterlocal_choice.

Validation: kernel3testsPASS0.000s; final6focusedtestsPASS0.062s;py_compilePASS(overlappingincrementchecks,not9unique). Testsreadguardfivecrossvalues,exactaxis/center/lower-coordinate ties/finite differences,noncrossandmetadata mutationinvariance,hash/candidatecontract,strict1.03oracleclause,zerodenominatornulls/tiedoracles andall120choicesfrozenbeforefamilyreporting. IndependentNumPyarrayauditverifies4source/6inputhashes,120choices,240finite differences,canonical120/hard1080matches,allratios/gaps/capturedgains/boundarycounts/ties/signs/quantiles/fiveclauses. No implementation/test/runfailure orpost-resultchange.

Command: `D:/anaconda3/python.exe -m ttie.local_geometry --source-sha 5247690887078540fd5bfe74c90052efe0f4fb93 --output research_log/T017A_run`. Python3.12.7CPU,stdlibraryanalysis. Oneformalrunexit0at2026-09-12T22:29:25.9205182Z.120choicesfrozen22:29:25.882587Z;familyreportingstart22:29:25.889597Z. DecisionsSHA2568aefe5e88aa823e6d415bb1580a00765aaf507e4a106766fef5495e238a1bc26;freezeSHA256b452ea83955af9a578f0c7cdf5e66ef5438a30d7bf9005e8440c36685ed03a7e. Hashesunchangedafterreporting. No images,cleanfiles,largepacks,CLIPfeaturevectors,TTTtrajectories,model,OOF,calibration,training,rerendering,A6000ornewdata reads/work. ExistingcandidateMSEvaluesonly.

Files: compactttie/local_geometry.py andtests/test_local_geometry.py;spec/baseline/test/runlogs/command/analysis;T017A_run baseline/config/decisions/freeze/evaluation/summary/receipt;verify_t017a.py andverificationJSON/log;HANDOFF. All6inputartifactpaths/commits/hashes inconfig. FouractualscientificruntimefilesbinddeclaredsourceSHA usingexistingprovenancehelper. No scientificdeviation.

Recommendednextdecision: preservecontrollednegative/inconclusive **4/5**. Thiscoarsetau.05local-axisreferencecrossdoesnotyetjustifycontinuousgeometryoptimization; itdoesnotruleoutalllearnedspatialbases. Thechoicesareexplicitlyreference-only,neverlabel-freetest-timeselection. StopafterT017-A; no derivative-supervisedobjective,geometryoptimizer,freshdataorunissuedfuturetaskstarted. No selfmerge. Evidencepushed; home/Frecoverymirrorsbeingfinalized, finalhashesinT017A_delivery.json. Existing15minuteheartbeatcontinues.
