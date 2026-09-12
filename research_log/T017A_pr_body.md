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