# T020-D analysis

Reference-only development diagnosis using accepted frozen T020-C predictions and T020-B targets/nine-hard MSE. No model training, feature extraction, TTT, new seed, threshold search or fresh artifacts. Standard-library CPU table arithmetic only.

The predeclared false-movement-dominance hypothesis is not supported. Among the20 originally harmful episodes,19 contain a wrong-direction axis,5 contain a false-move axis and6 contain a missed-move axis; overlaps are explicit. Exact patterns:9 wrong-direction only;6 missed-move plus wrong-direction;4 false-move plus wrong-direction;1 false-move only. Counts are associative; the prescribed oracle comparison supplies the interpretation.

|Counterfactual|Group|H/H0|Beneficial/equal/harmful|No move/x only/y only/both|
|---|---|---:|---|---|
|A|nonspatial_pool|0.9348168370038701|42/59/19|59/7/10/44|
|A|clean|1.2294721860512596|0/38/2|38/0/0/2|
|A|homogeneous_dark|0.9271463660341092|23/14/3|14/1/5/20|
|A|homogeneous_bright|0.9488845608685621|19/7/14|7/6/5/22|
|B|nonspatial_pool|0.9163200473622485|55/63/2|62/11/13/34|
|B|clean|0.7541456159734198|1/39/0|39/0/1/0|
|B|homogeneous_dark|0.9157938439777966|25/15/0|15/4/7/14|
|B|homogeneous_bright|0.9201105191609033|29/9/2|8/7/5/20|

A uses target move/no-move and frozen lower/upper logits, with lower-first tie break; it fails clean mean safety and clean zero-harmful: `[true,false,true,true,false]`. B retains frozen move/no-move and only replaces direction where target movement is useful; it passes `[true,true,true,true,true]`. B leaves false moves intact and still has2 harmful bright episodes, permitted by the original group-mean and clean-only zero-harm criteria. Thus5/5 does not mean universal zero harm.

The original clean harmful development episode contains an x missed-move and a y wrong-direction, and no false move. Across40 clean rows x errors are2 missed moves; y errors are1 missed move and1 wrong direction. All11 false-move axes occur in bright. Full x/y category counts and120-row attribution are in `T020D_run/axis_counts.json` and `axis_errors.json`; the20-row overlapping-harm table is `harmful_overlap.json`.

Nine immutable input origins are hash-checked and linked to accepted receipts before analysis. The original prediction SHA256 remains `731258100e6a71ec439935a06349c7c40851e35842bee0d72df3a45de67e01e8`. Source `b931f3b22ec83fbd2d395f5053f2bb174f1f61f7` froze both definitions before analysis. Two focused tests pass(0.04s), covering all nine class pairs, lower-first tie, ignored center logit, and preservation of false-move direction. Independent stdlib verifier imports no TTIE and reconstructs all120 rows, both oracle choices, counts, overlaps, metrics, literal clauses and interpretation from accepted Git blobs; PASS. No failures or scientific deviations.

Commands: `D:/anaconda3/python.exe -m pytest tests/test_movement_attribution.py -q`; `D:/anaconda3/python.exe -m ttie.movement_attribution --output research_log/T020D_run --source-sha b931f3b22ec83fbd2d395f5053f2bb174f1f61f7`; `D:/anaconda3/python.exe research_log/T020D_verify.py --output research_log/T020D_run`.

The diagnosis identifies direction sign as the bottleneck under the specified oracle test; it does not establish that a learned direction repair will generalize. Both counterfactuals use development references and are not deployable. Stopped after this audit; no T020-E or new model.

direction-dominant
