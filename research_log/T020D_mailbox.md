

---

## T020-D — DONE — direction-dominant

UTC: 2026-09-13T13:42:20.779050+00:00

- PR: https://github.com/word-ky/TTIE/pull/39 ; branch `codex/T020D-failure-attribution`.
- Source: `b931f3b22ec83fbd2d395f5053f2bb174f1f61f7`; evidence: `8f6d29eadc67f8d077cc891f827235a067ee3fd4`.
- Used only nine immutable accepted T020-C/T020-B artifacts; all hashes linked against accepted freeze/report receipts before analysis. Original OOF prediction hash unchanged: `731258100e6a71ec439935a06349c7c40851e35842bee0d72df3a45de67e01e8`.

|Oracle|Group|H/H0|Beneficial/equal/harmful|No move/x only/y only/both|
|---|---|---:|---|---|
|A|nonspatial_pool|0.9348168370038701|42/59/19|59/7/10/44|
|A|clean|1.2294721860512596|0/38/2|38/0/0/2|
|A|homogeneous_dark|0.9271463660341092|23/14/3|14/1/5/20|
|A|homogeneous_bright|0.9488845608685621|19/7/14|7/6/5/22|
|B|nonspatial_pool|0.9163200473622485|55/63/2|62/11/13/34|
|B|clean|0.7541456159734198|1/39/0|39/0/1/0|
|B|homogeneous_dark|0.9157938439777966|25/15/0|15/4/7/14|
|B|homogeneous_bright|0.9201105191609033|29/9/2|8/7/5/20|

A necessity-oracle/predicted-sign: `[true,false,true,true,false]` (3/5). B frozen-necessity/direction-oracle: `[true,true,true,true,true]` (5/5). Literal interpretation: **direction-dominant**. These are reference-only development counterfactuals, not deployable selectors. B retains all original move/no-move decisions and false-move signs; its2 harmful bright rows remain despite passing the prescribed clauses.

Axis category order: correct_center / false_move / missed_move / correct_move_direction / wrong_move_direction.

|Group|x counts|y counts|
|---|---|---|
|nonspatial_pool|63/6/12/22/17|61/5/12/28/14|
|clean|38/0/2/0/0|38/0/1/0/1|
|homogeneous_dark|19/0/3/14/4|15/0/4/18/3|
|homogeneous_bright|6/6/7/8/13|8/5/7/10/10|

Among20 harmful original episodes:19 contain wrong direction,5 false move,6 missed move. Overlap patterns:9 wrong-direction only;6 missed+wrong;4 false+wrong;1 false only. Clean harmful development row has missed x movement and wrong y direction, with no false move. No T020-A fresh row was opened.

- Files: new `ttie/movement_attribution.py`, focused tests, independent `research_log/T020D_verify.py`, immutable input origins, definitions,120-row axis errors,20-row harmful overlap table, both120-row counterfactual decisions/full summaries, verification receipts and `T020D_analysis.md`.
- Commands/tests: `D:/anaconda3/python.exe -m pytest tests/test_movement_attribution.py -q` =>2 passed0.04s; `python -m ttie.movement_attribution --output research_log/T020D_run --source-sha b931f3b22ec83fbd2d395f5053f2bb174f1f61f7`; `python research_log/T020D_verify.py --output research_log/T020D_run` =>PASS. Independent verifier imports no diagnostic implementation and reproduces all decisions, counts, overlaps, group MSEs/outcomes/movements, five clauses and interpretation.
- Failures/deviations: none. CPU standard-library table arithmetic; no GPU work needed because there is no training, feature recomputation, rendering or TTT. No threshold change, search, new model or fresh artifacts.
- Interpretation/next: prescribed counterfactual test identifies direction-sign failure rather than false-move dominance. It does not show that a learned direction repair will generalize. Stopped after T020-D; await research-lead review, no T020-E. Research-owned inbox/state unchanged.
