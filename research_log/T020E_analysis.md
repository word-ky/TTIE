# T020-E analysis

The fixed binary sign probe fails the predeclared non-spatial development OOF safety contract: pooled/dark/bright means pass, clean mean safety and clean zero-harmful fail. The frozen T020-C movement decisions are unchanged in every row/axis. The observed direction-oracle ceiling is not recovered by this fixed binary learner. This does not prove that every possible sign predictor using the representation must fail.

|Group|H/H0|Beneficial/equal/harmful|No move/x only/y only/both|Sign accuracy x,y on non-center targets|Wrong directions C→E x,y|E/oracle B|
|---|---:|---|---|---|---|---:|
|nonspatial_pool|0.9522174813562284|37/63/20|62/11/13/34|35/51, 37/54|17→14, 14→15|1.0391756505789824|
|clean|1.2072654157145626|0/39/1|39/0/1/0|0/2, 1/2|0→0, 1→1|1.600838604831342|
|homogeneous_dark|0.940199947158663|22/15/3|15/4/7/14|19/21, 21/25|4→2, 3→4|1.0266502153747368|
|homogeneous_bright|0.9774581912410226|15/9/16|8/7/5/20|16/28, 15/27|13→12, 10→10|1.0623269388686238|

Literal vector: `[true,false,true,true,false]`. Pooled harmful episodes remain20 (C20→E20); clean1→1, dark4→3, bright15→16. The clean H/H0 remains exactly1.2072654157145626. Pooled wrong-direction axes change31→29, insufficient for the required safety. Full MSE, absolute E−oracle-B gaps and counts are persisted in `T020E_run/summary.json` and `mechanism.json`. Sign accuracy uses binary predictions on every held-out axis whose post-freeze reference target is non-center, including axes retained at center by the frozen necessity decision. Wrong-direction counts use final moved axes only.

All five historical image-grouped folds retained32 training/8 held-out images (96/24 episodes). Each axis trains only on non-center targets extracted from its training rows. All ten subsets contain both signs: x lower/upper by fold26/14,22/17,23/18,22/17,27/18; y29/15,29/18,26/15,27/13,29/15. No structural stop was needed.

Each fold reuses byte-identical T020-C normalization files based on all96 training rows, also independently recomputed from label-free training features. Fixed84→64→64→2 SiLU, ordinary CE, AdamW lr1e-3/wd1e-4, batch256, seed7,100 epochs, final epoch. Ten heads total. Training preserves the original small CPU head runtime; all600 candidate feature vectors were reused unchanged from the previously completed A6000 run. No new feature/CLIP, renderer, TTT, GPU or fresh-data execution.

Source `36d2033b7ca77b3aca222b10a26ce771e2618d52`. Prediction SHA256 `dcad1a6f62b0402fb989993bf19595331c925bfe06193b18596b6af2e3eaa0ce`, frozen 2026-09-13T15:25:50.045424+00:00. Independent reference-free replay finished 2026-09-13T15:26:07.374913+00:00; evaluation opened 2026-09-13T15:26:34.690548+00:00. Donor held-out classes are reduced to movement booleans before training/inference. No donor sign/logit or held-out target is consumed by the sign head.

Four focused/affected tests passed(10.16s), covering exact84-D construction, fixed tie behavior and arbitrary held-out label poisoning with equality of training state/history hashes and held-out prediction hashes for both axes. Independent verifier imports no TTIE; pre-reference replay exactly reconstructs10 heads, original normalization and120 combined decisions. Post-reference verifier checks training subset labels/counts,120 MSE joins, necessity invariance, all diagnostics, oracle distances and five clauses. Both PASS. No engineering failures or scientific deviations.

Commands: `D:/anaconda3/python.exe -m pytest tests/test_binary_sign.py tests/test_deadband_probe.py -q`; `python -m ttie.binary_sign train --output research_log/T020E_run --source-sha 36d2033b7ca77b3aca222b10a26ce771e2618d52`; `python research_log/T020E_verify.py replay --output research_log/T020E_run`; `python -m ttie.binary_sign evaluate --output research_log/T020E_run`; `python research_log/T020E_verify.py metrics --output research_log/T020E_run`. All local Python invocations use D:/anaconda3/python.exe.

Development OOF only, no final selector or fresh qualification. No threshold rescue, necessity head, new seed, heterogeneous combined training or further experiment. Stopped for research-lead review.

negative 3/5
