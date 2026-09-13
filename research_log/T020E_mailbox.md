

---

## T020-E — DONE — negative 3/5

UTC: 2026-09-13T15:29:08.166576+00:00

- PR: https://github.com/word-ky/TTIE/pull/40 ; branch `codex/T020E-binary-sign`.
- Source: `36d2033b7ca77b3aca222b10a26ce771e2618d52`; evidence: `784445899bc7ab59b9e5c98f2757ede1ce8bef9e`.
- Exactly10 binary heads, original40-image/120-episode non-spatial development set and historical5 image-grouped folds. Frozen T020-C necessity unchanged for all240 axes. Reused exact cached features and each fold all96-training-row normalization, without refitting on target-selected subsets.
- All fold/axis subsets contain both sign classes. x lower/upper by fold:26/14,22/17,23/18,22/17,27/18. y:29/15,29/18,26/15,27/13,29/15. Fixed84→64→64→2 SiLU/unweighted CE/AdamW1e-3/wd1e-4/batch256/seed7/100 epochs/final epoch; only fold-training non-center targets supervise sign.

|Group|H/H0|Beneficial/equal/harmful|No move/x only/y only/both|Sign accuracy x,y|Wrong-direction axes C→E x,y|E/oracle B|
|---|---:|---|---|---|---|---:|
|nonspatial_pool|0.9522174813562284|37/63/20|62/11/13/34|35/51, 37/54|17→14, 14→15|1.0391756505789824|
|clean|1.2072654157145626|0/39/1|39/0/1/0|0/2, 1/2|0→0, 1→1|1.600838604831342|
|homogeneous_dark|0.940199947158663|22/15/3|15/4/7/14|19/21, 21/25|4→2, 3→4|1.0266502153747368|
|homogeneous_bright|0.9774581912410226|15/9/16|8/7/5/20|16/28, 15/27|13→12, 10→10|1.0623269388686238|

Literal acceptance vector `[true,false,true,true,false]`: pooled/dark/bright safety pass; clean safety and clean zero-harmful fail. Pooled harmful20→20; clean1→1, dark4→3, bright15→16. Sign accuracy is measured on every held-out non-center target axis after freeze; wrong-direction counts use final moved axes only. Full absolute E−oracle-B gaps/MSEs are in `T020E_run/mechanism.json` and `summary.json`.

- Prediction SHA256: `dcad1a6f62b0402fb989993bf19595331c925bfe06193b18596b6af2e3eaa0ce`, frozen2026-09-13T15:25:50.045424Z before independent prediction replay and held-out reference evaluation.
- Files: `ttie/binary_sign.py`, focused tests, independent `research_log/T020E_verify.py`,27 immutable input origins,10 subset counts/heads/normalizers/receipts,120 binary logits/classes/combined decisions, evaluation/mechanism evidence and `T020E_analysis.md`.
- Tests: `python -m pytest tests/test_binary_sign.py tests/test_deadband_probe.py -q` =>4 passed10.16s; poison held-out labels leaves both-axis training-state/history hashes and prediction hashes unchanged. Independent verifier imports no TTIE, reconstructs all10 heads/normalizers/120 combined predictions before opening references, then verifies training subsets, metrics/diagnostics/oracle gaps and five clauses; both stages PASS.
- Commands: `python -m ttie.binary_sign train --output research_log/T020E_run --source-sha 36d2033b7ca77b3aca222b10a26ce771e2618d52`; `python research_log/T020E_verify.py replay --output research_log/T020E_run`; `python -m ttie.binary_sign evaluate --output research_log/T020E_run`; `python research_log/T020E_verify.py metrics --output research_log/T020E_run`. Local executable D:/anaconda3/python.exe; original tiny CPU learner runtime retained. No new GPU/features/TTT/render execution was needed.
- Failures/deviations: none. No T020-A fresh artifacts, thresholds, second seed, resampling, necessity model or heterogeneous training.
- Interpretation: the fixed binary sign learner does not recover the direction-oracle safety ceiling on non-spatial development OOF. This is not a universal impossibility claim about the representation. No final model or fresh qualification. Stopped after T020-E; await research-lead review. Research-owned state/inbox unchanged.
