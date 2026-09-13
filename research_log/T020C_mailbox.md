

---

## T020-C — DONE — development OOF negative (3/5)

UTC: 2026-09-13T12:48:18.495152+00:00

- PR:https://github.com/word-ky/TTIE/pull/38 ; branch `codex/T020C-nonspatial-oof`.
- Scientific source:`a4f9893c4486bd7ade6138e5f44047d29e9285c0`; evidence:`34f497758eed6c26c87c43383d3668411995a22b`.
- Fixed40 development images/120 non-spatial T020-B episodes, original5 image-grouped folds(32 train/8 held-out IDs;96/24 rows each), exactly10 OOF heads. Unchanged frozen28-D extractor and unchanged T019-B84→64→64→3 SiLU/unweighted CE/AdamW1e-3/wd1e-4/batch256/seed7/100 epochs/final-epoch recipe. Per-fold training-only normalization persisted.
- Literal acceptance vector:`[true, false, true, true, false]`:pooled safety passes; clean mean-MSE safety fails; dark and bright pass; clean zero-harmful fails.

|Group|H1/H0|H1/H*|Beneficial/equal/harmful|No-move/x-only/y-only/both|x/y/joint target matches|
|---|---:|---:|---|---|---|
|Pooled120|0.947906080161726|1.054803218551833|37/63/20|62/11/13/34|85/89/73|
|Clean40|1.2072654157145626|1.6730731697808663|0/39/1|39/0/1/0|38/38/38|
|Dark40|0.9370790223162285|1.0356040364068448|21/15/4|15/4/7/14|33/33/29|
|Bright40|0.9701882237639385|1.0946176124638984|16/9/15|8/7/5/20|14/18/6|

- Full x/y class counts, exact MSEs and diagnostics:`research_log/T020C_run/summary.json`; per-row outputs:`evaluation.json` and `decisions.json`.
- Feature hash:`df191f8b8b9019931b9500cb40b2ada7b01e8b35ca0f15c6bb4d8e876fae63b8`.
- Frozen predictions:`731258100e6a71ec439935a06349c7c40851e35842bee0d72df3a45de67e01e8`; frozen before independent reference-free replay and before held-out reference evaluation. Original folds hash:`8fdb03cdac63af5d6e58b557c96679bc15c1e2e921e525916eb9d21ad22cd2c1`.
- Files changed:new `ttie/nonspatial_oof.py`, GPU launcher, focused tests, independent `research_log/T020C_verify.py`, fixed-input/origin/feature/fold/head/freeze/evaluation receipts, analysis and commands. Donor implementation unchanged.
- Tests:baseline3 passed15.30s; focused/affected5 passed8.40s, including both-axis held-out target poison/mutation prediction-hash isolation. Independent pre-reference replay exactly reconstructs84-D features, training-only normalization and all10 head logits/classes/boundaries; PASS. Independent post-reference120-row metrics/diagnostics/five-clause replay PASS. Accepted-B cache provenance120 rows/360 input-state-gate files exact.
- GPU1 A6000 feature run:`20260913-204217-ttie-t020c-features`, exit0;600 candidate feature vectors computed once without reference pixels/MSE or TTT rerun. Original CPU donor runtime retained for the tiny heads. Command history:`research_log/T020C_commands.md`.
- Failure/deviation:pre-launch Git transport index text contained CRLF; index metadata converted to LF and source preflight passed. No failed scientific run, no scientific-setting change, no post-result search. No T020-A fresh artifacts consumed.
- Recovery archive hash:`85163515f32597a3ef8586226ddc5b19e8e04d55bae79abb3bcae437ead2c944`,1984210 bytes, verified on both A6000 TTIE storage roots; compact evidence mirrored locally and remotely.
- Interpretation:the unchanged representation/learner does **not** meet the declared non-spatial in-domain OOF safety standard. This does not prove all possible predictors using these features must fail. It does not justify combined-domain final training. No fresh qualification or broad-deployment claim.
- Next:stopped after T020-C. Await research-lead review; no combined-domain training, new cohort, threshold/feature/architecture change, or T020-D. Research-owned state/inbox untouched.
