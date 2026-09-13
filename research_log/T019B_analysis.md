# T019-B grouped-OOF deadband-direction positive

All seven predeclared requirements pass on the same120 development episodes. This is a controlled label-only intervention: fewer harmful moves at a small cost in pooled MSE relative to T018-C. It is development cross-validation, not a fresh qualification or a final all-development selector.

## Fixed result

| Group | H0 | H1 | H* | H1/H0 | H1/H* | Beneficial/equal/harmful |
|---|---:|---:|---:|---:|---:|---:|
| pooled120 | .03504357374816512 | .03327835857635364 | .032563564518932255 | .949627992153514 | 1.021950731376653 | 46/69/5 |
| left/right40 | .03339701551012695 | .03280562967993319 | .032170985778793695 | .982292255126377 | 1.019727213381128 | 17/18/5 |
| quadrants40 | .030983849649783225 | .030983849649783225 | .030982188356574625 | 1 | 1.000053620912425 | 0/40/0 |
| offset40 | .04074985608458519 | .0360455963993445 | .03453751942142844 | .8845576368300302 | 1.0436648897540797 | 29/11/0 |

Acceptance vector, in inbox order: **[true,true,true,true,true,true,true]**. Pooled H1/H0<=.97, pooled H1/H*<=1.03, offset H1/H0<=.95, left/right and quadrants H1/H0<=1.01, pooled harmful<=10, quadrants harmful==0. The offset oracle ratio has no separately prescribed limit; it is reported without substituting it for the pooled oracle clause.

## Comparison and diagnostics

| Group | T018-C H1/H0 → T019-B | Harmful C → B | Moving episodes C → B |
|---|---|---|---|
| pooled | .9463400050770442 → .949627992153514 | 11 → 5 | 68 → 51 |
| left/right | .9837094741065749 → .982292255126377 | 11 → 5 | 35 → 22 |
| quadrants | 1 → 1 | 0 → 0 | 0 → 0 |
| offset | .874913447259421 → .8845576368300302 | 0 → 0 | 33 → 29 |

Pooled MSE is slightly worse than T018-C; the main benefit is the specified safety transfer, not a new best pooled value. Both-axis movements fall27→13; no-move52→69. The intervention preserves all five prior development clauses while reducing harmful episodes11→5. All five remaining harmful cases are left/right rows15,54,99,105,108; exact MSE and choices are in summary.json. Quadrants movement/harmful-case table is empty:40 center decisions,0 moves,0 harmful.

Deadband-target agreement is x109/120(90.8333%), y102/120(85%), joint94/120(78.3333%). Predicted x counts for center/low/high are93/27/0; y83/27/10. Joint counts (bx,by): (.5,.5)=69, (.5,.4)=18, (.5,.6)=6, (.4,.5)=14, (.4,.4)=9, (.4,.6)=4; every bx=.6 pair is0. Non-center predictions on center-target axes: x3,y6. Of the26 axes suppressed by T019-A,20 are predicted center and6 still move. Per-family distributions, complete suppressed-axis table, donor oracle ratios and all movement counts are in diagnostics.json.

## Reproduction and information boundary

Scientific source **c66a8a0f5f16d64bf160c74e6217a652d0950dc4**. The original eight donor source files, feature origins, folds and RECIPE are SHA/structure-exact with accepted T018-C config at b45bf9563e7c3f7e8aa27c923a35f1aabb95a5cb. The only scientific change is bx/by targets from accepted PR32 head **91f750e871d2c133624bbe4972f3fb3086f25a6c**, `research_log/T019A_run/decisions.json` SHA **d874bed74b0ebf68b680c8b6e60c8c5a44c9d82ce3cb7fd5730e16e53a980d45**. PR32 remains unmerged; no rerun or conflict repair was used.

The same5 image-grouped folds use96 training/24 held-out rows and32 training/8 held-out image IDs each. Exactly10 separate x/y heads use84→64→64→3, SiLU, unweighted CE, AdamW lr1e-3/weight decay1e-4, batch256, seed7,100epochs, finalepoch only, train-row mean/population-std normalization. Original Windows Python3.12.7/Torch2.13 CPU backend is retained for exact donor reproducibility; no GPU run was required for these small heads. No change to weights, folds, labels, epochs or features after seeing metrics.

Training hashes the original target bytes and scans top-level row boundaries, decoding only current training rows' bx/by scalars. It does not deserialize nested cross-MSE/gains or held-out labels. Each fold owns a96-row target subset. The unchanged donor inference function accepts only trained head and held-out84-D features. Fold membership, training labels and normalization files are hash-bound to the per-fold frozen receipts and global freeze. Feature loading follows the accepted donor artifact path; episode strings are used for post-freeze identity matching, never passed to the head.

All120 OOF logits/classes/choices frozen at **2026-09-13T07:11:50.382353Z**, SHA **f8b39e42d8956361fd74aabb9ad9312f30d2b72db295752375bc4e72a0ae70e9**. Independent reference/target-free reload reproduces all120 exactly at **07:12:08.558865Z**. Held-out references and labels open only at **07:12:30.557488Z**. Separate independent arithmetic/diagnostic verification passes at **07:12:47.671115Z**. Both verifiers import no TTIE implementation; replay opens only allowlisted scoring/fold/donor-config origins and saved heads/normalization, without opening target artifacts.

Commands (from worktree root, D:/anaconda3/python.exe):

```text
python -m pytest tests/test_direction_probe.py tests/test_deadband_probe.py -q
python -m ttie.deadband_probe train --source-sha c66a8a0f5f16d64bf160c74e6217a652d0950dc4 --output research_log/T019B_run
python research_log/T019B_verify.py replay
python -m ttie.deadband_probe evaluate --output research_log/T019B_run
python research_log/T019B_verify.py metrics
```

Baseline4 tests passed; final focused6 tests passed in19.71s. Train/replay/evaluate/metric processes all exited0. No scientific failures or deviations. A patch-format error during implementation was corrected before source commit; it did not run an experiment. T018-E fresh references, features, logits, family labels and outcomes were **not used for method development**. There was no fresh cohort, rerendering, CLIP/TTT, final fit, sweep, calibration or additional seed. Stop here for research-lead review; a later final fit/fresh qualification requires a new task.

## Compact origin preservation

`T019B_origins.pack` preserves59 original Git objects (278397bytes, SHA2566dd8d40c34b5b11c782690f8f89d286a6bdd8b36fd11d15ce9d84f3a0a3b317b), including the accepted unmerged T019-A labels and exact source/feature/fold/evaluation commit:path objects. All26 origin entries independently resolve and match SHA in an isolated bare repository; see T019B_origin_verification.json. A shallow reviewer checkout can import these objects with Python `subprocess.run(['git','index-pack','--stdin'], input=Path('research_log/T019B_origins.pack').read_bytes(), check=True)` before running the verifier. Preserve original run receipts: replay writes head_replay.json, so rerun on a copy when retaining the original chronological evidence.
