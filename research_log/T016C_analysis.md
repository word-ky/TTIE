# T016-C: grouped OOF feature-sufficiency probe

Result: **neither_probe_establishes_development_rankability**. Development-only supervised diagnostic; not fresh qualification or a deployable selector.

Forty fixed development IDs, sorted-ID modulo five folds; 32 training / 8 held-out IDs per fold. All conditions and candidates for an image stay together. Ten heads, CPU, exact fixed T014 value-head recipe. Train-only normalization; no model/epoch selection. All OOF predictions were frozen before reference evaluation.

## probe28

Fixed clauses: 0/5; {'spatial_improves_region2_3pct': False, 'spatial_within_hard_oracle_5pct': False, 'offset_improves_region2_5pct': False, 'left_right_no_more_than_1pct_worse': False, 'quadrants_no_more_than_1pct_worse': False}

| Group | Selected MSE | Region2 | Hard oracle | Frozen T016-B | /Region2 | /Hard oracle | /Frozen T016-B |
|---|---:|---:|---:|---:|---:|---:|---:|
| spatial_pool | 0.0358066896559 | 0.0350435737482 | 0.0325635645189 | 0.0378572608073 | 1.02177620106 | 1.09959367732 | 0.945834138349 |
| left_right | 0.0345864885952 | 0.0333970155101 | 0.0321709857788 | 0.0359704593895 | 1.03561614914 | 1.07508327016 | 0.96152479513 |
| quadrants | 0.0330392681877 | 0.0309838496498 | 0.0309821883566 | 0.0383517022361 | 1.06633838471 | 1.06639556275 | 0.861481140637 |
| offset_left_right_40 | 0.0397943121847 | 0.0407498560846 | 0.0345375194214 | 0.0392496207962 | 0.976550987127 | 1.15220527853 | 1.01387762168 |

| Group | Selected counts | Oracle counts | Disagreement | Outside oracle ties | Spearman summary |
|---|---|---|---:|---:|---|
| spatial_pool | [16, 9, 6, 16, 42, 11, 5, 14, 1] | [27, 2, 12, 23, 41, 14, 0, 1, 0] | 0.6083333333333333 | 0.6083333333333333 | {'count': 120, 'null_count': 9, 'mean': 0.41913425899437323, 'median': 0.5666666666666667, 'min': -0.9333333333333333, 'max': 1.0} |
| left_right | [7, 2, 0, 12, 10, 4, 1, 4, 0] | [2, 0, 0, 23, 2, 13, 0, 0, 0] | 0.725 | 0.725 | {'count': 40, 'null_count': 3, 'mean': 0.4968468468468468, 'median': 0.6, 'min': -0.4166666666666667, 'max': 1.0} |
| quadrants | [1, 4, 2, 0, 25, 2, 1, 4, 1] | [0, 0, 0, 0, 39, 0, 0, 1, 0] | 0.35 | 0.35 | {'count': 40, 'null_count': 0, 'mean': 0.5189309020427191, 'median': 0.6416666666666666, 'min': -0.38333333333333336, 'max': 0.9666666666666667} |
| offset_left_right_40 | [8, 3, 4, 4, 7, 5, 3, 6, 0] | [25, 2, 12, 0, 0, 1, 0, 0, 0] | 0.75 | 0.75 | {'count': 40, 'null_count': 6, 'mean': 0.21715686274509804, 'median': 0.275, 'min': -0.9333333333333333, 'max': 0.9666666666666667} |

| Fold | Held-out IDs | Selected MSE | LR / Quadrants / Offset MSE | Final train Huber |
|---|---|---:|---|---:|
| 0 | [55167, 56288, 57149, 57672, 58393, 59386, 60102, 60770] | 0.0432570583653 | {'left_right': 0.04247221723198891, 'quadrants': 0.04585543577559292, 'offset_left_right_40': 0.041443522088229656} | 0.0535377306795 |
| 1 | [55299, 56344, 57232, 58029, 58539, 59598, 60347, 60823] | 0.0247305707308 | {'left_right': 0.021285319118760526, 'quadrants': 0.023609573603607714, 'offset_left_right_40': 0.02929681946989149} | 0.0474748528666 |
| 2 | [55528, 56350, 57238, 58111, 58636, 59635, 60363, 60835] | 0.0421076838781 | {'left_right': 0.04103725589811802, 'quadrants': 0.03437544486951083, 'offset_left_right_40': 0.05091035086661577} | 0.066398843433 |
| 3 | [55950, 56545, 57244, 58350, 58705, 59920, 60449, 60855] | 0.0403848751448 | {'left_right': 0.039371089078485966, 'quadrants': 0.035608510952442884, 'offset_left_right_40': 0.0461750254034996} | 0.0547366887331 |
| 4 | [56127, 57027, 57597, 58384, 59044, 60052, 60507, 60886] | 0.0285532601605 | {'left_right': 0.02876656164880842, 'quadrants': 0.02574737573741004, 'offset_left_right_40': 0.03114584309514612} | 0.0528893273462 |

Tie/null details per condition: {'spatial_pool': {'prediction_tied_episodes': 14, 'oracle_tied_episodes': 12}, 'left_right': {'prediction_tied_episodes': 5, 'oracle_tied_episodes': 5}, 'quadrants': {'prediction_tied_episodes': 2, 'oracle_tied_episodes': 0}, 'offset_left_right_40': {'prediction_tied_episodes': 7, 'oracle_tied_episodes': 7}}.

## probe30

Fixed clauses: 0/5; {'spatial_improves_region2_3pct': False, 'spatial_within_hard_oracle_5pct': False, 'offset_improves_region2_5pct': False, 'left_right_no_more_than_1pct_worse': False, 'quadrants_no_more_than_1pct_worse': False}

| Group | Selected MSE | Region2 | Hard oracle | Frozen T016-B | /Region2 | /Hard oracle | /Frozen T016-B |
|---|---:|---:|---:|---:|---:|---:|---:|
| spatial_pool | 0.0354510319846 | 0.0350435737482 | 0.0325635645189 | 0.0378572608073 | 1.01162718846 | 1.08867172585 | 0.93643943668 |
| left_right | 0.0342112310929 | 0.0333970155101 | 0.0321709857788 | 0.0359704593895 | 1.02437989055 | 1.06341880004 | 0.951092415097 |
| quadrants | 0.0333758114488 | 0.0309838496498 | 0.0309821883566 | 0.0383517022361 | 1.07720027776 | 1.07725803822 | 0.870256325086 |
| offset_left_right_40 | 0.0387660534121 | 0.0407498560846 | 0.0345375194214 | 0.0392496207962 | 0.951317553899 | 1.1224330543 | 0.987679692841 |

| Group | Selected counts | Oracle counts | Disagreement | Outside oracle ties | Spearman summary |
|---|---|---|---:|---:|---|
| spatial_pool | [9, 5, 12, 16, 40, 25, 2, 7, 4] | [27, 2, 12, 23, 41, 14, 0, 1, 0] | 0.6333333333333333 | 0.5666666666666667 | {'count': 120, 'null_count': 7, 'mean': 0.5275345120253869, 'median': 0.5833333333333334, 'min': -0.7833333333333333, 'max': 1.0} |
| left_right | [4, 1, 4, 8, 10, 11, 1, 0, 1] | [2, 0, 0, 23, 2, 13, 0, 0, 0] | 0.725 | 0.65 | {'count': 40, 'null_count': 2, 'mean': 0.5548264357132587, 'median': 0.6166666666666667, 'min': -0.5, 'max': 1.0} |
| quadrants | [0, 1, 0, 2, 24, 4, 1, 6, 2] | [0, 0, 0, 0, 39, 0, 0, 1, 0] | 0.425 | 0.425 | {'count': 40, 'null_count': 0, 'mean': 0.5595833333333333, 'median': 0.625, 'min': -0.35, 'max': 0.95} |
| offset_left_right_40 | [5, 3, 8, 6, 6, 10, 0, 1, 1] | [25, 2, 12, 0, 0, 1, 0, 0, 0] | 0.75 | 0.625 | {'count': 40, 'null_count': 5, 'mean': 0.46127605624090157, 'median': 0.5, 'min': -0.7833333333333333, 'max': 1.0} |

| Fold | Held-out IDs | Selected MSE | LR / Quadrants / Offset MSE | Final train Huber |
|---|---|---:|---|---:|
| 0 | [55167, 56288, 57149, 57672, 58393, 59386, 60102, 60770] | 0.0431003960936 | {'left_right': 0.04263556282967329, 'quadrants': 0.04501480027101934, 'offset_left_right_40': 0.041650825180113316} | 0.0435444000694 |
| 1 | [55299, 56344, 57232, 58029, 58539, 59598, 60347, 60823] | 0.0239705079778 | {'left_right': 0.019814823172055185, 'quadrants': 0.02314280450809747, 'offset_left_right_40': 0.02895389625336975} | 0.0399213057977 |
| 2 | [55528, 56350, 57238, 58111, 58636, 59635, 60363, 60835] | 0.0423215969543 | {'left_right': 0.041563516831956804, 'quadrants': 0.037016590824350715, 'offset_left_right_40': 0.04838468320667744} | 0.0611170662774 |
| 3 | [55950, 56545, 57244, 58350, 58705, 59920, 60449, 60855] | 0.0406262602191 | {'left_right': 0.04043620638549328, 'quadrants': 0.035608510952442884, 'offset_left_right_40': 0.04583406331948936} | 0.0502439137134 |
| 4 | [56127, 57027, 57597, 58384, 59044, 60052, 60507, 60886] | 0.0272363986781 | {'left_right': 0.02660604624543339, 'quadrants': 0.02609635068802163, 'offset_left_right_40': 0.029006799100898206} | 0.0461615504766 |

Tie/null details per condition: {'spatial_pool': {'prediction_tied_episodes': 0, 'oracle_tied_episodes': 12}, 'left_right': {'prediction_tied_episodes': 0, 'oracle_tied_episodes': 5}, 'quadrants': {'prediction_tied_episodes': 0, 'oracle_tied_episodes': 0}, 'offset_left_right_40': {'prediction_tied_episodes': 0, 'oracle_tied_episodes': 7}}.

## Two-probe comparison

{'probe30_over_probe28': 0.990067284222979, 'median_spearman_difference': 0.01666666666666672}

Spearman uses average ranks for ties; constant ranks are null. Exact predicted-value and oracle ties use the first lexicographic boundary. Head inputs exclude IDs/condition/candidate IDs/reference MSE; probe30 appends only the prescribed gx/gy coordinates.

No threshold, fold, epoch, architecture or loss was changed. No new images, CLIP/A6000 scoring, rendering, TTT, deployable boundary model, or follow-on experiment. Stop for research-lead review.

## Execution and exact provenance

Scientific code `058e000429c89b3bef617bee2850a0f534205505`, branch `codex/T016C-feature-sufficiency`, PR18 https://github.com/word-ky/TTIE/pull/18. Started from main d97492a5204a57be2b57bc09ba882a1196bf4218 (task inbox bdcc64497c640facd860d84b3b8095b424a6a728). No PR17 topology work or change to accepted T014/T016-B assets. Only new grouped-fold, feature-appending, OOF driver and metric code; donor training function unchanged.

One completed CPU run: `D:/anaconda3/python.exe -m ttie.boundary_probe_run --source-sha 058e000429c89b3bef617bee2850a0f534205505 --output research_log/T016C_run`, exit0 at2026-09-12T18:30:07.3415472Z. Runtime Python3.12.7, PyTorch2.13.0+cpu. No A6000 experiment, CLIP call, candidate rendering, TTT or new image. All61 scientific file bytes were checked against actual Git blobs before input/model work. No scientific code change after the run.

Input evidence is read directly from immutable Git commit `4062e01cb93de731c394015c5ac741d6c08e04d8`. All five compact artifact hashes are verified/reported; selection/config hashes agree with the accepted T016-B receipt. No reference MSE was recomputed.

| Input artifact | SHA256 |
|---|---|
| research_log/remote_runs/20260913-013748-ttie-t016b-assets-ready/artifacts/scoring/selection.json | `99dbb10260e5045d5c5deb1d53b5f68d42567a02d5a1b898b234e6b5d113b866` |
| research_log/remote_runs/20260913-013748-ttie-t016b-assets-ready/artifacts/evaluation/evaluation.json | `8a2204fc481da3559b337ec634ef9c87cf890d63bd7edf2cf5ae947aaea0e22e` |
| research_log/remote_runs/20260913-013748-ttie-t016b-assets-ready/artifacts/scoring/selection_receipt.json | `1fd17da1fd52db1ffe86749542908cf81b5b441c2a6b6ca0a85819e440c5c3ac` |
| research_log/remote_runs/20260913-013748-ttie-t016b-assets-ready/artifacts/scoring/config.json | `e4c8d8ddfe16333fa4c766f6e1ecca2388e89a7f2ce9ba36c3537017cd666e64` |
| research_log/remote_runs/20260913-013748-ttie-t016b-assets-ready/artifacts/evaluation/evaluation_receipt.json | `d7b6865c0d6771e202c223cab18e721582297f2f0473597eea2106304782e900` |

## Training and held-out separation

Exactly five deterministic folds use sorted unique image IDs modulo5. Every fold has32train/8held-outIDs,96train/24held-out episodes,864train/216held-out candidate rows. All three conditions and nine candidates from an image remain in the same fold. Original episode order and lexicographic candidate order are preserved within training rows; seed7 determines each epoch shuffle. IDs/conditions are used only for grouping and reporting. The model receives28 saved features, or those same features plus gx/gy only.

Both probes call the unchanged T014 value-head trainer: QualityHead(D,SiLU), D-64-64-1, train-only x/y standardization, log(MSE+1e-6), Huber delta1, AdamW lr1e-3/weight_decay1e-4, batch256, seed7,100epochs, final epoch only. No Sobolev term, ranking loss or model search. Ten formal heads are saved with state dictionaries, recipes, complete100-epoch histories, normalization statistics, training/held-out ID lists, and hashes. Reported final train Huber is the reused trainer's last-epoch batch-weighted loss during that epoch, not a model-selection criterion.

The compact reference artifact is parsed for fold metadata and the permitted supervised training targets. Within each fold, only that fold's training-row MSEs are extracted for fitting; held-out prediction accepts feature tensors only, without targets or metadata. Held-out references are never used in its training/normalization/selection. A focused test rejects any held-out target access during train_fold and confirms that replacing those targets leaves training statistics, loss and predictions unchanged.

All ten fold prediction/selection files and both consolidated OOF tables were saved and hashed before reference evaluation. This is supervised development cross-validation, not label-free training and not a new fresh split. Each image contributes predictions from exactly one model that did not train on that image.

OOF finalized: `2026-09-12T18:30:06.215214+00:00`; evaluation started: `2026-09-12T18:30:06.218217+00:00`. OOF freeze-fileSHA `be6e3f5192107e8215aa09a5c556327faf0e99afd8bbcf296445766120638bdc`.

- probe28 OOF SHA256: `d756efe633d4613a35936b556f374db143e3f0061a3e2164f39ea218761388fd`.
- probe30 OOF SHA256: `c965d160ea681f8fae991cde43fa831a2a8d9b982492248bb92f29328989a420`.

## Verification and observed failures

Baseline4energy-coretestsPASS9.408s;4kerneltestsPASS8.930s;6finalfocusedtestsPASS9.833s;py_compilePASS. Tests cover group isolation, train-only standardization, exact28/30dimensions, prescribed geometry, bitwise equivalence of28Dtraining to accepted train_energy, exact ties, forbidden held-out reference arguments/access, target replacement and literal acceptance/interpretation. No implementation test failed. Full repository suite was not rerun because the task requests focused tests and no donor scientific code changed.

Independent post-run audit verified61sourceGitblobs,5sourceartifactblobs,allfivefoldassignments,10train-onlyx/ynormalizations,10head/history/receipt hashes,and2160saved-headOOF predictions with maxabsolute difference0. Predictions/selections remained unchanged. The auditor loads saved heads and performs forward verification only; it does not train, render or rerun CLIP. All MSE joins, group ratios/counts and five-clause decisions were checked. Separate SciPy rank audit verified111defined/9null correlations forprobe28 and113defined/7null forprobe30. Constant-rank null cases stay explicit.

The initial combined PyTorch+SciPy post-run audit process exited3 because the Windows environment loaded duplicate OpenMP runtimes. The original failure log is preserved. The minimal repair separated PyTorch source/head verification and SciPy correlation verification into distinct processes. No package or environment workaround was installed, no duplicate-runtime override was enabled, and the completed experiment was not rerun. Both audits then passed; no scientific source, score, checkpoint, selection or threshold changed.

## Literal conclusion and limitations

Bothprobe28andprobe30failallfiveclauses. Probe28 MSE.03580668965587393 is2.18% above canonicalRegion2; probe30 MSE.035451031984606134 is1.16% above. Both improve over the failed frozenT016-B selector (5.42% and6.36% respectively), but that weaker comparison is not an alternate pass route. Probe30/probe28=.990067284222979 (0.99% lower MSE), with medianSpearman difference+.01666666666666672.

Explicit coordinates improve offset ranking in this diagnostic, yet offset MSE ratio.9513175538986504 still misses the fixed.95threshold, while quadrants ratio1.0772002777588976 exceeds1.01. Higher mean correlation or lower trainingHuber does not establish reliable selection. Per-fold and per-condition results above remain visible, including folds whereprobe30isworsethanprobe28.

The prescribed conclusion is only that this fixed simple feature family/probe does not establish development rankability. The result does not distinguish a universally insufficient representation from a limited training objective/model/amount of development data, and it does not authorize a larger head or extra features. Do not conclude that explicit geometry is proved necessary or sufficient. These40IDs were already inspected in prior studies; OOF isolation does not turn them into fresh generalization evidence. Stop after this report and await the next research-lead decision.
