# T065-C — DONE / TRANSFER_NEGATIVE

Source commit: `1c3445e6cdfee941a26180628ea146e87cd78033`. Branch: `codex/T065C-knn-safety`. Authorization: `d8332a7779f8e961934d6e573b35282ba71e6da9`.

The exact fixed5-NN safety guard fails transfer:0/100 development rollbacks and0/100 transfer rollbacks. Both catastrophic transfer choices have five safe-labeled development neighbors, hence p_safe=1.0; their unsafe choices remain unchanged. Leave-one-image-out development evaluation misses80/81 unsafe states. This is an exposed-cohort audit, not fresh qualification; close this exact fixed-feature5-NN rule without another k/metric/normalization/threshold/model attempt.

Transfer absolute PSNR/SSIM: 15.471845204573 / 0.397896864536.

| Gate | Required | Development LOIO guard | Transfer | Transfer pass |
|---|---:|---:|---:|---|
| mean_delta_psnr | >=2.00 dB | 3.5695322981528443 | 3.8619303358477532 | True |
| median_delta_psnr | >0 dB | 3.093839109589621 | 3.814406659510415 | True |
| regressions_t026 | <=29/100 | 3 | 12 | True |
| worst_delta_t026 | >=-5.614 dB | -4.074446413712419 | -10.364944679494553 | False |
| mean_delta_ssim | >=-0.001 | 0.015011775217397242 | 0.01842184098158023 | True |

Development bank:2719safe/81unsafe states, total2800. Query image is excluded in full from development neighbor search (all28states); transfer uses all2800development states. No re-normalization or transfer labels.

Leave-one-image-out development state confusion matrix:

| True class | Predicted safe | Predicted unsafe |
|---|---:|---:|
| Safe | 2694 | 25 |
| Unsafe | 80 | 1 |

Unsafe->safe false negatives80/81. These LOIO state-level counts are distinct from T065-B training-set confusion and from per-image guard gates. `development_fit.json` separately records all100base-step probabilities and keep/rollback behavior. All100base choices are retained.

Development histogram (nonzero): {"21": 1, "22": 1, "23": 2, "24": 1, "25": 9, "26": 21, "27": 65}.
Transfer histogram (nonzero): {"18": 1, "21": 3, "24": 3, "25": 6, "26": 22, "27": 65}.

Post-freeze tail outcomes:

| Index | Step | Selected PSNR / SSIM | Margin vs T026 | p_safe |
|---|---:|---|---:|---:|
| 16 | 21 | 15.243922691784 / 0.730831917479 | -7.131115430332 | 1.0 |
| 86 | 25 | 13.247833801770 / 0.607845717517 | -10.364944679495 | 1.0 |

Five ordered development neighbors for each tail selected state (labels shown only as post-freeze diagnosis):

| Query index | Rank | Development image index | Step | Safe label | Euclidean distance |
|---|---:|---:|---:|---:|---:|
| 16 | 1 | 57 | 24 | 1 | 0.7862412114852282 |
| 16 | 2 | 57 | 23 | 1 | 0.8309705543013888 |
| 16 | 3 | 5 | 24 | 1 | 0.8649867433284708 |
| 16 | 4 | 5 | 25 | 1 | 0.8674883895464427 |
| 16 | 5 | 5 | 23 | 1 | 0.9537663128840276 |
| 86 | 1 | 73 | 24 | 1 | 1.092577457873168 |
| 86 | 2 | 73 | 25 | 1 | 1.106349785965698 |
| 86 | 3 | 5 | 26 | 1 | 1.200883281731119 |
| 86 | 4 | 5 | 25 | 1 | 1.231235293625571 |
| 86 | 5 | 5 | 27 | 1 | 1.235514767513963 |

Exact feature source `research_log/T065A/core.py`, SHA256 `fec15f6f2cb430377c1c52e2316a1036a1c6878a16ec37bdf113293017a0487f`; original normalization SHA256 `5a644c74b7b98a242e7c2945f2fa7b971a5d0e046ba3e66f4e5e7462ba7599ef`. Frozen bank SHA256 `59fb100d3adfbb5ac83dd9894b7becae7918b5d74ff5e2c832edb1304669b848`. The bank contains normalized features, development binary labels and deterministic image-index/step metadata. Query image IDs are not features; development IDs serve only exclusion and prescribed tie ordering.

Distance/selection: ordinary float64 Euclidean sqrt(sum squared differences) on GPU, sorted by(distance, development_image_index,step). Exactly5state neighbors, unweighted mean of their binary labels, threshold0.5. No distinct-image-neighbor constraint, weighting, temperature or metric learning was added. Keep base when predicted safe; otherwise latest earlier safe state, else0. Exact rho=0.9857470621423519, action/objective/states unchanged; zero Adam reruns.

Bank/rule/development LOIO predictions frozen 2026-09-20T12:01:27.057311+00:00; all100transfer choices/output hashes frozen 2026-09-20T12:01:51.472276+00:00; first transfer reference-quality read 2026-09-20T12:01:51.662384+00:00. Selector SHA256 `52df39c210babe2940d18f6b5478bf4b972364a94cafbcf03fa2531d74bb7ddb`; transfer freeze SHA256 `1998e9c379256e4015c9fcfd227a23ff65ed3fb9e32ea0db08881197642a3ffd`. Each feature/selection read set is100low PNGs+100frozen traces; the frozen development bank is the only labeled inference resource, as authorized. Transfer neighbor records contain IDs/steps/distances/probabilities, no transfer outcomes. Tail labels/metrics are emitted post-freeze. No official test/cross-dataset/new cohort access.

Independent verifier PASS: re-render2,800states per cohort, recompute features and bank normalization/labels; independently compute Euclidean distances with NumPy/einsum and order via Python tuple sorting. Every selected neighbor ID/step, probability, exclusion, rollback, output hash, read ordering, metric and classification matches. Synthetic tie/exclusion tests also pass. Maximum neighbor distance errors development/transfer 3.9968028886505635e-15 / 5.329070518200751e-15; feature errors 7.105427357601002e-15 / 6.217248937900877e-15; metric errors 1.0871303857129533e-12 / 8.846257060213247e-13. Accepted Git-bound development T026/T036 anchors are reused; state metrics and transfer controls are independently scored.

Validation: 16baseline passed10.70s;3neighbor tests passed8.06s;4including independent passed8.59s;15affected passed9.03s; remote15passed1.71s. Run `20260920-200051-ttie-t065c-knn`, exit0, primary pipeline 52.592265s. A6000GPU1 for renderer/features/primary distances; CPU deterministic sorting and independent numerical audit. One frozen bank, no rule retries. Source pushed before science.

Commands: `python -B -m pytest research_log/T065C/test_core.py research_log/T065A/test_core.py research_log/T063C/test_core.py research_log/T063A/test_core.py -q`; `python -B -m research_log.T065C.run --out /media/wenchang/F/wjq/TTIE/runs/T065C-knn-safety`; `python -B -m research_log.T065C.verify --out /media/wenchang/F/wjq/TTIE/runs/T065C-knn-safety`. Single-thread BLAS/sequential MKL, TF32off, CUDA_VISIBLE_DEVICES=1 as recorded.

Failures/deviations: one SSH status-query timeout recovered on the next query. A local archive read initially preceded fetch completion and was retried after the transfer finished. No scientific execution failure, result/source change, rerun or design deviation.

Evidence directory includes complete training/bank tables, fullLOIOneighbors/confusion/base behavior, rule manifest, transfer neighbors/probabilities/freeze, post-freeze tails/metrics, verifier/config/logs. Raw `/media/wenchang/F/wjq/TTIE/shared/t065c/T065C_raw.tar`, SHA256 `9486052a8f4500e38a026a87f66cb9b1a7e24c9c1e636ca4ee3f819ad2e5bf43`, 298567680 bytes. Recovery `/home/wenchang/asdasdsad/wjq/TTIE/shared/t065c/T065C_recovery.tar.gz`, also F: and local, SHA256 `076953ec9c6d362c8f9fb21e8a15210c87e97a06240bb3a7b59e5f23725d298d`, 2959520 bytes. Home/F/local recovery hashes match.

Recommendation: accept TRANSFER_NEGATIVE and close this exact fixed-feature5-NN rule. Its local neighborhoods provide no warning for these two severe transfer failures, and development LOIO unsafe-state recall is poor. Await a new bounded task; do not infer that every possible nonlinear model over these features must fail. No historical artifacts were merged into main.
