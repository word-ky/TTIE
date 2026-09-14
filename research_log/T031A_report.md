# T031-A — DONE

The fixed source-support diagnostic passes the predeclared strong gate: AUROC for invalid gradients **0.7204698309323111**, Spearman distance versus reference-gradient cosine **-0.48229925365892745**, and invalid-state median distance **1.1048132565555435** exceeds valid-state median **0.8210241936964588**. This is an association on development-used trajectories; it does not establish causality or qualify a deployable stopping rule.

## Fixed inputs and information boundary

Source commit: `65c56a2b6d18a6b0874d823175482bdd9d05a4bc`, branch `codex/T031A-source-support`, PR #56. The 400 accepted T014 `training_bank/*/bank.pt` files contribute exactly 7,346 feature rows from the frozen 80 source-training IDs. Every bank hash is recorded in `T031A_result/support/freeze.json` and bound to the accepted T014 receipt. No calibration features, real recalibration features, source derivatives, source targets, or source images enter the bank.

T014 training manifest SHA256: `92fd60d01ca0780880d724464c4d0a76ba1721ab5b8a2d054d4035a141673125`. Frozen energy SHA256: `c3d1eef9f20af163e1268823cf16d3fdaed315e7723fc94db3b1138ad1336521`. Reconstructed source mean/population standard deviation exactly reproduce the frozen float32 normalization buffers. Saved features and buffers are promoted to float64 solely for the predeclared distance calculation: nearest Euclidean distance in standardized 28-D space divided by sqrt(28).

The accepted T030 cohort SHA256 is `ec67f0a6af5682c8e1e929db56e1d771dfd3183f75cb4b365052cde024f55f2d`. All 400 saved trajectory/decision/output artifacts are bound in the freeze receipt; exactly 100 × 41 existing states/features are used. No new TTT trajectory or cohort was produced. The score API accepts only source features, state features and frozen normalization buffers.

All 4,100 scores and their independent SciPy replay completed at **2026-09-14T11:42:14.049761Z**, with zero image/normal decodes. Freeze SHA256: `3b84948baed0c84e257365a2c9ed8be18148b853327778dc78dc96e9a4037756`; scores SHA256: `1cebced374400a359142cc088aa5f6a5e7b886f86a20cbd43f74c37e828bda2d`; bound tensors SHA256: `42a192f86b37d19f551c180dec9e992b73507126a27d9c7925936e310ff74964`.

The separate `REFERENCE_GRADIENT_DIAGNOSTIC_ONLY` stage first decoded a normal at **2026-09-14T11:43:42.815090Z**. All 100 low and 100 normal opens have UTC receipts. It recomputes gE and isolated RGB-MSE gR at the same raw states, using active EV/gamma coordinates, positive dot validity and the T029 cosine convention. All features match saved features exactly; all values are finite; models, checkpoint assets and raw states during differentiation remain unchanged. Optimizer updates and selection decisions are both zero. No T029 reference artifacts enter the support calculation.

## Results

| Group | Count | Distance median | Q25 | Q75 | IQR |
|---|---:|---:|---:|---:|---:|
| Valid | 2406 | 0.8210241936964588 | 0.6297751546820336 | 1.05217420302468 | 0.4223990483426463 |
| Invalid | 1694 | 1.1048132565555435 | 0.8915356378636976 | 1.2783672477251944 | 0.3868316098614968 |

| Subset | Count | Median distance | Invalid fraction | Median cosine |
|---|---:|---:|---:|---:|
| Step 0 | 100 | 0.556907896199011 | 0.22 | 0.6380396697370678 |
| Step 10 | 100 | 0.7095573529348761 | 0.02 | 0.6967643396498862 |
| Step 20 | 100 | 0.9597094422039243 | 0.34 | 0.22100401517332668 |
| Step 30 | 100 | 1.193899971111255 | 0.76 | -0.25093367028099395 |
| Step 40 | 100 | 1.3021635596953227 | 0.76 | -0.23403530275005796 |
| Original T026-A selected | 100 | 1.3021635596953227 | 0.77 | -0.2730950144596993 |

Degenerate pairs: 0/4100. The fixed gate AUROC >=0.70, rho <=-0.30, and larger invalid median passes. Repeated states within each image and the common trajectory-time trend limit any causal interpretation. No threshold, early-stopping simulation, new selector, retraining, baseline quality run or official-test access was performed.

## Execution and verification

A6000 physical GPU 1, PyTorch 2.4.0+cu121 / CUDA 12.1 / Python 3.12.12. Release `20260914-194158-ttie-t031a-support`; support run `20260914-194203-ttie-t031a-support` and reference run `20260914-194331-ttie-t031a-reference` both exit 0. Actual commands and logs are retained in `T031A_result/`. The GPU distance kernel took 0.265912935 seconds; support computation/replay took 1.951723357 seconds excluding process imports/tests; the reference diagnostic took 302.939233591 seconds.

Focused tests: `python -m pytest -q tests/test_t031a_support.py`: local 2 passed in 23.52s; server 2 passed in 1.98s. Independent all-row SciPy distances have maximum absolute error **4.440892098500626e-16**, below 1e-9. Independent pairwise AUROC and rank-Pearson correlation match exactly. Local saved-tensor replay verifies all 4,100 distances and gradient scalar statistics to maximum error **4.440892098500626e-16**, all validity labels, exact AUROC/rho and all requested subgroup summaries. Source/artifact hashes and the pre-reference freeze ordering pass.

Local replay commands, using separate processes:

```text
python research_log/T031A_support/export_saved.py --result research_log/T031A_result --out research_log/T031A_download/verification_arrays.npz
python research_log/T031A_support/replay_saved.py --result research_log/T031A_result --arrays research_log/T031A_download/verification_arrays.npz
```

Observed failure: the initial local read-only verifier encountered duplicate OpenMP runtimes when Torch and SciPy computations shared one process. Separating Torch tensor export from the NumPy/SciPy checker resolved it without unsafe environment overrides, scientific changes or GPU reruns. The existing NVML warning did not prevent CUDA execution. No scientific deviations or unresolved blockers.

Full execution backup: `/media/wenchang/F/wjq/TTIE/shared/t031a/T031A_execution.tar`, 4,526,080 bytes, SHA256 `0caa752d7844d6633250a1df3e0a8429c9e1cf4e34eddfeb598014b6da3bfaf3`. Compact evidence: 1,091,553 bytes, SHA256 `5522e97b5c7c41c2592c0880abb793ffd87a055579911f63376ebfaeb76a3e9d`, retained on both server roots and extracted here. Exact per-file hashes are in `T031A_backup.json` and the frozen receipts.

Stop after this diagnostic and await research-lead review. The next scientific decision belongs to ChatGPT; no deployable rule is introduced by this result.

promising source-support proxy
