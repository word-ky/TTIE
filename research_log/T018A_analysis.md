# T018-A — reference hard local-direction target viable (5/5)

**Literal outcome: hard_local_direction_target_viable.** All five predeclared clauses pass. On the frozen development table, independent local hard-axis choices reduce pooled MSE by 6.9536% versus canonical hard Region2 and recover 98.2572% of available nine-hard oracle headroom. There are 69 beneficial, 51 unchanged and zero harmful combined moves. This is reference-only target viability, not a learned or deployable selector, nor fresh qualification.

## Literal clauses

| Clause | Observed | Limit | Pass |
|---|---:|---:|---|
| Pooled H1/H0 | 0.930464031121 | <= 0.97 | True |
| Pooled H1/H* | 1.00132726181 | <= 1.03 | True |
| Offset H1/H0 | 0.850731361516 | <= 0.95 | True |
| Left/right H1/H0 | 0.963289242688 | <= 1.01 | True |
| Quadrants H1/H0 | 0.999946381963 | <= 1.01 | True |

Vector `[true,true,true,true,true]`. No tolerance, fallback or post-hoc threshold.

## MSE, ratios and oracle headroom

| Group | H0 | H1 | H* | H1/H0 | H1/H* | Headroom recovered |
|---|---:|---:|---:|---:|---:|---:|
| spatial_pool | 0.0350435737482 | 0.0326067848946 | 0.0325635645189 | 0.930464031121 | 1.00132726181 | 98.257249% |
| left_right | 0.0333970155101 | 0.0321709857788 | 0.0321709857788 | 0.963289242688 | 1 | 100.000000% |
| quadrants | 0.0309838496498 | 0.0309821883566 | 0.0309821883566 | 0.999946381963 | 1 | 100.000000% |
| offset_left_right_40 | 0.0407498560846 | 0.0346671805484 | 0.0345375194214 | 0.850731361516 | 1.00375421076 | 97.912844% |

Recovery uses (mean(H0)-mean(H1))/(mean(H0)-mean(H*)). Per-episode fractions use each episode's denominator and preserve null when H0=H*. There are 51 undefined episode fractions: 7 LR, 39 quadrants, 5 offset. The quadrant aggregate denominator is positive but tiny (1.66129320860e-06); its 100% recovery reflects one small beneficial move, not a broad gain.

## Movement and oracle agreement

| Group | No move / x only / y only / both | Beneficial / equal / harmful | Oracle tie-set match | Tied oracle episodes |
|---|---|---|---|---:|
| spatial_pool | 51 / 6 / 33 / 30 | 69 / 51 / 0 | 115/120 (95.8333%) | 12 |
| left_right | 7 / 0 / 33 / 0 | 33 / 7 / 0 | 40/40 (100.0000%) | 5 |
| quadrants | 39 / 1 / 0 / 0 | 1 / 39 / 0 | 40/40 (100.0000%) | 0 |
| offset_left_right_40 | 5 / 5 / 0 / 30 | 35 / 5 / 0 | 35/40 (87.5000%) | 7 |

Oracle ties use exact equality to any minimum among the nine hard candidates, in the original lexicographic grid order. The local rule independently chooses each axis using center 0.5, lower 0.4, upper 0.6 tie priority. This priority is fixed before data inspection and gives identity on exact cross ties.

## Factorization regret H1-H*

| Group | Mean | Median | p95 | Max |
|---|---:|---:|---:|---:|
| spatial_pool | 4.32203756645e-05 | 0 | 0 | 0.00289808958769 |
| left_right | 0 | 0 | 0 | 0 |
| quadrants | 0 | 0 | 0 | 0 |
| offset_left_right_40 | 0.000129661126994 | 0 | 0.00041628764011 | 0.00289808958769 |

Only five offset rows have nonzero regret. The inherited linear quantile makes pooled p95 exactly zero because 115/120 rows have zero regret. All 120 selected axis moves are individually non-worse than center. There are no harmful combined moves, hence zero observed pure interaction failures and empty harmful-example lists in every family. The synthetic focused test does exercise a harmful joint move with individually improving axes; the formal table supplies no such example. No image IDs are attached, and any example record is keyed only by row index within its reported family.

## Per-episode oracle recovery distribution

| Group | Null count | Mean defined | Median | p5 | p95 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| spatial_pool | 51 | 0.990758186523 | 1 | 0.970471918878 | 1 | 0.659836877364 | 1 |
| left_right | 7 | 1 | 1 | 1 | 1 | 1 | 1 |
| quadrants | 39 | 1 | 1 | 1 | 1 | 1 | 1 |
| offset_left_right_40 | 5 | 0.981780424859 | 1 | 0.898870809515 | 1 | 0.659836877364 | 1 |

## Provenance and validation

Source freeze `c9f3a0ac763f2f0e5b3a8a67d009f2792f8fe118`. The only input artifacts are the original accepted T016-A candidate table and config from `ee5d8fdaf3ab48ee7ad3654d45bdc65419be8367`, with SHA256 `6091a0c928f115940997a647693b6571d8608131e7f9567a75882f0233b9c41e` and `e32b8b48ec0a93749698f51ab33637c5945bc8d827ec133175134bff93b168c5`. Five scientific source files are byte-bound to the declared commit.

Precheck verifies the exact 120x27 candidate grid, unique five hard-cross/nine hard positions and all 120 canonical MSEs. The selector reads only indices [12,3,21,9,15], without family or image ID. It reuses the accepted scalar axis argmin; inherited gx/gy fields describe hard finite differences only and never affect choices. Choices are persisted first, then H0/H1/H* and diagnostic quantities are computed and hash-frozen, then family labels are attached for reporting. No soft values affect either choice or quantities.

Four focused tests passed in 0.052 s; compilation passed. One formal CPU run exited 0. Independent `T018A_verify.py` passed all 120 choices and quantities, five source/two input hashes, every aggregate/count/quantile/oracle/interaction statistic, strict clause vector, and freeze/report ordering. Test, baseline, run and verification commands/timestamps are preserved in `T018A_*` receipts. No failures or deviations.

Research-owned PROJECT_STATE still names T017-C at task start; the newer explicit T018-A inbox governs this work. No research-owned file was edited.

## Bounded conclusion and stop

The existing hard landscape admits a viable local factorized reference target under all five declared clauses. This supports considering a later task to predict hard local directions without reference MSE at deployment. It does not establish that such a predictor can learn safely or generalize, and does not qualify a deployed method. Stop after T018-A: no predictor/head, derivative training, alternate tau/grid, rerendering, fresh split, images/models or GPU experiment. Await research-lead review; no automatic T018-B.
