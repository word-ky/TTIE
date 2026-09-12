# T017-B — soft-to-hard transfer dominates frozen T017-A failures

The fixed attribution rule returns **transfer-dominant**: 19/20 harmful hard moves overall (95%) and 10/11 in quadrants (90.91%) improve the same selected soft candidate before becoming harmful after hardening. Both meet the predeclared 2/3 requirement. One quadrant move is already harmful in the joint soft landscape. This is reference-only development diagnosis, not qualification or a deployable selector.

| Group | Episodes | Harmful | Transfer | Interaction | Zero/tie |
|---|---:|---:|---:|---:|---:|
| spatial_pool | 120 | 20 | 19 (95.00%) | 1 (5.00%) | 0 (0.00%) |
| left_right | 40 | 9 | 9 (100.00%) | 0 (0.00%) | 0 (0.00%) |
| quadrants | 40 | 11 | 10 (90.91%) | 1 (9.09%) | 0 (0.00%) |
| offset_left_right_40 | 40 | 0 | 0 (undefined) | 0 (undefined) | 0 (undefined) |

No offset move is harmful, so harmful-category fractions are undefined. Its 35 beneficial and 5 unchanged hard moves do not enter dominance denominators.

## Secondary diagnostics

| Group | Regret mean | Median | p95 | Max | Zero | First oracle | Any tied optimum | Tied / selected tied |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spatial_pool | 3.22436991458e-05 | 0 | 0.000164161715657 | 0.00128931924701 | 104 | 92/120 (76.67%) | 104/120 (86.67%) | 12 / 12 |
| left_right | 1.55713409185e-05 | 0 | 5.40266744792e-05 | 0.00034861266613 | 34 | 29/40 (72.50%) | 34/40 (85.00%) | 5 / 5 |
| quadrants | 1.21489632875e-05 | 0 | 3.24754975736e-05 | 0.000345431268215 | 37 | 37/40 (92.50%) | 37/40 (92.50%) | 0 / 0 |
| offset_left_right_40 | 6.90107932314e-05 | 0 | 0.000239783432335 | 0.00128931924701 | 33 | 26/40 (65.00%) | 33/40 (82.50%) | 7 / 7 |

Oracle ties use exact equality. First-oracle equality uses the original lexicographic candidate order; tie-aware equality counts any minimum. Regret p95 is the inherited linear quantile. None of these statistics overrides the fixed dominance rule.

## Gain sign contingency

Each cell contains hard-negative / hard-zero / hard-positive counts.

| Group | Soft negative | Soft zero | Soft positive |
|---|---|---|---|
| spatial_pool | 1 / 0 / 0 | 0 / 39 / 0 | 19 / 0 / 61 |
| left_right | 0 / 0 / 0 | 0 / 5 / 0 | 9 / 0 / 26 |
| quadrants | 1 / 0 / 0 | 0 / 29 / 0 | 10 / 0 / 0 |
| offset_left_right_40 | 0 / 0 / 0 | 0 / 5 / 0 | 0 / 0 / 35 |

## Provenance and validation

Source freeze: `3351bb8e23f028debd97a31a8066366a8aca2894`. Accepted A merge: `42bafee15964b75c9194e0b92d7f4bae23d049fc`. The formal audit reads seven merged T017-A JSONs plus the original accepted T016-A table/config from `ee5d8fdaf3ab48ee7ad3654d45bdc65419be8367`; all nine raw Git blobs and five scientific source files are hash-bound in `T017B_run/config.json`.

Before attribution, the restricted baseline exactly reproduces the 120x27 grid, all 120 frozen choices, hard MSEs, and entire T017-A summary/evaluation, including `[true,true,true,true,false]` (4/5). Decision bytes remain SHA256 `8aefe5e88aa823e6d415bb1580a00765aaf507e4a106766fef5495e238a1bc26`. The attribution path reads only candidate MSEs and row indices, persists quantities and a hash receipt, then attaches family labels for reporting. The separate reproduction of the old A grouped summary uses its existing family-reporting code.

Focused tests: 4 passed in 0.042 s; compilation passed. One formal CPU run exited 0. Independent verification passed: all source/input hashes, 120 choices and per-episode quantities, every group count/fraction/sign/regret/tie statistic, decision byte preservation, and quantity-freeze/report ordering. Commands and timestamps: `T017B_run_receipt.json`, `T017B_tests.txt`, `T017B_verification.json`; independent script: `T017B_verify.py`.

## Observed deviation and operational failure

The preliminary baseline reused the accepted A loader, which additionally read the legacy T016-B evaluation JSON for its old hard-oracle comparison. This extra comparison JSON did not enter B attribution. That original receipt is preserved; a separate restricted baseline and the formal B run use only the nine permitted A/T017-A inputs. No images or models were read. Initial mailbox HTTPS pull failed with a TLS/schannel handshake; a local fast-forward using the already fetched origin/main succeeded. No attribution rules changed and no formal run failed.

## Stop decision

Within this frozen development table, the current tau=0.05 neighborhood is not a faithful surrogate for hard-boundary deployment. Do not train a hard-deployment geometry objective from it in the next cycle. This does not establish impossibility for other neighborhoods or eliminate the one observed interaction failure. Stop after T017-B; no new choices, thresholds, optimizer, rendering, training, fresh data or GPU experiment. Await research-lead review.
