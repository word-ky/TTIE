# T017-C — matched-soft rescue is insufficient (2/5)

**Literal outcome: matched_soft_insufficient.** The frozen choices improve pooled MSE by 4.0170% over fixed soft, but fail pooled improvement and both left/right and quadrant safety against canonical hard Region2. The five-clause vector is `[false, true, true, false, false]`. This is a reference-only development capacity diagnostic, not a deployable selector or fresh qualification.

| Clause | Observed ratio | Required | Pass |
|---|---:|---:|---|
| Pooled S1/H0 | 1.04531306509 | <= 0.97 | False |
| Pooled S1/S* | 1.00088099316 | <= 1.03 | True |
| Offset S1/H0 | 0.902920140935 | <= 0.95 | True |
| Left/right S1/H0 | 1.05572902919 | <= 1.01 | False |
| Quadrants S1/H0 | 1.22136055801 | <= 1.01 | False |

The additional adaptive test passes: pooled S1/S0 = 0.959829954765 <= 0.99. This cannot override three failed viability clauses.

## MSE and ratios

| Group | H0 hard canonical | S0 fixed soft | S1 frozen-choice soft | S* nine-soft oracle |
|---|---:|---:|---:|---:|
| spatial_pool | 0.0350435737482 | 0.0381645783239 | 0.0366315054862 | 0.0365992617871 |
| left_right | 0.0333970155101 | 0.0364768592641 | 0.0352581987623 | 0.0352426274214 |
| quadrants | 0.0309838496498 | 0.0381196441362 | 0.0378424518974 | 0.0378303029342 |
| offset_left_right_40 | 0.0407498560846 | 0.0398972315714 | 0.036793865799 | 0.0367248550057 |

| Group | S1/H0 | S0/H0 | S1/S0 | S1/S* | S0/S* |
|---|---:|---:|---:|---:|---:|
| spatial_pool | 1.04531306509 | 1.08906068194 | 0.959829954765 | 1.00088099316 | 1.04276907403 |
| left_right | 1.05572902919 | 1.09221913117 | 0.966590859894 | 1.00044183258 | 1.03502099398 |
| quadrants | 1.22136055801 | 1.23030690399 | 0.992728362371 | 1.0003211437 | 1.0076483977 |
| offset_left_right_40 | 0.902920140935 | 0.979076625168 | 0.922216012235 | 1.00187913045 | 1.08638227612 |

## Fixed smoothing versus adaptive movement

| Group | Fixed gain (H0-S0)/H0 | Adaptive gain (S0-S1)/S0 | S1<S0 / = / > | Recovered soft headroom | Zero-headroom episodes |
|---|---:|---:|---|---:|---:|
| spatial_pool | -8.906068% | 4.017005% | 80 / 39 / 1 | 97.940116% | 39 |
| left_right | -9.221913% | 3.340914% | 35 / 5 / 0 | 98.738378% | 5 |
| quadrants | -23.030690% | 0.727164% | 10 / 29 / 1 | 95.801164% | 29 |
| offset_left_right_40 | 2.092337% | 7.778399% | 35 / 5 / 0 | 97.824634% | 5 |

Gains retain their signs. Fixed softness worsens pooled MSE by 8.9061%; adaptive movement recovers 4.0170% relative to that worse fixed-soft baseline. On quadrants, the fixed-soft penalty is 23.0307%, while movement recovers only 0.7272% relative to fixed soft. Staying soft therefore does not remove the family-safety failure against H0.

The percentage gains have different denominators and must not be added. The absolute-MSE decomposition is additive:

| Group | H0-S0 | S0-S1 | H0-S1 |
|---|---:|---:|---:|
| spatial_pool | -0.00312100457571 | 0.00153307283763 | -0.00158793173808 |
| left_right | -0.00307984375395 | 0.00121866050176 | -0.00186118325219 |
| quadrants | -0.00713579448638 | 0.000277192238718 | -0.00685860224767 |
| offset_left_right_40 | 0.000852624513209 | 0.0031033657724 | 0.00395599028561 |

## Soft-oracle headroom

Aggregate recovery is (mean(S0)-mean(S1))/(mean(S0)-mean(S*)). It is distinct from the mean of per-episode recovery fractions. Per-episode denominators S0-S* equal zero on 39/120 rows (5 LR, 29 quadrants, 5 offset); their fractions remain null and are excluded from defined-value distributions. No clipping or epsilon is applied, including the negative quadrant recovery.

| Group | Aggregate headroom MSE | Mean defined recovery | Median | p95 | Min | Max | Nulls |
|---|---:|---:|---:|---:|---:|---:|---:|
| spatial_pool | 0.00156531653677 | 0.955275326657 | 1 | 1 | -0.552783173689 | 1 | 39 |
| left_right | 0.00123423184268 | 0.977299446683 | 1 | 1 | 0.67991079369 | 1 | 5 |
| quadrants | 0.000289341202006 | 0.833834028507 | 1 | 1 | -0.552783173689 | 1 | 29 |
| offset_left_right_40 | 0.00317237656564 | 0.971418471764 | 1 | 1 | 0.722051018225 | 1 | 5 |

## Reproduction and evidence

Source freeze `1512d03347dbc83e85837e835067dfd8255066b9`; accepted B merge `0b052a0fd04acb12cdaa0ad69b9207c18c063119`. Eight permitted JSON inputs and six scientific source files are hash-bound in `T017C_run/config.json`. Inputs comprise B config/quantities/freeze/report receipt, A decisions/freeze, and original T016-A table/config. No legacy T016-B comparison input.

Before new aggregation, all 120 frozen choices and all T017-B per-episode quantities, including S0,S1,H0,H1,S*, reproduce exactly. Original decision bytes remain SHA256 `8aefe5e88aa823e6d415bb1580a00765aaf507e4a106766fef5495e238a1bc26`. New quantities and choice hash are persisted before family reporting; family labels never affect choices or per-episode arithmetic.

Three focused tests passed in 0.046 s; compilation passed. One formal CPU run exited 0. Independent `T017C_verify.py` passed: six source/eight input hashes, all 120 choices/B quantities/new decomposition values, every group MSE/ratio/count/distribution, exact clause vector and immutable choice/freeze receipts. Commands/timestamps are preserved in test, baseline, run and verification receipts. No failures or deviations.

## Stop decision

This frozen tau=0.05 matched-soft rescue is closed as insufficient. Near-oracle choice within the soft renderer does not imply a good renderer relative to canonical hard Region2. T014 remains the best fresh-validated deployable method. No alternate tau, smoothing schedule, selector, trainable geometry, rendering, images, new data or GPU experiment was used. Stop after T017-C and await research review; do not begin T017-D or a soft-geometry model.
