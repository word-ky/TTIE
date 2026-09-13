# T019-A — fixed1% utility-deadband target viable

All five development performance clauses pass and the harmful count is exactly zero. The120 development episodes yield56 beneficial,64 equal and0 harmful combined choices. A fixed1% per-axis deadband suppresses26 original non-center axis labels while retaining97.60179195810709% of the available nine-hard oracle headroom. This is a reference-only development target viability result. It does not establish a learned selector or fresh generalization.

## Fixed rule and input boundary

Delta is exactly0.01, as predeclared by the research lead from the existing1% safety tolerance; no threshold was searched. For each axis, compute the two relative improvements (H0-H_lower)/H0 and (H0-H_upper)/H0. Remain at0.5 unless their maximum is >=0.01. When a move qualifies, choose the lower-MSE non-center candidate; exact lower/upper ties choose0.4 before0.6. Apply this same rule independently to x and y. H_delta is the saved hard candidate at their combined location. All comparisons use ordinary full-precision Python float arithmetic and the exact requested formulas, without rounding relaxation.

The only data inputs are the two accepted T016-A development artifacts used by T018-A, from commit ee5d8fdaf3ab48ee7ad3654d45bdc65419be8367:

- candidate_metrics.json SHA2566091a0c928f115940997a647693b6571d8608131e7f9567a75882f0233b9c41e;
- config.json SHA256e32b8b48ec0a93749698f51ab33637c5945bc8d827ec133175134bff93b168c5.

Both original paths are recorded in T019A_run/config.json. The original120x27 table is reused unchanged: choice reads only five tau0 cross entries[12,3,21,9,15], and oracle/combined evaluation uses only nine tau0 entries[0,3,6,9,12,15,18,21,24]. No soft value affects the rule. The previously accepted T018-A original labels are exactly reproduced from these same five values for the suppression comparison; no additional label artifact is needed.

**No T018-E fresh MSE, features, logits, family labels or outcomes were inputs to this audit.** No images were read, rerendered or regenerated; no CLIP/TTT/GPU work, model training, normalization fitting, OOF head, confidence rule or family-specific decision logic was run. The earlier E provenance review repair was delivered separately in PR31 and is not part of this audit's data path. The research lead's fixed T019-A specification is preserved in T019A_spec.md.

## MSE and literal performance clauses

The reused measurement helper stores H_delta as H1, explicitly declared in config/summary. The following means and comparisons are unrounded values from the result artifacts.

| Group | H0 | H_delta | H* | H_delta/H0 | H_delta/H* | Beneficial/equal/harmful |
|---|---:|---:|---:|---:|---:|---|
| spatial_pool | 0.03504357374816512 | 0.032623040299707404 | 0.032563564518932255 | 0.9309278937744054 | 1.0018264517921731 | 56/64/0 |
| left_right | 0.03339701551012695 | 0.032212557108141485 | 0.032170985778793695 | 0.9645340044943147 | 1.0012921994257071 | 21/19/0 |
| quadrants | 0.030983849649783225 | 0.030983849649783225 | 0.030982188356574625 | 1.0 | 1.000053620912425 | 0/40/0 |
| offset_left_right_40 | 0.04074985608458519 | 0.034672714141197505 | 0.03453751942142844 | 0.850867155683366 | 1.0039144305101768 | 35/5/0 |

| Clause | Observed ratio | Maximum | Pass |
|---|---:|---:|---|
| Pooled gain |0.9309278937744054|0.97|true|
| Pooled oracle proximity |1.0018264517921731|1.03|true|
| Offset gain |0.850867155683366|0.95|true|
| Left/right safety |0.9645340044943147|1.01|true|
| Quadrants safety |1.0|1.01|true|

The additional safety condition H_delta<=H0 holds for all120 rows. Acceptance vector[five performance clauses, zero harmful] is[true,true,true,true,true,true]. Literal outcome: **T019-A utility-deadband target viable**.

## Direction distributions and suppression

For x/y distributions below, each triplet is count(0.4),count(0.5),count(0.6). Suppression counts compare against exact T018-A center-first local argmin labels on the same development table.

| Group | x distribution | y distribution | Suppressed x | Suppressed y | Suppressed axes / affected episodes |
|---|---|---|---:|---:|---|
| spatial_pool | [32, 88, 0] | [27, 79, 14] | 4 | 22 | 26 / 26 |
| left_right | [0, 40, 0] | [14, 19, 7] | 0 | 12 | 12 / 12 |
| quadrants | [0, 40, 0] | [0, 40, 0] | 1 | 0 | 1 / 1 |
| offset_left_right_40 | [32, 8, 0] | [13, 20, 7] | 3 | 10 | 13 / 13 |

| Group | (0.4,0.4) | (0.4,0.5) | (0.4,0.6) | (0.5,0.4) | (0.5,0.5) | (0.5,0.6) | (0.6,0.4) | (0.6,0.5) | (0.6,0.6) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spatial_pool | 11 | 15 | 6 | 16 | 64 | 8 | 0 | 0 | 0 |
| left_right | 0 | 0 | 0 | 14 | 19 | 7 | 0 | 0 | 0 |
| quadrants | 0 | 0 | 0 | 0 | 40 | 0 | 0 | 0 | 0 |
| offset_left_right_40 | 11 | 15 | 6 | 2 | 5 | 1 | 0 | 0 | 0 |

The26 suppressed axis labels occur in26 episodes;13 episodes change from a single-axis move to no move and13 from a two-axis move to a single-axis move. Therefore no-move episodes increase from51 to64. These counts are descriptive only and do not alter delta or the rule.

## Oracle headroom retained

Recovery is (mean(H0)-mean(H_delta))/(mean(H0)-mean(H*)). Per-episode recovery uses each row's own denominator; exact zero-headroom denominators remain null, following T018-A. Null is not replaced with zero or one.

| Group | Aggregate recovery | Zero-headroom / null episode count |
|---|---:|---:|
| spatial_pool | 0.9760179195810709 | 51 |
| left_right | 0.9660927232958842 | 7 |
| quadrants | 0.0 | 39 |
| offset_left_right_40 | 0.9782377023172527 | 5 |

Pooled headroom capture is97.6018%, versus the accepted T018-A target's98.2572%. Quadrants deliberately gives up its single tiny development gain and remains at canonical center for all40 rows: its positive but tiny aggregate oracle headroom is1.6612932086005894e-6, so aggregate recovery is0 while39 per-row recoveries remain null. Full per-episode recovery distributions and factorization regret are stored in summary.json.

## Interaction failures and exact ties

There are no harmful combined choices and no interaction failure where both independently qualified axis moves produce H_delta>H0; the exact case list is empty. The focused synthetic test separately covers a harmful joint choice despite individually qualifying axes, and verifies that zero-harmful acceptance rejects it even when all five aggregate clauses pass.

No formal axis lands exactly at the1% threshold. There are19 exact lower/upper loss ties across12 episodes; every one has gains[0,0] and remains centered. No formal moving tie invokes the0.4-before0.6 priority. Exact threshold equality, immediately adjacent float values, and qualifying lower/upper ties are covered in focused tests. All observed tie cases are:

| Row index | Family | Axis | Gains | Target |
|---:|---|---|---|---:|
| 0 | left_right | x | [0.0, 0.0] | 0.5 |
| 0 | left_right | y | [0.0, 0.0] | 0.5 |
| 2 | offset_left_right_40 | x | [0.0, 0.0] | 0.5 |
| 2 | offset_left_right_40 | y | [0.0, 0.0] | 0.5 |
| 11 | offset_left_right_40 | x | [0.0, 0.0] | 0.5 |
| 11 | offset_left_right_40 | y | [0.0, 0.0] | 0.5 |
| 42 | left_right | x | [0.0, 0.0] | 0.5 |
| 42 | left_right | y | [0.0, 0.0] | 0.5 |
| 44 | offset_left_right_40 | x | [0.0, 0.0] | 0.5 |
| 44 | offset_left_right_40 | y | [0.0, 0.0] | 0.5 |
| 72 | left_right | y | [0.0, 0.0] | 0.5 |
| 74 | offset_left_right_40 | y | [0.0, 0.0] | 0.5 |
| 84 | left_right | y | [0.0, 0.0] | 0.5 |
| 90 | left_right | y | [0.0, 0.0] | 0.5 |
| 92 | offset_left_right_40 | y | [0.0, 0.0] | 0.5 |
| 95 | offset_left_right_40 | x | [0.0, 0.0] | 0.5 |
| 95 | offset_left_right_40 | y | [0.0, 0.0] | 0.5 |
| 119 | offset_left_right_40 | x | [0.0, 0.0] | 0.5 |
| 119 | offset_left_right_40 | y | [0.0, 0.0] | 0.5 |

## Freeze, validation and reproducibility

Scientific source **d3e83c0782cd2727c902aca14a0f199b8621e06b** binds six scientific files. The new compact module reuses the unchanged accepted hard_local input precheck, quantity calculations, grouped statistics, oracle conventions and five performance clauses; only fixed-deadband choices and their diagnostics/zero-harmful acceptance are added.

- Decisions frozen **2026-09-13T06:14:30.260367+00:00**, SHA256 **d874bed74b0ebf68b680c8b6e60c8c5a44c9d82ce3cb7fd5730e16e53a980d45**.
- Decision freeze binds config SHA256eac710a5b3cc9b5627da33f53343796e0bf514c9409afb826b2ffd1c46ee660d, which pins source, exact input origins/hashes, delta and original Git-object pack.
- Quantities frozen **2026-09-13T06:14:30.269810+00:00** before family aggregation begins **2026-09-13T06:14:30.279063+00:00**.
- All120 choices and quantities exclude condition/image-ID fields. Family values enter grouped reporting only after these freezes.
- Exactly one formal CPU command ran from2026-09-13T06:14:29.341082+00:00 to2026-09-13T06:14:30.316065+00:00, exit0: python -m ttie.utility_deadband --source-sha d3e83c0782cd2727c902aca14a0f199b8621e06b --output research_log/T019A_run.

Four existing hard_local baseline tests PASS; original120 T018-A choices reproduce exactly. Four new focused tests PASS, including threshold/tie behavior, five-cross-only access, suppression, freeze ordering/zero-headroom nulls and the independent zero-harmful condition. The independent verifier imports no audit code and exactly recomputes all120 choices/quantities, every group statistic/distribution/suppression count, all tie and interaction cases, the five clauses and zero-harmful requirement.

A portable17-file artifact workspace with no discoverable Git checkout also verifies successfully. The82744-byte origin pack contains19 actual Git objects proving all six source locations and two accepted input locations, so verification does not require unreachable history after a squash. Pack SHA25643af6cdfb0ac0950b709478a12923fe1df56ac829ba2d8329ec0853d9b13277f. To review saved evidence, run python research_log/T019A_verify.py; do not relaunch the audit to obtain a different result. Original run, verification, portable verification and command receipts are in research_log/T019A*.

No scientific failure or deviation occurred. A console-only summary display attempted dictionary iteration on the decision list once; the saved artifacts and verifier were unaffected. Stop after reporting this reference-target result. A learned deadband selector or fresh qualification requires a separately issued research task.
