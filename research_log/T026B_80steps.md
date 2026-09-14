# T026-B:80-step budget probe on gamma0.5

**negative/insufficient.** The sole100-image A6000 run completed8000 target-free updates and passed structural/provenance checks. Mean PSNR gain is only+0.121049121dB, below the predeclared+0.50dB requirement; SSIM passes its floor. No new budget or follow-on experiment is run.

## Measured validation results

| Output | Mean PSNR dB | Median PSNR dB | Mean RGB-SSIM | Median RGB-SSIM |
|---|---:|---:|---:|---:|
| Raw | 8.109722672 | 7.600161541 | 0.160022843 | 0.138977265 |
| Accepted T026-A40steps | 11.120876417 | 10.607319299 | 0.373791825 | 0.367924983 |
| T026-B80steps | 11.241925539 | 10.718345442 | 0.378044410 | 0.376699587 |

| Paired B minus A | Mean | Median | p10 | p90 |
|---|---:|---:|---:|---:|
| psnr | +0.121049121 | +0.112722708 | -0.329288369 | +0.536034250 |
| ssim | +0.004252585 | +0.001489800 | -0.008615336 | +0.020725757 |

The frozen joint decision requires paired meanPSNR>=+0.50dB and meanSSIM>=0.373791825 within1e-12. PSNR fails; SSIM passes. The specified rounded floor is kept exactly, while the full accepted baseline value remains in the table. No selection criterion is changed after results.

PSNR improves on66 images and worsens on34; SSIM improves on59 and worsens on41. These counts are unchanged with either the fixed1e-12 comparison band or a descriptive1e-6 roundoff band. Worst PSNR delta is-1.072141497dB; worst SSIM delta is-0.029993824. Per-image results retain all regressions.

| Three largest PSNR regressions | Delta PSNR dB | Delta SSIM |
|---|---:|---:|
| Train/Low/low00110.png | -1.072141497 | -0.014844102 |
| Train/Low/low00295.png | -1.042870771 | -0.012700415 |
| Train/Low/low00464.png | -0.972495944 | -0.022983712 |

## Exact one-variable experiment

Only max_steps40 to80 changes from merged T026-A. The trajectory function itself is reused unchanged. Active gamma[0.5,1.25], inactive identity, darkEV[0,2], brightEV[-.5,0], hard Region2 geometry, frozen nuisance gate/CLIP/prototypes/T014 energy, renderer, identity initialization, Adam0.03 and earliest minimum predicted-energy selection are unchanged. Actual saved configs differ only in task name, max_steps and start timestamp; all assets and other configuration values are equal.

Source `38c966c622ec00a0d276f8ee5c2651313928e48b`; release `20260914-130657-ttie-t026b-80steps`; run `20260914-130732-ttie-t026b-80steps`, physical A6000GPU1. Accepted baseline run is `20260914-113853-ttie-t026a-gamma05`. The same frozen100-image validation split is used in the same order, SHA256 `b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b`.

All100 output/decision/81-state trajectory artifacts froze at `2026-09-14T05:15:40.284354+00:00` before task-specific reference deployment at `2026-09-14T05:16:05.040592+00:00`. The executable accepts no normal/reference-root argument and logs exactly100 low-only opens. Other-task references remain outside the decode allowlist; global reference absence is not claimed. No T025 oracle files or reference-derived inputs are staged/consumed; official test is untouched.

## Steps, runtime and saturation

Selected-step histogram (nonzero): `{"72": 2, "73": 3, "75": 2, "76": 1, "77": 1, "79": 3, "80": 88}`. All100 episodes perform80 updates and persist81 states;88 selected checkpoints are at80. This endpoint concentration alone does not establish useful additional headroom, as the joint quality gate fails.

Runtime mean/median/p95: 4.703975/4.791697/4.994254 seconds per image. The accepted40-step run's mean was2.274494s; the measured80-step mean is4.703975s. These are recorded runtimes, not a separately controlled hardware benchmark.

Saturation uses1e-6 physical-coordinate tolerance. Inactive collapsed identities are listed separately.

| State/coordinate | Lower hits | Upper hits | Denominator |
|---|---:|---:|---:|
| selected/ev/active | 34 | 9 | 392 |
| selected/ev/inactive | 8 | 8 | 8 |
| selected/gamma/active | 0 | 58 | 392 |
| selected/gamma/inactive | 8 | 8 | 8 |
| final/ev/active | 34 | 9 | 392 |
| final/ev/inactive | 8 | 8 | 8 |
| final/gamma/active | 0 | 57 | 392 |
| final/gamma/inactive | 8 | 8 | 8 |

## Verification and artifacts

- Baseline3tests pass; focused3tests pass: runner only-budget AST equivalence,100 synthetic target files mutated/withheld with identical inference artifact hashes and81-state checks, unchanged metric arithmetic. The scientific100-image run is executed once.
- All100 gate and action-box records equal T026-A. Assets and all other saved config values match. All selected/final states, trajectories, outputs and metrics are finite; selected raw state equals the selected trajectory state and bounds hold.
- All frozen scientific hashes survive reference evaluation;200 fetched decision/trajectory hashes verify locally. Independent metric max absolute error is 3.552713678800501e-15; independent NumPy/Python-statistics means, medians, quantiles and joint verdict agree.
- `T026B_structural_proof.json`, runner source diff, `T026B_config_diff.json`, fixed protocol, launch/evaluation commands, preflight/freeze/reference-deployment receipts, per-image metrics/deltas, summaries and numerical-change counts are persisted.
- Compact evidence is under `research_log/remote_runs/20260914-130732-ttie-t026b-80steps/`. Full float outputs plus source release are retained remotely and backed up to `/media/wenchang/F/wjq/TTIE/shared/t026b/T026B_execution.tar`; exact hashes are in `T026B_artifact_receipt.json`.

The small mean gain does not justify promotion under the agreed materiality rule. Report the negative/insufficient probe and retain the existing promoted candidate pending research-lead review. Do not add another step budget, change method settings, retrain, run baselines or access official test in this cycle.

negative/insufficient
