# T026-A: active gamma lower bound 0.8 to 0.5

**materially positive** under the predeclared validation-only joint gate. The single target-free A6000 run completes all 100 validation images. Only the active gamma lower bound changes; no T025 oracle state, reference pixel or metric enters adaptation or predicted-energy selection. No official-test image is accessed.

## Results

| Output | Mean PSNR dB | Median PSNR dB | Mean RGB-SSIM | Median RGB-SSIM |
|---|---:|---:|---:|---:|
| Raw | 8.109722672 | 7.600161541 | 0.160022843 | 0.138977265 |
| Accepted T022-C | 10.229554025 | 9.650373830 | 0.328231478 | 0.300251538 |
| T026-A gamma lower0.5 | 11.120876417 | 10.607319299 | 0.373791825 | 0.367924983 |

| Paired T026-A minus T022-C | Mean | Median | p10 | p90 |
|---|---:|---:|---:|---:|
| psnr | +0.891322392 | +0.857525058 | -0.000000027 | +2.012317231 |
| ssim | +0.045560347 | +0.049376713 | -0.000522597 | +0.095914172 |

PSNR mean gain 0.891322392 dB exceeds +0.50 dB; mean SSIM 0.373791825 exceeds the frozen floor 0.3282314776612914 (1e-12 tolerance). Both gate conditions pass. This establishes a material improvement for this validation split; it is not official-test qualification or a SOTA claim.

The benefit is not universal. The fixed comparison script uses1e-12 to count signed changes: PSNR88 improved/12 worsened, SSIM80 improved/20 worsened. With a separate descriptive1e-6 roundoff band (not a changed acceptance gate), PSNR84 improve/5 worsen/11 are within tolerance; SSIM77 improve/12 worsen/11 are within tolerance. All per-image values remain available; losses are not discarded.

| Three largest PSNR regressions | Delta PSNR dB | Delta SSIM |
|---|---:|---:|
| Train/Low/low00464.png | -2.030222440 | -0.050217393 |
| Train/Low/low00643.png | -0.613541140 | -0.009173150 |
| Train/Low/low00449.png | -0.184960536 | -0.006096033 |

## Frozen experiment and information boundary

Source `b2359721c89db732d17e03be273e0bdb71bb377a`; release `20260914-113810-ttie-t026a-gamma05`; job `20260914-113853-ttie-t026a-gamma05`; A6000 physical GPU1. The baseline source is accepted T022-C `824f36d9a7614acffa88977c4d128d0aa7afc85b`. Split SHA256 `b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b`. T014 energy SHA256 `c3d1eef9f20af163e1268823cf16d3fdaed315e7723fc94db3b1138ad1336521`. All CLIP/prototype/gate assets match accepted T022-C before and after inference.

Active gamma becomes[0.5,1.25], inactive gamma remains1. Dark EV[0,2], bright EV[-0.5,0], inactive EV0, Region2 geometry, renderer, identity initialization, Adam0.03,40 updates, and minimum predicted-energy/earliest-tie selection are unchanged. The existing physical map supports0.5 as an asymptotic raw endpoint; no extra raw cap, ISP operator, objective, training or search schedule is introduced.

All outputs, decisions and trajectories froze at `2026-09-14T03:42:56.673912+00:00`. The task-specific normal directory was deployed at `2026-09-14T03:43:30.371075+00:00`, after the freeze. Other-task references already existed elsewhere on the server and were outside the inference decoder allowlist; global reference absence is not claimed. The executable exposes no normal/reference-root argument and records exactly100 allowed low-light opens. No oracle files were staged or loaded.

Evaluation rechecks every frozen artifact hash before reference decoding, uses native full-frame RGB float PSNR and Gaussian11/sigma1.5 population/full-map-reflect SSIM, and compares paired only against T022-C. Independent metric max absolute error is 5.329070518200751e-15. NumPy aggregation agrees with Python statistics, including mean/median/quantiles and the joint decision.

## Selected steps, boundaries and runtime

Selected-step histogram (nonzero): `{"20": 1, "31": 1, "33": 1, "34": 3, "35": 2, "36": 1, "37": 1, "38": 1, "39": 1, "40": 88}`. All100 images run exactly40 updates;88 select step40. No second candidate or longer run is tried.

Runtime: mean 2.274494s, median 2.313844s, p95 2.453943s per image. These measurements are descriptive, not a hardware-controlled speedup claim.

Physical-bound saturation tolerance is1e-6; inactive identity coordinates are separated from active coordinates. Both selected and final states are checked against their boxes.

| State/coordinate | Lower hits | Upper hits | Denominator |
|---|---:|---:|---:|
| selected/ev/active | 26 | 10 | 392 |
| selected/ev/inactive | 8 | 8 | 8 |
| selected/gamma/active | 0 | 55 | 392 |
| selected/gamma/inactive | 8 | 8 | 8 |
| final/ev/active | 27 | 10 | 392 |
| final/ev/inactive | 8 | 8 | 8 |
| final/gamma/active | 0 | 55 | 392 |
| final/gamma/inactive | 8 | 8 | 8 |

Zero active gamma coordinates meet the tight0.5 lower-saturation tolerance at either selected or final state;55/392 meet upper1.25. The finite40-step trajectory and nonlinear raw-to-physical map must be kept in mind when interpreting saturation. Inactive lower/upper hits are the same collapsed identity coordinates.

## Tests and evidence

- Accepted baseline:3 tests pass. New variant:3 tests pass (exact bound/endpoint and finite gradients, runner/trajectory AST equivalence after documented wiring/metadata normalization,100 synthetic reference files mutated/withheld with identical output/decision/trajectory hashes). Real validation inference is run once.
- All100 gates exactly match T022-C; all100 boxes differ only at active gamma lower; selected raw state matches saved trajectory selection; all4000 updates, outputs, trajectories, metrics and bounds pass.
- All100 pre-reference hashes survive scoring;200 fetched decision/trajectory hashes match freeze. Independent metric and aggregation receipts pass.
- Source/config/split/asset bindings, launch/evaluation commands, structural proof/diff, low-only access log, reference deployment receipt, per-image metrics/deltas, step/saturation/runtime summaries and numerical-change counts are project-local.
- Compact evidence: `research_log/remote_runs/20260914-113853-ttie-t026a-gamma05/`. The100 full float outputs remain in the original remote run and `/media/wenchang/F/wjq/TTIE/shared/t026a/T026A_execution.tar`; exact archive hashes are in `T026A_artifact_receipt.json`.

This result supports retaining the gamma-range variant for research-lead review. The mean improvement coexists with real per-image regressions and does not settle generalization. No further bound/LR/step/selector change, retraining, baseline run or official test is performed. Stop after reporting; research lead owns promotion and the next task.

materially positive
