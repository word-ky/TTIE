# T021-A frozen fresh RGB-SSIM audit

UTC: 2026-09-13T17:25:40.286880+00:00

H0: `region2_ttt_energy_value_only`; H1: `region2_ttt_energy_sobolev`. Binding main commit `d7dc7ec8c2fa4599b946d1860a7453b43f730387`.

Exactly 200 primary rows / 40 source images yield mean paired delta **0.022577036833605**, median **0.007578251829058**, and image-cluster bootstrap 95% CI **[0.017565597352297, 0.028107683054429]**. The lower bound is strictly above zero. The 40 offset-stress rows were excluded before scoring.

| Condition | Rows | Mean delta SSIM |
|---|---:|---:|
| clean | 40 | -0.000228146515955 |
| homogeneous_dark | 40 | +0.051095212481867 |
| homogeneous_bright | 40 | +0.011366921205037 |
| left_right | 40 | +0.033483154083681 |
| quadrants | 40 | +0.017168042913397 |

This supports the specified matched-control metric transfer on the accepted T014 protocol. Clean slightly decreases; this is not a universal-improvement or external SOTA claim.

## Provenance and reproduction

Scientific source `8cfc2e9214534e1668a53e12b5cba3c113aac31e`; corrected verifier `436d9e84b4b973c56747db9e5b820a48ba768d61`. Remote run `20260914-012055-ttie-t021a-ssim`.

Commands: `python -m ttie.ssim_audit --receipt research_log/T021A_outputs_verified.json --out <audit>`; `python research_log/T021A_independent_replay.py <audit>`.

`remote_runs/20260914-012055-ttie-t021a-ssim/artifacts/audit/` contains the selected 200-row binding frozen before tensor reads, full score CSV with output/reference hashes and paths, summary, 10,000 bootstrap draws and means, hashes, and independent verification. All selected 400 output/decision file sizes and SHA256 matched the accepted receipt before loading tensors. Same-source clean-condition frozen identity.image recovers the original clean reference at the original dimensions. No TTT, training, rendering, degradation generation, resizing or cohort reconstruction occurred.

Accepted SSIM arithmetic is unchanged: float64 RGB [0,1], 11x11 Gaussian sigma1.5, population covariance, K1=.01/K2=.03, reflect border, full-map mean/no crop. Bootstrap remains source-image clustered, 10,000 resamples, seed7 PCG64, percentile95%/linear quantiles. CPU SciPy preserves this accepted implementation; subsequent substantial image/model experiments should prefer GPU as requested.

Independent replay imports no TTIE/adaptation modules and uses an explicit separable Gaussian kernel via scipy.convolve1d. All 200 rows, mean/median, five condition means, bootstrap draws, CI and verdict agree. Maximum SSIM error 3.3306690738754696e-16; maximum bootstrap error 3.8163916471489756e-17. Four focused tests passed in 4.51s. Downloaded artifact hashes match. Git archive applied line-ending conversion; exact executed source bytes and their hashes are retained in `T021A_executed_sources/`, with verified equality to committed source after line-ending normalization. No arithmetic changed.

## Observed failures

First independent replay stopped on numeric JSON image IDs versus CSV strings after the evaluator had successfully finished. The verifier now compares serialized IDs explicitly and retains the original numeric cluster ordering. Only independent verification was replayed; score/bootstrap artifacts and initial failure logs remain intact. Remote skimage was absent; the independent explicit-kernel verifier required no package changes. No scientific settings changed after scoring.

Stop after T021-A and await research-lead review of existing PR41. No further metric or method expansion.

positive
