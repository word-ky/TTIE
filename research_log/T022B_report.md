# T022-B frozen validation trajectory headroom audit

UTC: 2026-09-13T19:30:31.809542+00:00

Status: **audit-complete**. All100 frozen trajectories pass provenance; selected-state reconstruction has maximum and mean pixel error **0**. All4100 native-resolution checkpoint PSNR/SSIM pairs are finite. Independent standard-library aggregation passes, and reconstructed selected metrics equal T022-A exactly.

The practical bottleneck is not primarily which of these saved checkpoints is selected. Even a separate per-image reference oracle gains only **0.147162227699 dB PSNR** or **0.004886823540 SSIM** on average. These small gaps leave the oracle itself at only9.420031dB or0.277893SSIM. This supports prioritizing the trajectory/objective/action-space side in the next scoped study; it does not identify which of those components is causally responsible or authorize changing it now.

## Selected and independent reference oracles

| Rule | Mean PSNR | Median PSNR | Mean SSIM | Median SSIM |
|---|---:|---:|---:|---:|
| selected | 9.272868945613 | 8.680092886101 | 0.273006012336 | 0.242630539370 |
| psnr_oracle | 9.420031173313 | 8.772839067155 | 0.277015757983 | 0.246038538443 |
| ssim_oracle | 9.363257137512 | 8.748445128720 | 0.277892835876 | 0.246038538443 |

PSNR and SSIM oracles are computed separately over saved steps0..40, with earliest-step ties. They use validation references offline and are not deployable selectors. No combined metric objective or new final method is selected.

| Corresponding oracle gain | Mean | Median | p95 | Exact step match |
|---|---:|---:|---:|---:|
| PSNR | 0.147162227699 | 0.013949644080 | 0.714365624419 | 34% |
| SSIM | 0.004886823540 | 0.001741192125 | 0.017339742634 | 34% |

Both cross-metric gain distributions are also retained in summary.json. The34% match statistic compares exact step indices; different indices need not imply a material quality gap. Full selected/oracle step histograms are retained as41-bin arrays, plus all per-image selections in per_image_oracles.json.

## Global fixed steps

All41 uniform-step mean/median/p95 metrics are in T022B_global_fixed_steps.csv and global_fixed_steps.json. Best global mean PSNR is step35: **9.274624949639dB**, only **+0.001756004026dB** over the learned selection. Best global mean SSIM is step32: **0.272972204903**, slightly below learned selection0.273006012336. These are validation diagnostics only, not a selected deployment policy.

## Projection-bound saturation

Percentages are over all400 coordinates per channel, using the saved physical grids and saved per-image projection bounds with tolerance1e-6. Either-bound counts the union; collapsed inactive bounds are included and not double-counted. Upper/lower columns therefore need not sum to the union.

| Rule | EV lower | EV upper | EV either | Gamma lower | Gamma upper | Gamma either |
|---|---:|---:|---:|---:|---:|---:|
| selected | 7.50% | 70.25% | 75.75% | 75.75% | 15.50% | 89.25% |
| psnr_oracle | 5.75% | 86.75% | 90.50% | 72.50% | 4.00% | 74.50% |
| ssim_oracle | 4.75% | 78.75% | 81.50% | 67.50% | 3.00% | 68.50% |

Saturation is common, including at the reference oracles. An EV upper bound may be0 for a bright-gated coordinate; these fractions alone do not prove that widening a particular bound will improve quality. Bounds and all other hyperparameters remain unchanged.

## Frozen inputs, reconstruction and information boundary

Accepted T022-A merge `98054ad96d87f02ff2b6dea60a9199e0214b41f7`; source run `20260914-024025-ttie-t022a-core`. Same100 validation pairs/order, split SHA256 `b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b`. Original freeze SHA256 `510d8979bf60eeed1e5b7937b3c492c25644736cdb9bd7a74d25b9bc8114dbe3`. All original output/decision/trajectory file hashes and byte sizes, low/normal encoded file hashes and config/freeze identities are verified before any reference pixels are read. All100 selected states are then reconstructed and compared against their original output tensors before reference evaluation; max and mean absolute pixel error are0.

Frozen raw state k and original active mask are fed only into the unchanged T014 Region2 EV+gamma renderer. Rendering uses A6000 GPU1, no gradients or parameter update. No trajectory, Adam, CLIP scoring, learned energy, retraining, tuning or output regeneration through adaptation occurs. No new checkpoint image archive is produced. Both scientific renderer and original metric/loader Git blobs match accepted donors (T022B_source_proof.json). The official100 LOL-v2 Real test pairs remain untouched.

PSNR retains float64 MSE arithmetic over original float32 [0,1] pixels. SSIM retains RGB Gaussian11x11 sigma1.5, population covariance, K1=.01/K2=.03, reflect border, complete-map mean, no crop/resize/Y conversion/normalization. GPU handles ISP rendering; SciPy CPU arithmetic remains unchanged. All256 possible uint8 intensity decoding levels were checked to match the preceding T022-A reference conversion exactly.

## Reproduction and evidence

Scientific source `dd62f834245defd7f8d25657cdd9655bfbfa891e`. Run `20260914-032239-ttie-t022b-headroom` completed with exit0. Entry: `python -m ttie.trajectory_headroom --original <T022A/audit> --low-root <low> --normal-root <normal> --split <fixed-split> --freeze <accepted-freeze> --out <audit>`; independent verification: `python scripts/verify_t022b.py <audit> <T022A/audit>`.

Artifact root: `research_log/remote_runs/20260914-032239-ttie-t022b-headroom/artifacts/audit/`. It contains provenance, selected reconstruction,4100-row checkpoint_metrics.csv,41-row global_fixed_steps.json, per_image_oracles.json, summary.json with all histograms/headroom/saturation, and independent_aggregation.json. The standalone verifier imports only standard-library modules and recomputes global steps, each separate oracle, distributions, histograms and saturation. It passed remotely and on fetched local artifacts; selected-vs-T022-A metric error is0. Executed driver hash matches the frozen local source. T022B_artifact_manifest.json lists result hashes.

Four donor loader/metric tests passed11.16s; three new renderer/oracle/saturation tests passed5.54s. The full4100-row run is the affected end-to-end validation. There were no scientific failures or protocol changes. Two Git push connection timeouts recovered on a later direct retry, without persistent Git configuration changes or restarting computation.

Stop after T022-B. Recommend that the research lead scope one trajectory/objective/action-space hypothesis next, using these small oracle gaps to avoid spending the next cycle only on checkpoint selection. No tuning has been launched.

trajectory headroom is limited
