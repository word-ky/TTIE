# T049-A DONE — monotonic tone capacity

REFERENCE_ORACLE_ONLY. No deployable changes or official-test access.

SOTA-scale monotonic-tone capacity not supported under fixed probe

| Metric | T048 mean | T049 mean | T048 median | T049 median |
|---|---:|---:|---:|---:|
| psnr | 21.064979986042 | 21.872255249768 | 21.643626706016 | 22.646542542570 |
| ssim | 0.456663812929 | 0.504757870543 | 0.502562166949 | 0.540393723574 |

Paired deltas: `{"delta_psnr": {"mean": 0.8072752637260204, "median": 0.5988926100953424}, "delta_ssim": {"mean": 0.048094057614282766, "median": 0.027634449011933793}}`. Sole gate: mean paired PSNR>=2.00dB AND median>=1.00dB.

Win/equal/loss: `{"psnr": {"win": 100, "equal": 0, "loss": 0}, "ssim": {"win": 86, "equal": 0, "loss": 14}}`.

![Paired changes](T049A_result/paired_tone.png)

## Provenance / fixed protocol

T049-A-EXEC authorization ea7112c2 was observed during publication after the sole run/replay completed. It specifies the exact same tested source a93a37af and settings; this completion covers both T049-A and T049-A-EXEC without a second run.

Tested source a93a37af1e45cd568e10d09d5a18dad3dca4554b; branch codex/T049A-monotonic-tone; PR https://github.com/word-ky/TTIE/pull/74. T048 accepted152ae5b757df2423a96e003d7185192720a0fd8e, freeze6ff1ad745422d8842a31bb62f17dada5d170bab9b0d9a61d2a21ce23d7f2da2b, pairs29a33623bbcfc8d9500254344173b68556a6187ce15b1115a435b53fb2a18b48. Exact100 splitb88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b. All56source bindings in T049A_source_binding.json/config/preflight. Selected raw/lift/q/knot/output hashes in freeze.json.

All100low-only reconstructions completed 2026-09-15T13:23:36.449518+00:00, max error 0.0, all bit-exact True, normal decodes0. First reference decode 2026-09-15T13:24:25.146532+00:00; selected outputs frozen 2026-09-15T13:44:43.467287+00:00 before metrics.

All T048 EV/gamma/common-gain raws and lift are immutable buffers. Order: exposure -> shifted gamma -> common gain -> lift -> identity contrast -> clamp -> tone LUT -> hard gate. Inactive outputs exactly original low. OneRGBsharedregionalLUT, x_k=k/8,k=0..8; d=softplus(q), normalize d/sum(d), cumulative interior y, fixed y0=0,y8=1. q0equalincrements gives exact identity knots. Fixed segment indices/fractions from immutableclampedinput; interpolate ylo+(yhi-ylo)*fraction. No channel/spatialinterpolation. Onlyq trainable:4x8storage,8raws peractive region, inactiveq0. FreshAdam.03, exactly500updates, onezero start,501retained states, earlieststrictfullRGBMSE minimum, float32renderer/float64loss.

Release 20260915-212240-ttie-t049a-tone; preflight 20260915-212322-ttie-t049a-preflight; oracle 20260915-212417-ttie-t049a-oracle; evaluation 20260915-214534-ttie-t049a-eval. Commands/environments saved in run.sh/meta/train.log. GPU1 A6000, TF32off,seed7,threads1. Preflight 3.268269s; oracle summed imagewall 1217.729844s. Successful runs exit0.

## Tone / optimizer diagnostics

`{"active_curves": 392, "segment_count": 3136, "segment_min": 0.002351999282836914, "segment_max": 0.5688010454177856, "segment_mean": 0.1250000000213824, "segment_median": 0.12297841906547546, "zero_width_segments": 0, "interior_knot_min": 0.05780346691608429, "interior_knot_max": 0.9976480007171631, "interior_knot_mean": 0.5212575171153677, "interior_knot_median": 0.5041643381118774, "fixed_endpoints": [0, 1]}`

Distribution describes selected active segment widths and interior output knots; zero width means floating boundary collapse. Fixed endpoints map0/1to0/1.

Best-step histogram: `{"500": 91, "498": 2, "499": 1, "474": 1, "493": 1, "488": 1, "169": 1, "497": 1, "495": 1}`; at500: 91/100.

## Independent replay / tests

`{"status": "PASS", "images": 100, "history_states": 50100, "selected_output_metrics": 200, "scalar_checks": 720, "max_abs_error": 7.105427357601002e-15, "independent_renderer_max_abs": 4.76837158203125e-07, "all_t048_coordinates_exact": true, "all_luts_monotonic": true, "inactive_outputs_exact": true, "all_states_finite_bounded": true, "all_earliest_selected_raws_exact": true, "preflight_before_normals": true, "freeze_before_metrics": true, "seconds": 13.035195635980926}`

Independent NumPy logaddexp/normalization/cumsum reconstructs all50100monotonic LUTstates; checks identity, selected states, hashes, frozenraw/lift, endpoints, inactive outputs. NumPy interpolation verifies selected render from low/affine state (CPU/GPU tolerance1e-6); independent200RGBmetrics plus summaries/counts/distributions/verdict tolerance1e-10. No main renderer/metric helper imports.

Local acceptedT048+focusedT049 tests4passed27.95s; server4passed2.51s; compilePASS. Covers identity/endpoints/monotonicity/qinventory, gateabstention, frozenaffine, qonlyupdates, earliestselection and verdictboundaries.

## Limitations / delivery

Remaining mean Retinexformer training-exposed anchor minus T049: -0.393468845337dB / 0.285303338511SSIM. Scalar context only; no baseline use/rerun.
Remaining mean SNR-Aware training-exposed anchor minus T049: 1.524074670977dB / 0.319006524389SSIM. Scalar context only; no baseline use/rerun.

Reference-only marginal capacity at frozen affine state, not deployable performance or globalceiling. The SOTA-scale gate names the requested increment, not held-outSOTA. Stopawaitreview, no sweep/retraining/newoperator.

Failures: archive-helper preparation initially referenced a file excluded by sparse checkout; corrected to existing accepted worktree path before helper execution. Main push raced research-lead inbox update and was rejected; rebased the single append onto the new main and pushed successfully, preserving both authors. No scientific/test/run failure or deviation. Existing NVML/protobuf warnings nonblocking.

Archives: `{"all_output_history_hashes_unchanged": true, "source_bindings_unchanged": true, "full": {"path": "/media/wenchang/F/wjq/TTIE/shared/t049a/T049A_full.tar", "bytes": 296192000, "sha256": "2df166b9e71da3d2cac44dee2855ce4383b1f42752832069209e3151fa3e41de"}, "compact": {"path": "/home/wenchang/asdasdsad/wjq/TTIE/shared/t049a/T049A_compact.tar.gz", "bytes": 5786340, "sha256": "d6846306110b6e2e2556e4c486d42fb6cca81a7e6160f610c481c998e7757504"}, "compact_backup": "/media/wenchang/F/wjq/TTIE/shared/t049a/T049A_compact.tar.gz"}`. Fullimages server/F; compacthistory, starts, metrics, receipts, report and recovery projectlocal.
