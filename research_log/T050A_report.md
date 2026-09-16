# T050-A DONE — monotonic tone convergence extension

REFERENCE_ORACLE_ONLY. No deployable changes or official-test access.

material monotonic-tone underconvergence not supported under fixed extension

| Metric | T049 mean | T050 mean | T049 median | T050 median |
|---|---:|---:|---:|---:|
| psnr | 21.872255249768 | 21.874262727355 | 22.646542542570 | 22.647607082917 |
| ssim | 0.504757870543 | 0.505160717504 | 0.540393723574 | 0.540449382935 |

Paired deltas: `{"delta_psnr": {"mean": 0.002007477586970072, "median": 0.0006543314521287869}, "delta_ssim": {"mean": 0.00040284696045923455, "median": 5.309164177508263e-05}, "total_vs_t048_psnr": {"mean": 0.8092827413129902, "median": 0.6004874122174133}, "total_vs_t048_ssim": {"mean": 0.04849690457474201, "median": 0.02788832369653374}}`. Sole gate: mean paired PSNR>=1.00dB AND median>=0.50dB.

Win/equal/loss: `{"psnr": {"win": 99, "equal": 1, "loss": 0}, "ssim": {"win": 70, "equal": 1, "loss": 29}}`.

![Paired changes](T050A_result/paired_tone.png)

## Provenance / fixed protocol

Tested source 9109f3296716a378a8d1424360ccec2c813a9713; branch codex/T050A-tone-convergence; PR https://github.com/word-ky/TTIE/pull/75. T049 acceptedd93d65570c486de06b24a24c2ca529a570f5f45b, freezef7ff8fcb0f405a00b0b07475dcb7f4c87301d2e9c44fade07f6192a4b81a1880, pairs3c92f22701a69591f78737b3d90a6cfb84fe3a62efe6911ad5db29189752ffd1. Exact100 splitb88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b. All62source bindings in T050A_source_binding.json/config/preflight. Selected raw/lift/q/knot/output hashes in freeze.json.

All100low-only reconstructions completed 2026-09-15T16:44:47.222613+00:00, max error 0.0, all bit-exact True, normal decodes0. First reference decode 2026-09-15T16:45:15.064067+00:00; selected outputs frozen 2026-09-15T18:24:58.985299+00:00 before metrics.

All T048 EV/gamma/common-gain raws and lift are immutable buffers. Order: exposure -> shifted gamma -> common gain -> lift -> identity contrast -> clamp -> tone LUT -> hard gate. Inactive outputs exactly original low. OneRGBsharedregionalLUT, x_k=k/8,k=0..8; d=softplus(q), normalize d/sum(d), cumulative interior y, fixed y0=0,y8=1. Same accepted parameterization; q starts at each exact accepted T049 selected state, not identity. Fixed segment indices/fractions from immutableclampedinput; interpolate ylo+(yhi-ylo)*fraction. No channel/spatialinterpolation. Onlyq trainable:4x8storage,8raws peractive region, inactiveq0. FreshAdam.03, exactly1000additionalupdates, one accepted T049 start with fresh moments and no inherited state,1001retained states, earlieststrictfullRGBMSE minimum, float32renderer/float64loss.

Release 20260916-004244-ttie-t050a-extension; preflight 20260916-004324-ttie-t050a-preflight; oracle 20260916-004508-ttie-t050a-oracle; evaluation 20260916-023322-ttie-t050a-eval. Commands/environments saved in run.sh/meta/train.log. GPU1 A6000, TF32off,seed7,threads1. Preflight 5.454667s; oracle summed imagewall 5983.430298s. Preflight/oracle exit0; evaluation computed metrics then initial replay exited1 on a stale prior-task hash constant. Corrected replay run 20260916-023442-ttie-t050a-replay passed.

## Tone / optimizer diagnostics

`{"active_curves": 392, "segment_count": 3136, "segment_min": 1.811981201171875e-05, "segment_max": 0.5917223691940308, "segment_mean": 0.12499999997505386, "segment_median": 0.12267990410327911, "zero_width_segments": 0, "interior_knot_min": 0.062297191470861435, "interior_knot_max": 0.9999817609786987, "interior_knot_mean": 0.5209381680721118, "interior_knot_median": 0.5042145252227783, "fixed_endpoints": [0, 1]}`

Distribution describes selected active segment widths and interior output knots; zero width means floating boundary collapse. Fixed endpoints map0/1to0/1.

Best-step histogram: `{"1000": 50, "998": 2, "984": 3, "414": 1, "999": 10, "910": 1, "994": 1, "659": 1, "816": 1, "991": 1, "992": 1, "995": 5, "671": 1, "240": 1, "993": 1, "981": 1, "977": 1, "997": 2, "799": 1, "983": 1, "499": 1, "952": 1, "986": 2, "966": 1, "996": 1, "918": 1, "559": 1, "988": 1, "0": 1, "978": 1, "987": 1, "640": 1, "542": 1}`; at1000: 50/100.

## Independent replay / tests

`{"status": "PASS", "images": 100, "history_states": 100100, "selected_output_metrics": 200, "scalar_checks": 924, "max_abs_error": 7.105427357601002e-15, "independent_renderer_max_abs": 4.76837158203125e-07, "all_t048_coordinates_exact": true, "all_luts_monotonic": true, "inactive_outputs_exact": true, "all_states_finite_bounded": true, "all_earliest_selected_raws_exact": true, "preflight_before_normals": true, "freeze_before_metrics": true, "seconds": 13.60323457699269}`

Independent NumPy logaddexp/normalization/cumsum reconstructs all100100monotonic LUTstates; checks exact T049 starting q, selected states, hashes, frozenraw/lift, endpoints, inactive outputs. NumPy interpolation verifies selected render from low/affine state (CPU/GPU tolerance1e-6); independent200RGBmetrics plus summaries/counts/distributions/verdict tolerance1e-10. No main renderer/metric helper imports.

Local acceptedT049+focusedT050 tests4passed21.54s; server4passed2.84s; compilePASS. Covers exact renderer/knots reuse, zero-start optimizer equivalence, nonzero accepted q starts, fresh-moment repeatability, frozen affine coordinates, earliest minima and verdict boundaries.

## Limitations / delivery

Remaining mean Retinexformer training-exposed anchor minus T050: -0.395476322924dB / 0.284900491551SSIM. Scalar context only; no baseline use/rerun.
Remaining mean SNR-Aware training-exposed anchor minus T050: 1.522067193390dB / 0.318603677429SSIM. Scalar context only; no baseline use/rerun.

Reference-only marginal capacity at frozen affine state, not deployable performance or globalceiling. The verdict concerns only the fixed additional1000updates; cumulative T050-minus-T048 is descriptive, not a second gate. Boundary-selected winners do not certify convergence or a global capacity ceiling. Stopawaitreview, no sweep/retraining/newoperator.

Failures: initial independent replay retained an incorrect prior-task pairs SHA constant; corrected only that constant to the accepted T049 pairs SHA and replayed frozen artifacts from shared/t050a/replay_corrected.py. Original tested release and scientific source bindings remain unchanged. GPU oracle took about100minutes, exceeding the approximate one-hour estimate; no updates or settings were added. T050-A-EXEC on main29fff601 accepts the same source and is fulfilled by this single run. Initial preflight launch SSH255 timed out before tmux started; confirmed no job/log and started the identical generated run.sh once. No scientific trajectory or settings were changed or repeated. Existing NVML/protobuf warnings nonblocking.

Archives: `{"all_output_history_hashes_unchanged": true, "source_bindings_unchanged": true, "full": {"path": "/media/wenchang/F/wjq/TTIE/shared/t050a/T050A_full.tar", "bytes": 303052800, "sha256": "aed671406d3e507f13966ab6fc73107c3c4cc55fc23961112efb9d8faa76fb54"}, "compact": {"path": "/home/wenchang/asdasdsad/wjq/TTIE/shared/t050a/T050A_compact.tar.gz", "bytes": 9645887, "sha256": "c1e984b45fcb1a45ee0a250b945575019da9c747708ed6b03706b7193964f326"}, "compact_backup": "/media/wenchang/F/wjq/TTIE/shared/t050a/T050A_compact.tar.gz"}`. Fullimages server/F; compacthistory, starts, metrics, receipts, report and recovery projectlocal.
