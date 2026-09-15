# T048-A DONE — post-gamma affine closure

REFERENCE_ORACLE_ONLY. One fixed probe; no deployable changes or official-test access.

post-gamma affine coupling materially supported

| Metric | T047 mean | T048 mean | T047 median | T048 median |
|---|---:|---:|---:|---:|
| psnr | 20.055972040168 | 21.064979986042 | 19.294980253326 | 21.643626706016 |
| ssim | 0.414273525771 | 0.456663812929 | 0.442005243193 | 0.502562166949 |

delta_psnr: mean +1.009007945875, median +0.558350473396.

delta_ssim: mean +0.042390287158, median +0.031274867560.

total_vs_t046_psnr: mean +1.491752144287, median +0.775739218132.

total_vs_t046_ssim: mean +0.047296728314, median +0.043263733988.

Sole positive gate: paired T048-minus-T047 PSNR mean >= +0.50 dB AND median >= +0.25 dB. T046 totals and all other quantities are descriptive.

Win/equal/loss: `{"psnr": {"win": 100, "equal": 0, "loss": 0}, "ssim": {"win": 97, "equal": 0, "loss": 3}}`.

![Paired changes](T048A_result/paired_affine.png)

## Provenance and fixed protocol

Tested source 44c74506b7866b5b726a48e8f2cedb0479e9157e; branch `codex/T048A-affine-closure`; PR https://github.com/word-ky/TTIE/pull/73. T047 accepted merge6659d7aeec5afa94cdfa3013518daa714ab7849e. T047freeze c82e6b609c14a0464e2cab1fdf6a5e11036aeac6e8d800c708e8640646e6a75e; pairs d4b12f46734c05369efea777b5e699e659d82a16d1e969d4bd93a9cdecd37790. T047 raw state equals accepted T046; priorpreflight90edc063 binds gates/raw/source. Exact100 split b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b. Full source bindings in T048A_source_binding.json/config/preflight; selected gain/lift/raw/output/file hashes in freeze.json.

All100 low-only T047 reconstructions completed 2026-09-15T12:26:22.032662+00:00; max error 0.0; bit-exact True; normals0. First reference decode 2026-09-15T12:26:55.438077+00:00. All100 outputs frozen 2026-09-15T12:37:19.521786+00:00 before metric aggregation.

Exact order: exposure -> shifted gamma -> common gain -> additive lift -> accepted identity contrast -> clamp -> hard-gate compositing. EV/gamma are immutable buffers, bit-exact accepted T046 states. Only immutable pre-gain pixels are cached. Common-gain raw initializes at T046 accepted value and uses the unchanged physical tanh map/bounds[0.5,2]; accepted projection forces inactive gain raw0. Lift initializes at T047 selected physical value, projection[-0.20,+0.20], inactive0. One fresh Adam per image with gainraw lr0.05 and lift lr0.01; exactly500updates, one start, retain501states, earliest strict fullRGBMSE minimum. Float32 renderer/float64 MSE. All retained starts, selected states, bounds, frozen EV/gamma and output identities verified independently.

Release 20260915-202436-ttie-t048a-affine-complete; preflight 20260915-202511-ttie-t048a-preflight-complete; oracle 20260915-202646-ttie-t048a-oracle; evaluation 20260915-203828-ttie-t048a-eval. Saved meta/run.sh/train.log contain exact commands. GPU1 A6000, TF32off, seed7, threads1. Successful preflight 3.127647s; oracle image-wall sum 623.541036s. Successful preflight/oracle/evaluation exit0.

## Descriptive distributions

Selected active physical gain: `{"active_coordinates": 392, "mean": 1.1097304139818465, "median": 1.1119588613510132, "min": 0.5001010298728943, "max": 1.9999998807907104, "lower_bound_hits": 0, "upper_bound_hits": 0}`.

Selected active lift: `{"active_coordinates": 392, "mean": 0.09932004957078547, "median": 0.08440084755420685, "min": -0.02416345104575157, "max": 0.20000000298023224, "lower_bound_hits": 0, "upper_bound_hits": 87}`. Bound hits count exact selected physical bounds, denominator active coordinates.

Best-step histogram: `{"202": 1, "203": 1, "500": 12, "477": 1, "240": 1, "307": 2, "372": 1, "422": 1, "343": 1, "432": 1, "270": 1, "160": 1, "263": 1, "212": 1, "205": 1, "301": 1, "330": 1, "191": 1, "418": 1, "413": 1, "265": 1, "410": 1, "352": 1, "375": 1, "496": 1, "230": 1, "247": 1, "227": 1, "206": 1, "286": 1, "253": 1, "187": 2, "195": 2, "204": 1, "466": 1, "465": 1, "190": 1, "404": 2, "395": 1, "171": 1, "294": 2, "471": 1, "222": 1, "406": 1, "233": 1, "172": 1, "289": 1, "474": 1, "442": 1, "322": 1, "198": 1, "223": 1, "256": 2, "498": 2, "237": 1, "213": 2, "331": 1, "447": 1, "273": 1, "215": 1, "431": 1, "214": 1, "472": 1, "376": 1, "346": 1, "380": 1, "345": 1, "257": 1, "433": 1, "365": 2, "201": 1, "293": 1, "207": 1, "424": 1, "185": 1, "272": 1, "277": 1, "383": 1, "341": 1, "216": 1}`; at500: 12/100.

## Independent replay and tests

`{"status": "PASS", "images": 100, "history_states": 50100, "selected_output_metrics": 200, "scalar_checks": 924, "max_abs_error": 7.105427357601002e-15, "independent_renderer_max_abs": 2.384185791015625e-07, "ev_gamma_raws_exact": true, "all_states_finite_bounded": true, "all_earliest_selected_raws_exact": true, "preflight_before_normals": true, "freeze_before_metrics": true, "seconds": 10.683769380033482}`

Independent replay imports no main renderer/metric helper. Verifies all50100paired gain/lift states, starts, earliest minima, hashes, frozenEVgamma and bounds; reconstructs selected outputs from low/state with CPU-vs-GPU tolerance1e-6; recomputes200metrics plus deltas/totals/summaries/counts/distributions/verdict tolerance1e-10.

Local acceptedT047 + focusedT048 tests:4passed18.90s; server4passed2.89s; compilationPASS. Covers bit-exact T047 initialization, trainable parameter inventory, parameter-group learning rates, abstention, both-coordinate updates, immutableEVgamma, projected lift and earliest selection, both verdict boundaries.

## Failures / limitations / delivery

Initial deployment captured the local stage before file extraction finished; preflight202318 exited4 with missing test file before any tests/images/reference decode. After stage completion, exact same source redeployed202436. Retry preflight202511 encountered SSH255 before script/process creation; inspected no job/no log and recovered the identical command from saved metadata/template. Successful preflight then preceded the sole scientific run. No scientific restart, changed settings, or deviations. Compact archive download connection closed once; identical download retry succeeded and SHA matched. Existing NVML/protobuf warnings nonblocking.

Remaining mean Retinexformer training-exposed anchor minus T048: 0.413806418389 dB / 0.333397396125 SSIM. Scalar context only; no baseline outputs used/rerun.
Remaining mean SNR-Aware training-exposed anchor minus T048: 2.331349934703 dB / 0.367100582003 SSIM. Scalar context only; no baseline outputs used/rerun.

This is incremental reference-only affine capacity with EV/gamma frozen under a fixed probe, not deployable performance or a certified global ceiling. Stop and await research-lead review; no new operator/sweep/retraining.

Archives: `{"all_output_history_hashes_unchanged": true, "source_bindings_unchanged": true, "full": {"path": "/media/wenchang/F/wjq/TTIE/shared/t048a/T048A_full.tar", "bytes": 291389440, "sha256": "e0bd970585818e6c9baa2e9acfd1cea017841ad08b56e3c11533a314e6011bc8"}, "compact": {"path": "/home/wenchang/asdasdsad/wjq/TTIE/shared/t048a/T048A_compact.tar.gz", "bytes": 1270242, "sha256": "a24e29059ac3a3ce54eadf2657e98a89f1c526e1d1dbe95b3a7d52402d1d4fcc"}, "compact_backup": "/media/wenchang/F/wjq/TTIE/shared/t048a/T048A_compact.tar.gz"}`. Full images remain server/F; compact histories/starts/metrics/receipts and recovery persist under project research_log.
