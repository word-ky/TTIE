# T051-A DONE — smooth spatial exposure capacity

REFERENCE_ORACLE_ONLY. Zero deployable changes; zero official-test access.

not supported under fixed probe

| Metric | T050 mean | T051 mean | T050 median | T051 median |
|---|---:|---:|---:|---:|
| psnr | 21.874262727355 | 22.533267888905 | 22.647607082917 | 23.265271057274 |
| ssim | 0.505160717504 | 0.521992520232 | 0.540449382935 | 0.554501897674 |

Paired deltas: `{"delta_psnr": {"mean": 0.6590051615495335, "median": 0.45254539039826547}, "delta_ssim": {"mean": 0.016831802727691733, "median": 0.008217066854546262}}`. Sole gate: mean PSNR>=1.50dB AND median>=0.75dB AND mean SSIM>=0.

Win/equal/loss: `{"psnr": {"win": 100, "equal": 0, "loss": 0}, "ssim": {"win": 96, "equal": 0, "loss": 4}}`.

![Paired changes](T051A_result/paired_field.png)

## Fixed protocol and provenance

Tested scientific source cab61cdbfb5a4a6d45d22e5a6ebd914bda21f6c9; branch codex/T051A-spatial-exposure; accepted T050 head1160d11748cf5e025908879a509a492020491296, frozen b58696cd5874d92a72cebd03132fa0dd8e93fd75bfba638caf36642532b1dd5c, pairs2f3cf008677427fec0a1be0202e63272b1cd6d275957f295972161ec237716d1. Cohort b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b. Source68files T051A_source_binding.json; hashes embedded in preflight/config. All existing EV/gamma/common-gain/lift/q are immutable buffers. Onlyu[1,1,8,8] trainable; zero start, e=2tanh(u), then bilinear fullresolution align_corners=False. Multiply accepted clamped affine by2**e, clamp0/1, frozen8segment regional tone, existing hardgate; inactive output exactlyoriginal low. FreshAdam.05,500updates,one start,earlieststrictfullRGBMSEminimum0..500;float32renderer/float64loss;GPU1A6000,TF32off,seed7,threads1.

All100 low-only reconstructions completed 2026-09-15T19:20:44.558739+00:00, max error 0.0, bitexact True, normals0. First new normal 2026-09-15T19:21:25.582155+00:00; all100 frozen 2026-09-15T19:23:29.530231+00:00 SHAd66edca57c6cac1ffe15a428552c73e2e343363165a2808f16a1b01cce460667 before first metric decode 2026-09-15T19:24:35.079427+00:00.

Runs: preflight 20260916-032032-ttie-t051a-preflight, oracle 20260916-032119-ttie-t051a-oracle, evaluation/replay 20260916-032428-ttie-t051a-eval. Commands/envs/logs included. Preflight3.120058s; oracle122.976536s. Local4tests20.73s, server4tests2.35sPASS, compilePASS. Preflight/oracle exit0. Initial evaluation computed metrics then replay exited1 due to verifier float64 interpolation coordinates. Corrected independent replay032631 uses float32 coordinates and tanh to match the specified float32 renderer; tolerance remains1e-6. Original tested manifest retained in T051A_tested_source_binding.json; current source binding updated for the corrected verifier.

## Diagnostics and independent replay

Selected active-pixel EV distribution: `{"active_pixels": 23520000, "min": -1.9926578998565674, "max": 1.3630602359771729, "mean": -0.08553029074602378, "median": -0.0058067485224455595, "lower_hits": 0, "upper_hits": 0}`. Exactbound hits are floating equality at +/-2EV.

Best-step histogram: `{"421": 1, "221": 1, "500": 23, "278": 2, "234": 2, "496": 1, "210": 2, "453": 1, "291": 1, "229": 1, "218": 1, "206": 1, "342": 1, "205": 1, "255": 1, "295": 1, "385": 1, "169": 1, "362": 1, "224": 1, "208": 2, "198": 2, "199": 1, "207": 1, "259": 1, "233": 1, "294": 1, "158": 1, "203": 2, "239": 1, "176": 1, "192": 2, "170": 1, "425": 1, "345": 1, "298": 1, "285": 1, "200": 1, "260": 1, "249": 1, "194": 1, "303": 1, "318": 1, "244": 1, "272": 1, "223": 1, "225": 2, "262": 1, "201": 1, "389": 1, "211": 1, "443": 1, "322": 1, "227": 2, "230": 1, "350": 1, "228": 1, "293": 1, "287": 1, "276": 1, "195": 1, "268": 1, "361": 1, "404": 1, "336": 1, "159": 1, "310": 1, "261": 1, "467": 1}`; at500: 23/100.

`{"status": "PASS", "images": 100, "history_states": 50100, "selected_output_metrics": 200, "scalar_checks": 716, "max_abs_error": 3.552713678800501e-15, "independent_renderer_max_abs": 8.940696716308594e-07, "independent_interpolation_max_abs": 8.344650268554688e-07, "all_old_coordinates_exact": true, "inactive_outputs_exact": true, "preflight_before_normals": true, "freeze_before_metrics": true, "classification": "not supported under fixed probe", "seconds": 27.192875106993597}`

Independent NumPy half-pixel bilinear interpolation verifies selectedEVfields; frozen affine/NumPy tone reconstruct selectedoutputs within1e-6. Replaychecks all50100u histories finite/range/zero starts/earliestminima, all oldcoordinatesbitexact, originalinactiveoutputs, selectedhashes, 200metrics and summaries within1e-10. Exact savedfield is used for float32 renderer replay after independentinterpolation validation.

## Limits and delivery

This is reference-only marginal capacity on the frozen development cohort, not deployable improvement or held-out SOTA. No clean target or oracle quantity may enter deployable TTT. No sweep, new cohort, baseline rerun, or T052.

GitHub publication is blocked by account email verification403. Local source/evidence and remote artifacts are preserved; research-lead explicitly authorized using unmerged acceptedT050head. No scientific trajectory failures or settings changes. Initial replay precision mismatch was repaired only in the verifier, with all100 independent interpolation errors <=8.344650268554688e-7; no threshold relaxation.

Archives: `{"all_output_history_hashes_unchanged": true, "source_bindings_unchanged": true, "full": {"path": "/media/wenchang/F/wjq/TTIE/shared/t051a/T051A_full.tar", "bytes": 398745600, "sha256": "46ecf02da122e202d82e1c37d430710dfaec9c6dca31b88cdbc4aa16de8bda2a"}, "compact": {"path": "/home/wenchang/asdasdsad/wjq/TTIE/shared/t051a/T051A_compact.tar.gz", "bytes": 11122905, "sha256": "36cb77f91bcd6497108fcf93cde06ae664f060d04ed5c2fca8999cd08d574e9a"}, "compact_backup": "/media/wenchang/F/wjq/TTIE/shared/t051a/T051A_compact.tar.gz"}`. FullimagesremoteF; compacthistories/receipts/plots/projectlocal.
