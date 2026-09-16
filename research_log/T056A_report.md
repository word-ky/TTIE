# T056-A DONE — fixed coarser detail band

REFERENCE_ORACLE_ONLY; zero deployable changes; zero official-test access.

not supported under fixed probe

| Metric | T055 mean | T056 mean | T055 median | T056 median |
|---|---:|---:|---:|---:|
| psnr | 24.349792215572 | 24.470167646517 | 24.813333145194 | 24.987490380312 |
| ssim | 0.759127904630 | 0.769874733634 | 0.777913353031 | 0.792686632527 |

Paired deltas: `{"delta_psnr": {"mean": 0.1203754309445748, "median": 0.10197221026392533}, "delta_ssim": {"mean": 0.010746829003759981, "median": 0.004411192607561454}}`. Frozen gate meanPSNR>=.50dB AND median>=.25dB AND meanSSIM>=.020.

Win/equal/loss: `{"psnr": {"win": 100, "equal": 0, "loss": 0}, "ssim": {"win": 81, "equal": 0, "loss": 19}}`.

Tested source 5cb1a4f2048db1a34fdd333adbcb4688b3aa9687; branchcodex/T056A-second-detail; basefe0d1a24ee35220f5cc15b522fbb0e17ed27c69b. Accepted T055 sourcee6a79d80740307fbea3b4391a9e9f9a718f14e89/evidence202de51182e50d5cc40e19b65992106487175aba, verifier49606b5d/evidence70041a56. T055freeze64ca0dd47ece431012fac01c9b0a319e24e42e62cff4683c587effe765abd6ae, pairsc1c1fc5830c50f9ef9ca331747ea0df0a9a3af2d8e7617fff5f7d33f03ad104c, config04172bb832de9a6530974d2430ee9bcd155a833f878142f5898056e76a0f1c94. Cohortb88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b; all93sourcebindings persisted in preflight/config and unchanged.

Fixed D2=B5(y0)-B9(y0) from frozen pre-detail T052 y0. B5=[1,4,6,4,1]/16, B9=[1,8,28,56,70,56,28,8,1]/256; horizontal then vertical ordered float32 sums with reflect padding2/4. Only raww[1,1,8,8] trainable, zero start, c2=tanh(bilinear(w,align_corners=False)); RGB shared. Active output clamp(frozenT055+c2*D2), inactive exactT055. FreshAdam.05, exactly500updates, one start, states0..500 earlieststrictmin;float32renderer/float64fullRGBMSE. Oldv/EV/gamma/gain/lift/tone/u/b/mask/constants and all old buffers unchanged. GPU1 A6000.

All100 low-only starts bit-exact=True, maxerror0.0, normals0, persisted as start_000.pt...start_099.pt with individual hashes. Preflight completed2026-09-16T16:38:59.602602+00:00, SHAedcb620eb57047e854c54cbe499cd9dbd3377549ce6e11ae5db3f6186ed7b66a; first reference2026-09-16T16:39:35.560270+00:00. All100 outputs frozen2026-09-16T16:41:32.724635+00:00 SHA1f4783d496ce1a4914cb684b1346791788f962d71e78e2e2f6d467330967d721 before metricdecode2026-09-16T16:42:16.016471+00:00.

Runs/commands/logs: `{"release": "20260917-003813-ttie-t056a-band", "preflight_run": "20260917-003838-ttie-t056a-preflight", "oracle_run": "20260917-003927-ttie-t056a-oracle", "evaluation_run": "20260917-004209-ttie-t056a-eval"}`; exact receipts in result pack.

Tests: baseline3passed7.54s; localfocused+affected5passed6.59s; server5passed3.28s; compilePASS. Analytic9tapimpulse/constant/reflectramp and band-difference tests; zeroidentity/onlyw/oldbufferfreeze/earliestminimum tests. Preflight9.839182519s; oracle112.577301731s; all jobs exit0.

Selected-step histogram: `{"500": 100}`; at500:100/100.

c2 distributions (6400controls;24000000fullfieldpixels including ignored inactive locations; exactfloat32±1 hits and fractions <=float32−.99 / >=float32+.99): `{"selected_controls": {"count": 6400, "min": -1.0, "max": 1.0, "mean": 0.04670615412520988, "median": 0.17575129121541977, "exact_hits": {"-1.0": 149, "1.0": 97}, "fraction_le_neg099": 0.25859375, "fraction_ge_pos099": 0.25234375}, "selected_field": {"count": 24000000, "min": -1.0, "max": 1.0, "mean": 0.07709090185313759, "median": 0.31836096942424774, "exact_hits": {"-1.0": 99831, "1.0": 73832}, "fraction_le_neg099": 0.2290875, "fraction_ge_pos099": 0.19779666666666668}}`.

Independent replay: `{"status": "PASS", "images": 100, "history_states": 50100, "selected_output_metrics": 200, "scalar_checks": 724, "max_abs_error": 3.552713678800501e-15, "independent_renderer_max_abs": 0.0, "independent_basis_max_abs": 3.219911377527751e-07, "independent_interpolation_max_abs": 1.1920928955078125e-07, "all_old_coordinates_exact": true, "inactive_outputs_exact": true, "preflight_before_normals": true, "freeze_before_metrics": true, "classification": "not supported under fixed probe", "seconds": 40.35042778297793}`. SciPyfloat64mirrorB5/B9 independently reconstruct D2; accepted CUDAFMA NumPy interpolation reconstructs c2 fromraw w. Saved exact c2/D2 used only after independent checks to isolate renderer rounding. All50100histories/selection and200metrics, frozenoldcoords, inactivepixels, persistedstarts, ordering, distributions/nearboundaries/verdict verified. Tolerances1e-6 basis/interpolation/renderer and1e-10 scalars unchanged.

Failures/deviations: `[]`. No scientificrepair/rerun/sweep. ExistingNVML/protobufwarningsnonblocking.

Archives: `{"all_output_history_start_hashes_unchanged": true, "source_bindings_unchanged": true, "full": {"path": "/media/wenchang/F/wjq/TTIE/shared/t056a/T056A_full.tar", "bytes": 2055720960, "sha256": "895c643cfbc6b9efb3c81a8b627607f69c4777c9b99cb3e039fe46683bd4c7bd"}, "compact": {"path": "/home/wenchang/asdasdsad/wjq/TTIE/shared/t056a/T056A_compact.tar.gz", "bytes": 12197181, "sha256": "7f72e0980eb35d4f89a562a74249d08417d1fabe3e57640bcf1666b3219cef17"}, "metadata": {"path": "/home/wenchang/asdasdsad/wjq/TTIE/shared/t056a/T056A_metadata.tar.gz", "bytes": 184066, "sha256": "79820b89f9fcdab1b700a416f4970fde9ba51670063f152ea91b6caa0ed40176"}, "compact_backup": "/media/wenchang/F/wjq/TTIE/shared/t056a/T056A_compact.tar.gz"}`. Full histories, persistedstartimages and selectedimages remain server/F; metadata/receipts/report projectlocal. Finite-budget reference diagnostic, not deployable/held-outSOTA or certifiedglobaloptimum. No T057/selfmerge/historycleanup.

Stop after reporting not supported under fixed probe; await research-lead review/new OPEN. No T057 or rerun.
