# T057-A DONE — fixed chroma detail

REFERENCE_ORACLE_ONLY; zero deployable changes; zero official-test access.

not supported under fixed probe
Pre-optimization all100 max absolute RGB-channel sum: 1.7881393432617188e-07; tolerance 1e-06. Zero-mean applies before final clipping; clipping can change RGBmean. Independent value in replay receipt.

| Metric | T055 mean | T057 mean | T055 median | T057 median |
|---|---:|---:|---:|---:|
| psnr | 24.349792215572 | 24.435394370105 | 24.813333145194 | 24.888460331456 |
| ssim | 0.759127904630 | 0.767366704658 | 0.777913353031 | 0.783815849341 |

Paired deltas: `{"delta_psnr": {"mean": 0.08560215453276322, "median": 0.0472394206083564}, "delta_ssim": {"mean": 0.008238800027482271, "median": 0.006407254338146262}}`. Frozen gate meanPSNR>=.50dB AND median>=.25dB AND meanSSIM>=.020.

Win/equal/loss: `{"psnr": {"win": 100, "equal": 0, "loss": 0}, "ssim": {"win": 100, "equal": 0, "loss": 0}}`.

Tested source 0549d9724f9d8ffa855ac84426690a392bbe71ce; branchcodex/T057A-chroma-detail; basefe0d1a24ee35220f5cc15b522fbb0e17ed27c69b. Accepted T055 sourcee6a79d80740307fbea3b4391a9e9f9a718f14e89/evidence202de51182e50d5cc40e19b65992106487175aba, verifier49606b5d/evidence70041a56. T055freeze64ca0dd47ece431012fac01c9b0a319e24e42e62cff4683c587effe765abd6ae, pairsc1c1fc5830c50f9ef9ca331747ea0df0a9a3af2d8e7617fff5f7d33f03ad104c, config04172bb832de9a6530974d2430ee9bcd155a833f878142f5898056e76a0f1c94. Cohortb88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b; all93sourcebindings persisted in preflight/config and unchanged.

Fixed D=y0-B5(y0); B5=[1,4,6,4,1]/16 ordered separablefloat32 reflect. D_chroma=D-meanRGB(D), arithmeticRGBmean broadcast per pixel. Only raww_c[1,1,8,8] trainable, zero start,c_c=tanh(bilinear(w_c,align_corners=False)),RGBshared. Activeclamp(frozenT055+c_c*D_chroma),inactiveexactT055. FreshAdam.05 exactly500updates one start,states0..500 earlieststrictmin;float32renderer/float64fullRGBMSE. Alloldv/parameters/mask/constants frozen. NoT056/B9/D2 imported. GPU1 A6000. ZeroRGBsumchecked<=1e-6 beforeoptimization and independently.

All100 low-only starts bit-exact=True, maxerror0.0, normals0, persisted as start_000.pt...start_099.pt with individual hashes. Preflight completed2026-09-16T17:09:24.057888+00:00, SHA0dadb98787fddafae1ed92a200bb2dcbaf0e0cce0842dcae740596de95f96e2a; first reference2026-09-16T17:10:07.098903+00:00. All100 outputs frozen2026-09-16T17:12:05.539143+00:00 SHA3d233345258d8bbad5d66df55b584536b5e6b4f922c4f70ba5b51db032008742 before metricdecode2026-09-16T17:12:55.460653+00:00.

Runs/commands/logs: `{"release": "20260917-010837-ttie-t057a-chroma", "preflight_run": "20260917-010902-ttie-t057a-preflight", "oracle_run": "20260917-010959-ttie-t057a-oracle", "evaluation_run": "20260917-011247-ttie-t057a-eval"}`; exact receipts in result pack.

Tests: baseline3passed15.36s; localfocused+affected5passed16.99s; server5passed3.62s; compilePASS. Analytic RGBprojection/grayzeroresponse/zerochannelsum/idempotence tests; zeroidentity/onlyw_c/oldbufferfreeze/earliestminimum tests. Preflight9.486837593s; oracle113.626910948s; all jobs exit0.

Selected-step histogram: `{"445": 1, "497": 2, "177": 1, "167": 2, "190": 1, "253": 1, "488": 1, "188": 1, "230": 1, "182": 2, "251": 1, "265": 1, "466": 1, "209": 1, "279": 1, "500": 5, "356": 1, "181": 2, "247": 1, "171": 1, "278": 1, "163": 1, "199": 1, "155": 2, "235": 1, "173": 1, "201": 4, "179": 2, "218": 2, "212": 1, "261": 1, "145": 1, "147": 1, "192": 2, "303": 1, "481": 1, "418": 1, "348": 1, "289": 2, "388": 1, "286": 1, "475": 1, "232": 2, "223": 2, "337": 1, "196": 1, "248": 2, "496": 1, "254": 2, "302": 1, "222": 1, "183": 1, "226": 1, "225": 1, "185": 1, "239": 1, "221": 1, "214": 1, "154": 1, "156": 1, "217": 1, "210": 1, "394": 1, "422": 1, "191": 1, "400": 1, "470": 1, "202": 1, "203": 1, "291": 1, "194": 1, "381": 1, "256": 1, "162": 1, "471": 1, "208": 1, "281": 1, "172": 1, "358": 1, "165": 1}`; at500:5/100.

c_c distributions (6400controls;24000000fullfieldpixels including ignored inactive locations; exactfloat32±1 hits and fractions <=float32−.99 / >=float32+.99): `{"selected_controls": {"count": 6400, "min": -0.9995948076248169, "max": 0.9207387566566467, "mean": -0.1982167764002895, "median": -0.10218117386102676, "exact_hits": {"-1.0": 0, "1.0": 0}, "fraction_le_neg099": 0.00234375, "fraction_ge_pos099": 0.0}, "selected_field": {"count": 24000000, "min": -0.9995930194854736, "max": 0.9207387566566467, "mean": -0.20609009524752375, "median": -0.11660748720169067, "exact_hits": {"-1.0": 0, "1.0": 0}, "fraction_le_neg099": 0.0013418333333333333, "fraction_ge_pos099": 0.0}}`.

Independent replay: `{"zero_rgb_sum_max": 1.7881393432617188e-07, "zero_rgb_sum_tolerance": 1e-06, "status": "PASS", "images": 100, "history_states": 50100, "selected_output_metrics": 200, "scalar_checks": 724, "max_abs_error": 3.552713678800501e-15, "independent_renderer_max_abs": 0.0, "independent_basis_max_abs": 1.6530975699424744e-07, "independent_interpolation_max_abs": 1.1920928955078125e-07, "all_old_coordinates_exact": true, "inactive_outputs_exact": true, "preflight_before_normals": true, "freeze_before_metrics": true, "classification": "not supported under fixed probe", "seconds": 40.653197876992635}`. SciPyfloat64mirrorB5 and RGBmean subtraction independently reconstruct D_chroma; accepted CUDAFMA NumPy interpolation reconstructs c_c fromraw w_c. Saved exact c_c/D_chroma used only after independent checks to isolate renderer rounding. All50100histories/selection and200metrics, frozenoldcoords, inactivepixels, persistedstarts, ordering, distributions/nearboundaries/verdict verified. Tolerances1e-6 basis/interpolation/renderer and1e-10 scalars unchanged.

Failures/deviations: `["Local template rename touched hash literals; source inspection restored accepted hashes before commit/deploy; no scientificrun affected.", "First deployment010823 SSH255 before archive upload/science; same exactsource redeployed010837 successfully."]`. No scientificrepair/rerun/sweep. ExistingNVML/protobufwarningsnonblocking.

Archives: `{"all_output_history_start_hashes_unchanged": true, "source_bindings_unchanged": true, "full": {"path": "/media/wenchang/F/wjq/TTIE/shared/t057a/T057A_full.tar", "bytes": 2055731200, "sha256": "62b00468f73fc4531cf1afbfb393dae36b9863a7051ddc2361ae6606c0cd277f"}, "compact": {"path": "/home/wenchang/asdasdsad/wjq/TTIE/shared/t057a/T057A_compact.tar.gz", "bytes": 8280927, "sha256": "f16cb71251180936b00a4f759d4cb1eb669ebc3400c8a4482c20b16ae9fc32ee"}, "metadata": {"path": "/home/wenchang/asdasdsad/wjq/TTIE/shared/t057a/T057A_metadata.tar.gz", "bytes": 186005, "sha256": "90fc2f53955a2ed8425f436a64bfbf862abcfc5731958240deb2d261f89e2355"}, "compact_backup": "/media/wenchang/F/wjq/TTIE/shared/t057a/T057A_compact.tar.gz"}`. Full histories, persistedstartimages and selectedimages remain server/F; metadata/receipts/report projectlocal. Finite-budget reference diagnostic, not deployable/held-outSOTA or certifiedglobaloptimum. No T058/selfmerge/historycleanup.

Stop after reporting not supported under fixed probe; await research-lead review/new OPEN. No T058 or rerun.
