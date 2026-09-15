# T047-A DONE — additive lift marginal capacity

REFERENCE_ORACLE_ONLY. The fixed 500-update probe is complete. No deployable TTT changes, no official-test access.

additive-lift material marginal capacity not supported under fixed probe

## Metrics

| Quantity | T046 mean | T047 mean | T046 median | T047 median |
|---|---:|---:|---:|---:|
| psnr | 19.573227841755 | 20.055972040168 | 18.922069433101 | 19.294980253326 |
| ssim | 0.409367084615 | 0.414273525771 | 0.433142581723 | 0.442005243193 |

Paired PSNR mean +0.482744198412 dB / median +0.133182257884 dB. Paired SSIM mean +0.004906441156 / median +0.000590596695. Sole positive gate: mean PSNR >= +0.50 dB AND median >= +0.25 dB. SSIM and all other descriptions do not affect the verdict.

Win/equal/loss: `{"psnr": {"win": 100, "equal": 0, "loss": 0}, "ssim": {"win": 81, "equal": 0, "loss": 19}}`.

![Paired changes](T047A_result/paired_lift.png)

## Fixed protocol and provenance

Tested source **e0ae50ac57afe8b1039b25ff6de3dd5db16c6466**, branch `codex/T047A-additive-lift`, PR https://github.com/word-ky/TTIE/pull/72. Accepted T046 merge0081ad2307da0b6f192099a4df5a9b504350ed24, frozen outputs SHA2cdf5b8f104dc11445a44e6a68ebc8209192288e08e10bb60859cb95d5693057, accepted pairs SHA93e6b6dc1d463d1b088cb30fed5b995797c9f1d7dcb72d2544ac2751b7ef2652. Exact100 split b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b. All source bindings in T047A_source_binding.json/config/preflight; all selected lift/raw/output/file hashes in freeze.json.

All100 low-only baseline reconstructions completed 2026-09-15T11:17:44.254958+00:00; max absolute error 0.0; all bit-exact True; normal decodes0. First oracle decode 2026-09-15T11:18:44.524044+00:00; first reference decode 2026-09-15T11:18:44.796062+00:00. All100 selected outputs frozen 2026-09-15T11:24:24.458716+00:00 before metric aggregation.

Operator order: accepted exposure -> shifted gamma -> common gain -> additive lift -> existing identity-contrast arithmetic -> clamp -> hard-gate compositing. Immutable preclamp common-gain tensors are cached for efficiency; no legacy coordinate is trainable. One physical RGB-shared scalar per active quadrant, inactive lift exactly0. Every image starts at lift0, fresh Adam([lift],lr=0.01), projection[-0.20,+0.20], exactly500 updates, 501 retained states, earliest strict full-frame RGB-MSE minimum. Float32 renderer and float64 MSE. All legacy raw tensors remain bit-exact accepted T046 states.

Runs: preflight `20260915-191729-ttie-t047a-preflight`, oracle `20260915-191837-ttie-t047a-oracle`, evaluation `20260915-192506-ttie-t047a-eval`; release `20260915-191702-ttie-t047a-lift`. Commands/environments in saved run.sh/meta/train.log. GPU1 NVIDIA RTX A6000, TF32off, seed7, threads1. Preflight 3.345344s, oracle sum image wall times 339.161053s. All actual runs exit0.

## Descriptive lift and optimizer diagnostics

Selected active lift distribution: `{"active_coordinates": 392, "mean": 0.035180033791318364, "median": 0.018697068095207214, "min": -0.008508111350238323, "max": 0.20000000298023224, "lower_bound_hits": 0, "upper_bound_hits": 12}`. Bound hits are exact selected float32 physical bounds; denominator is active coordinates.

Best-step histogram: `{"296": 1, "425": 2, "171": 2, "272": 1, "252": 2, "491": 1, "342": 1, "451": 1, "211": 1, "240": 1, "222": 1, "247": 1, "293": 1, "218": 1, "154": 1, "178": 1, "23": 1, "273": 1, "161": 1, "237": 2, "236": 1, "447": 1, "303": 1, "156": 1, "320": 1, "260": 2, "210": 2, "456": 1, "275": 3, "325": 1, "274": 2, "163": 1, "203": 1, "151": 1, "344": 1, "168": 2, "232": 1, "152": 1, "160": 1, "308": 1, "204": 1, "246": 2, "197": 2, "346": 2, "249": 3, "262": 1, "226": 1, "330": 1, "200": 1, "413": 1, "185": 1, "490": 1, "276": 2, "316": 3, "449": 1, "220": 2, "483": 1, "195": 1, "255": 1, "248": 1, "196": 1, "351": 1, "184": 1, "150": 1, "432": 1, "277": 1, "192": 1, "493": 1, "202": 1, "338": 1, "250": 1, "254": 1, "422": 1, "206": 1, "466": 1, "312": 1, "234": 1, "24": 1, "221": 1, "368": 1, "341": 1}`. 0/100 at step500.

## Independent replay and tests

`{"status": "PASS", "images": 100, "history_states": 50100, "selected_output_metrics": 200, "scalar_checks": 716, "max_abs_error": 7.105427357601002e-15, "classification": "additive-lift material marginal capacity not supported under fixed probe", "independent_renderer_max_abs": 2.980232238769531e-07, "legacy_raws_exact": true, "all_states_finite_bounded": true, "all_earliest_selected_raws_exact": true, "preflight_before_normals": true, "freeze_before_metrics": true, "seconds": 11.000893750053365}`

Replay independently checks all50100 bounded lift states, zero starts, earliest selected identities, frozen legacy raws, independently renders selected outputs from low/state (CPU vs GPU tolerance1e-6), recomputes200 metrics and all summaries/counts/lift distributions/verdict (tolerance1e-10). No main metric/render helper imports.

Local accepted-baseline + focused tests:4passed26.16s. Server same4tests:4passed2.53s. Compilation passed. Tests cover exact zero-lift identity, preclamp placement on saturated values, gate abstention, lift-only parameterization, projection, frozen legacy, earliest minima and both verdict boundaries.

## Interpretation / delivery

Remaining mean Retinexformer training-exposed anchor minus T047: 1.422814364263 dB / 0.375787683283 SSIM. Scalar context only; baseline outputs were not used or rerun.
Remaining mean SNR-Aware training-exposed anchor minus T047: 3.340357880577 dB / 0.409490869161 SSIM. Scalar context only; baseline outputs were not used or rerun.

This is marginal capacity with frozen T046 coordinates under one fixed probe; it is not joint-operator capacity or a deployable gain. Stop and await research-lead review; no next operator, retraining or sweep.

Archives: `{"all_output_history_hashes_unchanged": true, "source_bindings_unchanged": true, "full": {"path": "/media/wenchang/F/wjq/TTIE/shared/t047a/T047A_full.tar", "bytes": 290529280, "sha256": "c276e32f762cb2555699f01578300e11949a82c0208374fbe91480ea3ed1cca0"}, "compact": {"path": "/home/wenchang/asdasdsad/wjq/TTIE/shared/t047a/T047A_compact.tar.gz", "bytes": 765695, "sha256": "8cdbc88eb813fcd3ccb851e7cdbc593b0995fb542f1d85ec0a54a9b98eef6cc7"}, "compact_backup": "/media/wenchang/F/wjq/TTIE/shared/t047a/T047A_compact.tar.gz"}`. Full selected images stay on server/F; compact histories, starts, metrics and receipts persist in project research_log.

Failures: none in implementation tests, scientific run or evaluation so far; existing NVML/protobuf warnings nonblocking. No scientific deviations.
