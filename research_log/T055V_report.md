# T055-V DONE — T055 replay verified

REFERENCE_ORACLE_ONLY; zero deployable changes; zero official-test access.

Frozen scientific source `e6a79d80740307fbea3b4391a9e9f9a718f14e89`, evidence `202de51182e50d5cc40e19b65992106487175aba`; accepted T054 source `f4baa579e4441a6edf5ec818ddca87f28f1b7e5d` / evidence `a4d37006cbce1814290fc279bc0b6ae72e0dd952`. Verifier-only source `49606b5d654a5e69b68b9895e81759c0f01cd175`, authorized by main `c78f97792756a1ef69a93e0197be8fca751126cc`. Same cohort `b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b`.

## Arithmetic diagnosis

At image index12 / pixel(y=200,x=40), four raw controls are [[9.8465681076,.0678445920],[-9.6322908401,-.7223473787]]. Source coordinates [3.50999999046,.04000001401], fractional weights [.50999999046,.04000001401]. Original independently interpolated raw value −.0975489616394 differs from CUDA −.0975500643253. Original c=−.0972407162189 versus saved/CUDA c=−.0972418114543 gives1.0952353477478027e−6 error. Reconstructing CUDA bilinear+tanh from saved rawv alone matches the entire saved c field exactly; savedc is only the comparison target.

Cause: separately rounded NumPy weighted multiply/add differs from contracted CUDA multiply-add during cancellation. The correction globally emulates `fma(a,b,c)` by float64 product-plus-add followed by float32 rounding, in both horizontal sums and the vertical sum. The fixed weighted expression is documented in [PyTorch 2.4 CUDA bilinear source](https://github.com/pytorch/pytorch/blob/v2.4.0/aten/src/ATen/native/cuda/UpSampleBilinear2d.cu#L65-L70). No per-image/value/shape/outcome special case. `verifier.diff` shows only this arithmetic change plus a separate receipt output path. Corrected raw field matches CUDA exactly on index12; corrected coefficient maxerror1.1920928955078125e−7. Nine synthetic random/alternating/constant grids across three output sizes independently give rawerror0 and coefficienterror<=1.1920928955078125e−7.

## Full replay and immutable evidence

Replay run `20260917-000911-ttie-t055v-replay`, exit0; {"status": "PASS", "images": 100, "history_states": 100100, "selected_output_metrics": 200, "scalar_checks": 724, "max_abs_error": 3.552713678800501e-15, "independent_renderer_max_abs": 0.0, "independent_basis_max_abs": 1.771841198205948e-07, "independent_interpolation_max_abs": 1.1920928955078125e-07, "all_old_coordinates_exact": true, "inactive_outputs_exact": true, "preflight_before_normals": true, "freeze_before_metrics": true, "classification": "material local-detail underconvergence not supported under fixed extension", "seconds": 37.450051062973216}.

Before/after proof: {"status": "PASS", "files": 299, "source_files": 86, "all_scientific_files_unchanged": true, "baseline_sha256": "514172ee1dd73e2635b5a045b40bebf63fd1dafe6ced7c9b6eaf336a325e2709", "original_verifier_sha256": "95ee4a247c4830bcc5cfae704de5c3dd5dc49cb31cad54d44b2d059be3aa6e64", "new_verifier_sha256": "ac5a055aab5f8bf9380fd3587841589e361d52d57dfeed3f9ecb515e7396cd69", "optimizer_calls": 0, "metric_files_regenerated": false}. All299 frozen scientific/source/preflight files unchanged, including86 source bindings,100 histories,100 selected outputs, pairs/summary/config/freeze. Frozen JSON also matches the locally retained original evidence bytes. Old verifier remains untouched at SHA95ee4a247c4830bcc5cfae704de5c3dd5dc49cb31cad54d44b2d059be3aa6e64; new verifier SHAac5a055aab5f8bf9380fd3587841589e361d52d57dfeed3f9ecb515e7396cd69. The new replay receipt is outside the frozen audit. No optimizer calls, new states/outputs, or regenerated metric files. Original1e−6 basis/interpolation/renderer and1e−10 scalar tolerances and all scientific settings/gates unchanged.

## Adjudication

**T055 replay verified.** Final original verdict: **material local-detail underconvergence not supported under fixed extension**. Unchanged T055 meanPSNR24.349792215571938 / SSIM.7591279046304531; paired meanPSNR+.004305474682432848, median+.002918943444738531, meanSSIM+.0013706655157820719 miss .25/.10/.010 gate. Close pure step/LR-budget rescue for this one-scale family; this is not a global-convergence certificate. No follow-on experiment.

Implementation failures: first copy-generation replacement missed CRLF, so initial synthetic tests still used original arithmetic; corrected verifier-file generation before full replay, then9casesPASS. No failed full replay in T055-V. Evidence SCP initiallySSH255, read-only retry succeeded. Original T055-A failure and PARTIAL report preserved.

Archives: {"path": "/home/wenchang/asdasdsad/wjq/TTIE/shared/T055V_evidence.tar.gz", "backup": "/media/wenchang/F/wjq/TTIE/shared/t055v/T055V_evidence.tar.gz", "bytes": 21677, "sha256": "b0ffa400502fa5b677b120983a73108cb81e5e9f3cbab1de8c943a3b73ac503d"}. Diagnosis, synthetic receipts, global diff,299file manifest, replay receipt and logs included. Stop awaiting research-lead review; no T056/history cleanup/selfmerge.
