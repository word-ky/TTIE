# T072-L — UHDLL_ANALYSIS_SPEC_SEALED

The outcome-independent UHD-LL analysis specification is sealed for all 150 canonical native 3840x2160 RGB images, three methods and 450 jobs. No target payload or target outcome was accessed. `inference_runs=0`, `reference_reads=0`, `real_metrics=0`.

Specification SHA256: `df9c5e4a6c2c5ee8cf812c4535938ecb80d5f6b34fbc659d2fc8302923f93391`.
Shared bootstrap index SHA256: `f3348c731c348b52eed32160d6b4b2b904101e59a542e1c5dc22850e812da904` (little-endian int64, C order).

## Exact metric provenance

Accepted T071-B scientific source `579c3691a80f5b7cadfd706a2fe6750876c53aa0` contains `research_log/T071B/evaluate.py`, which directly imports `research_log.T071A.core.metrics`. That function computes full-RGB float64 PSNR and delegates RGB-SSIM to `ttie.ssim_transfer.rgb_ssim`. All three Git blob identities are sealed in `analysis_spec.json`; their SHA256 values independently match the accepted T071-B `binding.json`. `verification.json` records the full chain. The independent T071-B replay implementation is not the primary metric implementation and is not substituted.

The reference conversion is uint8 divided by 255 in float64, rounded to float32, then promoted to float64; frozen float32 outputs are promoted to float64. No resizing, cropping, brightness matching or output quantization is introduced. RGB-SSIM uses the exact original scipy gaussian_filter call, sigma=(1.5,1.5,0), truncate=3.5, reflect borders, K1=.01/K2=.03, population covariance and mean over all RGB pixels. The older cluster_bootstrap helper in that module is not called: its seed7 recipe is unrelated to the newly specified paired-image bootstrap.

## Fixed analysis and information boundary

Per method: mean PSNR, median PSNR, mean RGB-SSIM. Against each baseline: per-image Ours-minus-baseline PSNR and RGB-SSIM differences, mean/median delta and strict-positive win fraction with denominator150. For the four mean paired deltas, report 95% percentile intervals from one shared PCG64 seed20260922 index matrix with 10000 rows of 150 replacement draws; quantiles .025/.975 use linear interpolation. No additional significance or superiority threshold is introduced.

All samples and all methods are mandatory. Missing, duplicate or nonfinite outputs, geometry/binding mismatches, incomplete coverage or reference mapping fail the entire future evaluation. Nonfinite metrics also fail without removing rows. No subgroup or exclusion is allowed. The accepted T072-E-R1 verifier commit/blob is pinned, and reference access is permitted only after independent verification of all450 frozen hashed outputs and bindings. This task seals the contract; execution of the future benchmark remains a separate task.

## Synthetic validation and reproduction

Run from this worktree:

```
python research_log/T072L/verify.py
python research_log/T072L/test_spec.py
```

The verifier imports no producer, model, metric code or target loader. It checks the entire contract against an independent literal digest, rehashes accepted source blobs against the actual runtime bindings, recomputes the accepted dispatch root/Cartesian coverage from committed metadata only, and regenerates the index stream row-by-row independently of the producer's matrix call. Git history containing the pinned commits and NumPy are required.

Synthetic tests include 19 malicious contract mutations with recomputed accompanying seal digests: dispatch/method/metric identity changes, bootstrap seed/count/sample-size/shared-stream changes, exclusions, permissive failure handling, early reference access, incomplete coverage, reversed comparisons, altered ties and extra metrics. Additional synthetic arrays verify paired bootstrap translation/scaling and deterministic reproduction. This does not claim execution-time validation of real outputs or references.

## Handoff

Latest authorization read from origin/main `9ece5675` on 2026-09-23. Work reused the existing clean `codex/T072L-analysis-spec` worktree. Prior monitoring only inspected the stale outer checkout and missed remote instructions; future checks must inspect origin/main's inbox. No further T072-K work or GPU polling was performed. `coordination/PROJECT_STATE.md` remains unchanged. Stop after one completion mailbox entry.
