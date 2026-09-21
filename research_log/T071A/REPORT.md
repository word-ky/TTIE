# T071-A: OFFICIAL_LOLV2_REAL_TEST_RESULT_FROZEN

The exact T070-A frozen Final-Ours method was evaluated once on all100 official LOL-v2 Real test pairs. No images were excluded and no scientific settings were changed. The result is a held-out measurement, not a tuning criterion.

| Measurement | Exact result |
| --- | ---: |
| Mean PSNR (dB) | 18.53226686142791 |
| Median PSNR (dB) | 18.192788332030815 |
| Mean RGB-SSIM | 0.5734772617839705 |
| Images | 100 |
| Inference seconds total | 70.03712362400256 |
| Mean inference seconds/image | 0.7003712362400256 |

Selected steps min/median/max:17/22/24. Histogram:{"17": 5, "19": 5, "20": 7, "21": 10, "22": 29, "23": 34, "24": 10}. Inference timing uses CUDA synchronization and includes adaptation/features/selection; it excludes initial model loading, image decoding, output serialization and evaluation.

## Source and provenance

Evaluation runner source `cdabd083ad0853b702f409c4115f73f657d89918`; branch `codex/T071A-official-lolv2-real`; authorization `e60c9a24b1fa19f675261fca740feb2b33c6a744`. Scientific source remains exactly `aa4d920dff4b5b76751c24266e95ac9696d55d90`; Final-Ours manifest SHA256 `e7f129d931d27531e5f6a14cd72e8c94734c474c40c3c403959cc8f5764e5ab9`. No changes to the image-only API, gate/model/resources,27float32 Adam updates, loss, probability.5, rho.9857470621423519, lambda.875, preprocessing or environment bindings. Task-owned files: `research_log/T071A/{run.py,evaluate.py,verify.py,core.py,test_core.py,provenance.json,binding.json,PLAN.md,authorization.md,scientific_state.md,REPORT.md,evidence/**}`. Review this task directory, not the stacked historical aggregate as a merge recommendation.

Canonical author source: [flyywh/CVPR-2020-Semi-Low-Light](https://github.com/flyywh/CVPR-2020-Semi-Low-Light), whose LOL-v2 link targets [the original dataset ZIP](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view?usp=sharing). The existing T022A acquisition record binds1046491030bytes and archive SHA256 `9820d8b112438d94d1f5d4d25817eee618a8cf4bc63a65cd6b19f5a98c3faefa`, reverified in this run. Exact official paths: `LOL-v2/Real_captured/Test/Low/low00690.png` through `low00789.png`, paired with `Test/Normal/normal00690.png` through `normal00789.png`. Complete archive directory contains exactly these200test files; no resampling, train mixing or split substitution.

Before inference the full pair manifest was frozen with low SHA256 plus reference CRC32/size taken from ZIP central-directory metadata. This reconciles pre-run file hashes with the no-reference-payload rule: reference SHA256 values were deliberately deferred until postfreeze. Pre-run opaque archive hashing verifies provenance but does not open/decompress any reference member. This convention was declared in PLAN before inference; all postfreeze reference CRC32 values and hashes verify.

## Output-first boundary and metrics

Pair manifest frozen 2026-09-21T09:21:08.346711+00:00. Complete100-row output/decision table frozen 2026-09-21T09:22:41.280717+00:00, SHA256 `7b00345e2437b558f40e196eba4ce8114a50c219d24dd856c1da94c2596e32e3`. First reference-access marker 2026-09-21T09:22:51.922789+00:00. The inference ReadScope allowed only the100staged low files; the archive/reference paths were outside that scope. Exactly100low reads, inference `reference_reads=0`. Every output, selected state, k_FS/k_rho/selected step, full target-free decision/features/probabilities and trajectory hashes was saved before reference access.

Evaluation subsequently read100normal members once and added their SHA256/per-image metrics. The independent verification made its own100postfreeze reference reads to recompute the metrics; it did not rerun inference. All ordering, file/output/decision hashes, pair completeness and aggregates verified. `evidence/pairs_manifest.json`, `output_freeze.json`, `reference_open.json`, `reference_reads.json`, `per_image_metrics.json`, and per-image `outputs/*/decision.json` preserve the evidence.

Metrics use the existing project convention: native full RGB, no crop/resize/quantization; float64 arithmetic over float32[0,1] output/reference pixels, per-image PSNR=-10log10(RGBMSE). RGB-SSIM:11x11Gaussian sigma1.5, reflect padding, population covariance, channel/pixel mean, K1=.01,K2=.03, data_range1. The verifier uses torch float64 MSE and independent separable-convolution SSIM; maximum metric discrepancy `7.105427357601002e-15`. No quality pass threshold, baseline metric/run, cross-dataset access or outcome-driven rerun.

## Validation and execution

Core3passed1.34s; final local6passed16.69s; remote6passed1.61s. Tests cover complete/missing/duplicate official pairing, known-value metrics/aggregation and frozen inference API/selection/constants. Command: `python -B -m pytest research_log/T071A research_log/T070A -q`. Independent verifier PASS for100pairs, provenance, frozen scientific bindings, output-table hash, selections, information-boundary ordering and metrics.

Run `20260921-172057-ttie-t071a-official-real` on GPU1 A6000: start2026-09-21 17:21:03+08; metric result17:23:04+08; independent verification17:23:28+08; wrapper exit0 at17:23:29+08. Accounting:100optimizer runs,2700updates,globalmodel_fits0; independent optimizer_runs0. Release `/media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t071a-official-real`; outputs `/media/wenchang/F/wjq/TTIE/runs/T071A-official-real`. Sequence after tests: `research_log.T071A.run`, then `.evaluate`, then `.verify`, each with `--out` above. Full command/environment in run.sh and config; unchanged scientific environment is bound by the T070A manifest. Wrapper metadata contains a stale historical releaseId; explicit cd/source/config identify the actual release. Repository meta.json is LF-normalized; original bytes remain archived.

Pre-run issues only: initial git fetch timed out then succeeded unchanged; synthetic ZipInfo test fixture omitted CRC metadata and was corrected before tests passed. No scientific run failed or reran. Existing timm import deprecation warning was non-fatal.

Raw archive 8100536320bytes, SHA256 `8d66da5c1b3df96b5618fd772b9c25eb4b3f9fc5e5affd2ac99bb3320fc5c2ad`; recovery 637068bytes, SHA256 `7840f8a0349dc18d3980a20297dbd4b43c41a2fc02162546fa4c421b0174e9e2`. Compact recovery verified locally and stored on both server filesystems; complete raw output archive remains on F. Raw size is large because torch serialization retained the selected image view backing trajectory storage; scientific output hashes/results are unaffected and files were left unchanged. Exact paths in archives.json.

Next step: research-lead review of this frozen held-out result. Stop here; no tuning, baseline comparison or cross-dataset run is authorized by this result.
