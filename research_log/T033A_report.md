# T033-A — DONE

The fixed target-free Retinexformer benchmark is complete on the exact original100-image development validation split. Retinexformer mean **21.478786404431204 dB /0.7900612090545553 RGB-SSIM**, versus accepted read-only T026-A **11.120876417349557 dB /0.3737918251517076**. Paired Retinexformer-minus-T026-A means are **+10.357909987081648 dB /+0.4162693839028476 SSIM**. This task has no promotion threshold; no model or setting was selected from these results.

## Source, checkpoint and inference

Engineering source `96daa548da625666d9aa52aac606e79cba217c18`, branch `codex/T033A-retinex-benchmark`, PR #58. `scripts/run_t033a.py` is an audit wrapper that invokes unchanged accepted `ttie.retinex_exporter.main`; there is no fork of its forward, checkpoint loading, preprocessing or postprocessing. The exporter accepts only low image paths, checkpoint, config and output. The wrapper accepts only low root, split, binding and output. It restricts cv2.imread to the exact100 low paths and denies all PIL image decoding during inference. No reference or metric value enters the exporter, and no original target-loading CLI is called.

Official source `caiyuanhao1998/Retinexformer` commit **1e9a0efce4b306b6701b824768370ff26066c32a**, with accepted six official source/config/license/README hashes and source archive bound before and after inference. Architecture SHA256 `1567c89d55285a3c1a4d4ca33f288b1fd98dc6b6bbdf971eda02d745f9087631`. Official `LOL_v2_real.pth`: **6,478,393bytes**, SHA256 **539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b**. Strict params loading and official network config (in/out3, n_feat40, stage1, blocks[1,2,2]) are unchanged. The MIT license remains preserved in the accepted official snapshot. No download/source substitution was needed.

Inference retains OpenCV BGR→RGB, float32 /255, native400×600, factor4 reflect-pad/unpad (no effective pad at this geometry), original network, output clamp[0,1], float32 HWC .npy before any PNG quantization, GT_mean=false and self-ensemble=false. Accepted seed7, cudnn deterministic/benchmark settings and DataParallel are unchanged. All100 outputs are finite, native400×600×3, float32 and in[0,1]. Each image has input SHA256, array and file output SHA256, shape, runtime and peak-memory receipt. All bound sources/checkpoint match after execution.

## Cohort and reference ordering

Frozen split SHA256 **b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b**, exactly the T022/T026 validation100, unchanged order and low hashes. No new split, substitution, Ours rerun or official test access. Binding manifest SHA256 `a9a61665602618adf20c5fb6b05af22da5906c293876fe27342c05a5da833d00`.

All100 outputs and their receipt were frozen at **2026-09-14T16:53:28.656322Z**, freeze SHA256 **30a4e132b6bb064a54f9346c7bcc08dd96a08101b9e42482b31732cde3ec2722**. The inference audit records exactly100 low decodes,0 normal decodes. Existing normals elsewhere on the server are outside the allowlist; they are not claimed globally absent. A separate evaluation authorization was persisted at **2026-09-14T16:53:54.194512Z**, after checking all output hashes, and binds the exact freeze hash. The evaluator checks that binding and all source cohort/output identities before opening references. First normal decode: **2026-09-14T16:54:31.464648Z**. Per-normal decode UTC records are preserved. All100 outputs remain unchanged through evaluation/backup.

## Metrics and paired comparison

Exactly the T026 convention: native full-frame RGB, no crop/resize/Y conversion or brightness matching; float32[0,1] image pixels promoted to float64 arithmetic. Critically the evaluator rounds the reference to float32 before the float64 calculation, exactly as accepted `evaluate_t026a.py`. PSNR=-10log10 RGB MSE. RGB-SSIM uses unchanged `ttie.ssim_transfer.rgb_ssim`: data_range1,11×11 Gaussian sigma1.5, population covariance, K1=.01/K2=.03, reflect boundary, RGB/spatial mean. No LPIPS or other metric.

Accepted T026-A metrics are read-only from run `20260914-113853-ttie-t026a-gamma05`, CSV SHA256 **ad704fca9dbc393a0e30737eae60d212d41bb797e630e6599d39059380e5a397**, verified before and after evaluation. All100 low/normal names and ordering match the frozen cohort; no Ours pixels or model are recomputed.

| Metric | Retinexformer mean | T026-A mean | Paired mean delta | Paired median delta | Retinexformer win/equal/loss |
|---|---:|---:|---:|---:|---:|
| PSNR dB | 21.478786404431204 | 11.120876417349557 | 10.357909987081648 | 10.706507585058114 | 99/0/1 |
| RGB-SSIM | 0.7900612090545553 | 0.3737918251517076 | 0.4162693839028476 | 0.4544842685020235 | 98/0/2 |

Retinexformer median PSNR21.398940825560743 and SSIM0.8160833485171493. All per-image values and paired deltas are in `T033A_result/audit/metrics.csv`; full aggregates are in `summary.json`.

This is a descriptive development-split quality anchor, not a held-out generalization comparison for the external checkpoint. The100 validation pairs were carved from the official training set; the accepted official Retinexformer YAML specifies `Real_captured/Train/Low` and `Train/Normal` for supervised training. Thus this split is not an independent holdout for that released training recipe. Inference remains strictly target-free, but training-exposure differences must be retained in interpretation. No SOTA or official-test conclusion is supported by this task.

## Runtime, verification and failures

A6000 physical GPU1, Torch2.4.0+cu121/CUDA12.1, Python3.12.12. Release `20260915-005241-ttie-t033a-retinex`; sole100-image inference run **20260915-005309-ttie-t033a-retinex** exit0; separate evaluation **20260915-005425-ttie-t033a-eval** exit0. Exact commands/logs are in the result's runs directory.

Mean/median/p95 inference time **0.06874116765276994 /0.05904646898852661 /0.060601643240079286 seconds/image**; peak allocated CUDA memory **627,518,464bytes**. Timing starts with CUDA input ready and ends with CPU float output, excludes decode/model load, and includes the first call. It is not an end-to-end application speed claim.

Baseline exporter tests3passed14.67s before changes. Final local focused tests `python -m pytest -q tests/test_retinex_exporter.py tests/test_t033a_binding.py`: **4passed9.18s**; server **4passed1.47s**. Tests cover pad/unpad/clamp, native zero-pad, target-free exporter CLI and rejection of a replaced freeze manifest. The actual100-image wrapper run establishes the low-only decode audit and one-to-one output mapping.

Independent PSNR dot-product and independently implemented SSIM agree with all100 output scores to maximum absolute error **1.4210854715202004e-14**. Local compact replay verifies exact100 cohort/pair identities, accepted Ours values and all paired deltas, hash-bound reference order, all metric aggregates and runtime aggregates: **maximum aggregation error0**. It uses exact Git-blob split/CSV bytes from the deployment stage to avoid Windows checkout line-ending transformations. Replay tool `scripts/check_t033a_saved.py` accepts result, split, ours CSV and output receipt paths. All deployed source bytes and compact archive hashes verify locally. Full image outputs are hash-verified on server; compact local evidence does not claim to contain those arrays.

No failed attempt, rerun, scientific deviation or unresolved blocker. The existing NVML warning did not prevent CUDA execution. No changes were made to Ours, official source/checkpoint, validation cohort, preprocessing or inference modes.

## Artifacts and next action

Full backup `/media/wenchang/F/wjq/TTIE/shared/t033a/T033A_execution.tar`,**300,431,360bytes**, SHA256 **e2c28081e7c4c87b7cfd861bc204b73be68b0bc82c38c4168d2f8d6b0a0520f7** includes100float outputs, bound source/checkpoint, runs and evaluation. Compact archive**41,302bytes**, SHA256 **70b2f738b9ffb5efa04f39093916d38d6acc8bf9532be38b190c759791578591**, retained on both server roots and extracted locally. Per-file provenance is in `T033A_backup.json`, and recovery-critical report/check receipts are mirrored to the project root research_log.

Return the measured gap and its development/training-exposure limitation to the research lead. Stop after this requested baseline; do not run SNR-Aware, tune Ours, open another split or access the official test. No self-merge.

benchmark complete
