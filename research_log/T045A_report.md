# T045-A: SNR-Aware fixed-validation benchmark

**SNR-Aware development benchmark complete.** On the fixed 100-image development validation split, SNR-Aware mean PSNR is **23.3963299207 dB**, RGB-SSIM **0.8237643949**. All provenance, output-freeze, target-isolation and independent-replay checks pass. There is no performance threshold.

**Training-exposed development anchor:** this split was carved from official LOL-v2 Real training pairs and the released SNR-Aware checkpoint is supervised on LOL-v2 Real. These numbers are not independent held-out SOTA evidence. Retinexformer below has the same training-exposure caveat.

## Fixed bindings and inference

Split SHA `b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b`, exactly the accepted T022/T033 100-image split. Canonical upstream `JIA-Lab-research/SNR-Aware-Low-Light-Enhance`, commit `1113144c82adc8bcc4a9ec27749ed75f196a4e4d`. All53 accepted source-file hashes are pinned in `T027B_source_binding.json` and checked before and after inference; no upstream change. Checkpoint `LOLv2_real.pth`, **156523164 bytes**, SHA `432d29d370e9f674f1b6763d371b4c24569a86d21f0fd45a5797226274d85781`. Unchanged T027-B exporter SHA `72a8bdec919d1c35056f549aca3c2bfcca7a1a2960999a7bfa0727434a766b61`, unchanged config SHA `fcb29f50538cfd09ec425c83d7f2072477b24f7c2ab4f23507c3e1b37016b3fb`. T045 binding contains68 source/checkpoint/exporter/config/local-code hashes; it does not open any prior metric file during inference.

Mode is exactly `ttie_native_pad16`: native RGB float32/255, official5x5 low-derived blur, right/bottom reflect padding of low and blurred feature to a multiple of16, low-derived SNR, direct network, native unpad and clamp[0,1], HWC float32 output. The accepted exporter main is called unchanged. No resize-based test4, ensemble, GT statistics, brightness matching or alternative configuration.

Source **62de789443ff951123868697f43071559a3f0386**, branch `codex/T045A-snr-benchmark`, PR https://github.com/word-ky/TTIE/pull/70. Release `20260915-161131-ttie-t045a-snr`; GPU1 inference run `20260915-161153-ttie-t045a-snr`. Environment: `CUDA_VISIBLE_DEVICES=1`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONPATH=$PWD`, OMP/MKL/OPENBLAS threads1. Seed7 and deterministic cuDNN settings are the unchanged exporter settings. Exact command and environment in `T045A_result/inference/run.sh`. Local4tests passed13.87s; server4tests passed1.32s; wrapper/evaluator/replay compilation passed. Tests cover accepted padding/unpad parity, low-derived SNR normalization, low-only CLI and freeze authorization binding.

## Output freeze and evaluation boundary

Exactly100 low paths decoded, in the split order. cv2 image reads are limited to those100 paths and PIL decoding is disabled during inference. All input hashes, output count400x600x3 shape, float32 dtype, finiteness, [0,1] range, tensor-value hashes and output-file hashes pass. Parameter hash before/after is identical to accepted T027-B: `11d3d667821719bd51e6e608c7876774b001643a28ed85193054bb45428190d4`. Zero learned-parameter updates; no Ours/adaptation/selection changes.

All100 outputs frozen **2026-09-15T08:12:11.237493+00:00**, freeze SHA `890649d354a3b9ff5bfde9bf08b1969c5cf4a6e5923404277d44aad71af48027`. External evaluation authorization **2026-09-15T08:12:50.485697+00:00**, strictly later. The freeze lists every input/output hash and decoded path; zero normal decodes before freeze. Evaluation run `20260915-161312-ttie-t045a-eval` opens only the same100 paired normals after the freeze and validates every normal hash and pairing. Both actual runs exit0. No official-test access, fresh cohort, retraining or reference-driven decision.

## Comparable development results

| Method | Mean PSNR | Median PSNR | Mean RGB-SSIM | Median RGB-SSIM |
|---|---:|---:|---:|---:|
| T026-A (accepted, not rerun) | 11.1208764 | 10.6073193 | 0.3737918 | 0.3679250 |
| T033 Retinexformer (accepted, not rerun) | 21.4787864 | 21.3989408 | 0.7900612 | 0.8160833 |
| T045 SNR-Aware | 23.3963299 | 23.6347420 | 0.8237644 | 0.8485793 |

Descriptive paired SNR-minus-T026 differences: PSNR mean **+12.2754535034 dB**, median **+13.0108553731 dB**; RGB-SSIM mean **+0.4499725698**, median **+0.4658904529**. Both metrics improve on99 images and decline on1 (these counts need not identify the same image). Exact accepted T026 CSV SHA `ad704fca9dbc393a0e30737eae60d212d41bb797e630e6599d39059380e5a397`; only read after the output freeze. No T026 or Retinexformer inference was rerun, and these comparisons create no gate.

## Metric convention and independent replay

Reuse accepted T033/T026 pixels and RGB-SSIM: native400x600 RGB, no crop/resize/Y conversion/output quantization or brightness matching. Normal uint8/255 is rounded tofloat32 before promotion tofloat64; output float32 is promoted tofloat64. PSNR=-10log10(mean squared RGB error), range1. RGB-SSIM uses11x11 Gaussian sigma1.5, K1=.01/K2=.03, population covariance, reflect half-sample symmetric borders, all pixels/RGB channels averaged. The main evaluator changes only baseline names/labels from T033; metric math is unchanged.

Main per-image alternative calculation max error **1.811883976188255e-12 <1e-11**. In addition, a separate replay reloads all100 frozen outputs and100 normals, checks hashes again, recomputes200 PSNR/SSIM values with torch float64 PSNR and explicit separable convolve1d SSIM (main gaussian_filter), and recomputes means/medians with standard-library statistics without importing the main aggregation path. It also recomputes paired T026 deltas. **612 scalar checks**, max discrepancy **7.105427357601002e-15 <=1e-10**, PASS; replay elapsed **7.126072s**. All output hashes remain unchanged.

## Runtime, artifacts and failures

Exporter forward-only runtime: total **2.317620300s**, mean **0.023176203s/image**, median **0.019710099s/image**, p95 **0.021557698s/image**. Peak allocated GPU memory **669319680 bytes**. These timings exclude model load, disk decode/write and evaluation; the first forward is included, so do not read them as end-to-end service latency.

No failures, retries or scientific deviations. The existing NVML initialization warning was nonblocking; CUDA inference completed and driver configuration was unchanged. One fixed SNR benchmark only; no search or model modification.

All100 output hashes and all68 source/checkpoint bindings were verified again during archival. Full inference+evaluation archive on F: **288419840 bytes**, SHA `35674c522a80efa12b2aaa68c6900b8848d09791884a0f82264f0b21a90827b0`, path `/media/wenchang/F/wjq/TTIE/shared/t045a/T045A_full.tar`. Compact evidence **44956 bytes**, SHA `a061a02cf66973fe7e151a7398d5bc41dec9cfbdb529de3a7e891833ca282b93`, verified at home/F shared/t045a and locally. Full float outputs remain on the server and F archive; compact provenance, per-image metrics, summaries, commands and logs are in this PR and project research_log. A Git-exact recovery bundle preserves final source/evidence and published main-mailbox entry.

Stop after this benchmark. The missing strong-baseline row is now available for the research lead to decide the next method investment. Do not infer independent held-out superiority, tune Ours, or launch another baseline from this task.

SNR-Aware development benchmark complete
