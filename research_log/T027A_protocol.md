# T027-A fixed exporter smoke

Only integration/provenance; no quality scoring. Research inbox 9cb2fa95,
accepted T026-B merge 7c9445c7d25469f9ded9b05556088619ea5e3f9f.
T026-A remains promoted. Stop after eight-image parity verdict.

Official donor: caiyuanhao1998/Retinexformer at
1e9a0efce4b306b6701b824768370ff26066c32a, MIT (Yuanhao Cai, 2023).
Exact official Drive LOL_v2_real.pth: 6478393 bytes, SHA256
539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b.
Six relevant source blobs match the pinned Git tree. A remote Git clone failed
with GnuTLS -110; official commit-addressed codeload archive replaced transport,
not source version. Remote environment only added missing einops 0.8.1.

## Baseline and reuse

Official test script unconditionally loads targets for metrics even when
GT_mean is false. T027A_official_path extracts its entire native forward AST
from `(b,c,h,w)=input_.shape` through the final clamp/CPU/HWC conversion,
ending before GT_mean. Those statements are executed unchanged. The dataloader,
target paths, metrics and image quantization are not executed. The model factory
is replaced by direct import of the same unmodified RetinexFormer class,
official YAML network_g, strict checkpoint params load, CUDA and DataParallel.
The resulting extracted source is persisted. This is a target-disabled official
forward adapter, not an execution of the original target-reading CLI.

Baseline release 20260914-144301-ttie-t027a-baseline, run
20260914-144320-ttie-t027a-baseline: PASS eight finite native400x600x3float outputs,
eight allowlisted image reads, exit0. No real low rerun.

Reuse map: official network/config/checkpoint unchanged; official cv2 BGR-to-RGB
float32/255 decode unchanged; native factor4 reflect-pad, forward, unpad, clamp
preserved in the thin TTIE exporter. No optimizer, target statistics or metric.
Exporter arguments contain only lows/checkpoint/config/out. Its compact config
contains only architecture source path and official network parameters.

## Frozen execution and acceptance

- Exactly8 of589 nonvalidation official training lows, ascending SHA256 of
  `TTIE-T027A-smoke|<Train/Low/relative-name>`; archive metadata enumerated,
  only selected low members extracted. No normal member read; validation/test
  excluded before extraction. Preserve manifest, hashes and selected members.
- A6000 physicalGPU1, float32, batch1, no autocast/selfensemble/GT_mean.
  Native600x400 requires zero effective padding.
- Each real low once in official adapter and once in exporter. Compare saved
  float32HWC arrays before PNG quantization, max absolute error <=1e-6, all
  finite and native geometry. No tolerance relaxation.
- Both real forward processes deny image reads outside exact8low allowlist.
  Existing reference directories elsewhere are not claimed globally absent.
- Separate generated32x40 tensor with actual pretrained network tests both
  paths under absent/mutated/withheld inaccessible canary files; output hashes
  must be invariant. These are synthetic canaries, not paired normal targets;
  no real smoke image is rerun for this counterfactual.
- Runtime includes first-call CUDA initialization, from CUDA input through CPU
  float output; excludes model load and image decode. Report mean/median/p95,
  peak allocated memory, and every image time. Do not infer benchmark speed.
- No quality metrics, validation/test decoding, retraining or other baseline.

Increment: baseline green -> exporter padding/CLI tests -> real GPU parity and
synthetic target counterfactual -> report/recovery. Local tests initially could
not import cv2 (not installed locally); moving image decoder import into main
allows pure tensor/CLI tests without installing unused local dependencies.
