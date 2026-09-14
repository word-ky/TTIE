# T027-A — Retinexformer low-only exporter smoke

**Verdict: exporter-ready.** Eight native training lows produce exactly equal
float32 outputs through the target-disabled official forward and TTIE exporter:
maximum and mean absolute differences are **0** for every image. All output
hashes match, arrays are finite with shape400x600x3. This is integration parity,
not a quality result or qualification of the untouched official test set.

## Binding and execution

Scientific source: `84a64ba1cb8a96ffe0ad1f27004535470bd60da6`.
Official repository: https://github.com/caiyuanhao1998/Retinexformer at
`1e9a0efce4b306b6701b824768370ff26066c32a`.
The MIT license and six exact official files are preserved. Their Git blobs
match the pinned official tree; source/checkpoint hashes also match after both
GPU paths complete. No network/weight edit, retraining or parameter selection.

The checkpoint is the official README-linked Drive folder's
`pretrained_weights/LOL_v2_real.pth`, file ID
`1tChRwTfqhs-A67QzG8a9Lrx7qKB3m89K`, 6478393 bytes, SHA256
`539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b`.
Folder listing, locator and source hashes are in `T027A_provenance.json` and
`T027A_official/drive_folder.html`. Strict state loading matches every key.

The original official CLI pairs target paths and computes metrics even with
GT_mean disabled. We execute its unchanged native forward statements via AST
extraction, stopping before GT_mean, and bypass its target/data/metric code.
`T027A_official_path.py` records this adapter explicitly; the exact executed
function is in `T027A_evidence/baseline/executed_official_forward.py`.
The official factory is replaced by direct import of its same network class,
official network_g settings, strict params loading and DataParallel. Neither
path calls the original target-reading CLI or its evaluator.

`ttie/retinex_exporter.py` accepts only `--low`, `--checkpoint`, `--config`,
`--out`. It decodes RGB float32/255, uses factor4 reflect-pad, model, unpad and
clamp[0,1], then saves HWC float32 `.npy` before PNG quantization. GT_mean and
self-ensemble are disabled, with no target-derived adjustment or metric.
The committed config contains only official architecture path and network
parameters. This smoke binds native LOL-v2 geometry, not larger/tiled inputs.

## Cohort and information boundary

The split hash is
`b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b`.
From689 official training low filenames, subtract100 frozen validation names,
then sort the remaining589 by SHA256(`TTIE-T027A-smoke|<relative-low-path>`).
The resulting ordered eight are:

| Train/Low image | Max absolute difference | Mean absolute difference | Float hash equal |
|---|---:|---:|---|
| low00149.png | 0 | 0 | yes |
| low00277.png | 0 | 0 | yes |
| low00582.png | 0 | 0 | yes |
| low00308.png | 0 | 0 | yes |
| low00197.png | 0 | 0 | yes |
| low00200.png | 0 | 0 | yes |
| low00504.png | 0 | 0 | yes |
| low00263.png | 0 | 0 | yes |

Manifest includes each low hash/selection hash and the exact eight ZIP members
read. ZIP metadata enumeration does not decode other images. Validation/test
overlaps are empty. Each real low is decoded once in each process; all other
image reads are denied. No normal, validation or official-test image was
decoded. Existing references elsewhere on the server are outside the image
allowlist, not claimed globally absent.

For a target mutation/withholding counterfactual, both actual pretrained
forward paths separately process one generated32x40 tensor under absent,
mutation-A, mutation-B and withheld canary-file conditions. The canaries are
synthetic bytes, not real paired normal images. All eight attempted Python/
OpenCV canary reads are denied. Both paths produce the same invariant hash
`cf4ae61efc5770097784f59c54f9e7fa5407816cbc584c1cbfbb78db86508546`.
This test does not repeat any real smoke low or claim actual-target mutation.

## Tests, runtime and evidence

- Local focused tests: **3 passed in9.91s** (pad/unpad/clamp, native zero-pad,
  low-only CLI surface).
- Remote focused tests: **3 passed in1.33s**.
- Official run `20260914-144320-ttie-t027a-baseline`:8 forwards, exit0.
- Exporter run `20260914-144902-ttie-t027a-exporter`:8 real forwards plus
  synthetic counterfactual, all parity/isolation assertions pass, exit0.
- Local artifact audit: deployed Python files match committed source bytes;
  six official Git blobs match; all eight output hashes/zero differences and
  split exclusion receipts agree. Extracted official forward has no target or
  metric identifiers. Postrun official source/archive/checkpoint hashes match.

| Path | Mean s/image | Median s/image | p95 s/image | Peak allocated bytes |
|---|---:|---:|---:|---:|
| Official adapter | 0.217949840 | 0.059286242 | 0.882795990 | 627518464 |
| TTIE exporter | 0.181741170 | 0.059274351 | 0.694143718 | 627518464 |

These eight-image timings include first-call initialization and measure CUDA
input ready through CPU float output; image decode/model load excluded. They
are not a benchmark speed claim. Environment: Python3.12.12, PyTorch2.4.0+cu121,
CUDA12.1, NVIDIA RTX A6000 physicalGPU1, NumPy1.26.4, OpenCV4.11.0;
package receipt records einops0.8.1 and pytest8.3.5.

Observed setup failures: remote Git clone GnuTLS-110 (used official pinned
codeload archive, verified blobs); local cv2 absent (decoder import moved into
CLI main for pure tensor tests); remote pytest absent (first exporter job
`20260914-144755-ttie-t027a-exporter` stopped before image forwards; installed
pytest8.3.5 and restarted). No real-low forward was repeated. Existing NVML
warning did not affect actual CUDA execution. Official trailing whitespace is
preserved to keep source hashes exact; authored code whitespace passes.

Run scripts/logs, manifests/config, per-image differences, hashes, times,
counterfactual and package receipts are in `T027A_evidence/`.
Full outputs, checkpoint, source and runs are backed up at
`/media/wenchang/F/wjq/TTIE/shared/t027a/T027A_execution.tar` (69570560 bytes),
SHA256 `a50a8ea36d82e950413dd1e18add9e40aa23283dc7fe718d59b011d777355a5a`.
Fetched compact bundle SHA256
`803cdd43c4378c0c481a420e030a9eca75150edd959f6bb890f70d30a23873fe`.
Full float tensors stay in the source run storage/full backup, not Git.

No PSNR/SSIM scoring, reference processing, dataset-wide inference, validation/
official-test execution, other baseline or Ours change. Stop pending review;
future benchmark use requires a new research-lead task.

exporter-ready
