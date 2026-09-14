# T027-B — SNR-Aware native-pad16 exporter smoke

**Verdict: exporter-ready.** Each of eight smoke lows produces an exactly equal
HWCfloat32 output through the independently wired native-pad16 adapter and TTIE
exporter: maximum/mean absolute difference0, matching float hashes, finite
400x600x3geometry and values in[0,1]. No quality scoring or test qualification.

This is the predeclared **ttie_native_pad16 protocol adaptation**, not exact
reproduction of the official resize-based `test4` path or published results.

## Provenance and implementation

Scientific source: `d6ff3a2157f8d33d7ac508488f78ad1d6f03f0dd`.
Canonical repository https://github.com/JIA-Lab-research/SNR-Aware-Low-Light-Enhance,
commit `1113144c82adc8bcc4a9ec27749ed75f196a4e4d`. The53 relevant Python/YAML/
README files match pinned Git blobs and SHA256 values, also unchanged after both
runs. No project-level license grant was found: **no third-party source,
checkpoint or archive bytes are committed in this task**. TTIE Git contains only
authored adapters/exporter/tests and evidence hashes/locators/receipts. External
source and weights remain in project-local storage and remote backups.

The official README designates Drive file
`1g3NKmhz7WFLCm3t9qitqJqb_J7V4nzdb`, a1017280510-byte
`pretrain_model_cvpr_snr.zip`. Its exact member `pretrain_model/LOLv2_real.pth`
is156523164bytes, CRC32`1209377b`, SHA256
`432d29d370e9f674f1b6763d371b4c24569a86d21f0fd45a5797226274d85781`.
The server cannot reach Google Drive. The member was fetched locally using
official HTTP206 byte ranges, inflated with size/CRC verification, then sent by
workflow SCP. No mirror or alternate checkpoint. Full ZIP was not downloaded;
compressed-member range/hash and ZIP tail range/hash are bound in
`T027B_checkpoint_binding.json`. Strict loading matches every network key.

The independent adapter uses external official `networks.define_G`,
`data.util.read_img_seq` and `Video_base_model4_m.VideoBaseModel.test`. The
official dataset's four low-only blur statements are extracted/executed at
runtime without persisting third-party source. It supplies only netG, var_L and
nf; it does not instantiate the paired dataset or call target-reading feed_data,
visualization, metrics or test4. The missing center option maps to official None
default. The TTIE exporter independently instantiates the external architecture,
loads the strict state, computes low-only SNR and calls the network directly;
it never calls the independent adapter or official test method.

Fixed sequence: decode native RGBfloat32/255; native5x5 OpenCV blur with default
REFLECT_101 border; reflection-pad both low and blur right/bottom to multiple16;
compute SNR from the padded low/feature; direct network; exact unpad; clamp[0,1].
Native400x600 gets left/right/top/bottom=[0,8,0,0], network400x608, output400x600.
No input resizing. Official network-internal operations remain unchanged.
SNR luminance uses .299/.587/.114 RGB weights, absolute low-minus-blur noise,
signal/(noise+.0001), per-image maximum normalization with+.0001 and clamp[0,1].
`T027B_evidence/export_config.json` freezes architecture parameters, blur, pad,
SNR and output conventions. Exporter CLI: only low/checkpoint/config/source/out.

## Eight-image evidence

Subtract the frozen100 validation filenames from689 training lows, then select
eight of589 by SHA256(`TTIE-T027B-smoke|<relative-low-path>`) ascending. Split SHA
`b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b`.
ZIP metadata enumerated; only selected low members read. Manifest includes each
low hash/selection hash, selected member names and empty validation/test overlap.

| Train/Low image | Max absolute difference | Mean absolute difference | Float hash equal |
|---|---:|---:|---|
| low00585.png | 0 | 0 | yes |
| low00318.png | 0 | 0 | yes |
| low00278.png | 0 | 0 | yes |
| low00244.png | 0 | 0 | yes |
| low00539.png | 0 | 0 | yes |
| low00166.png | 0 | 0 | yes |
| low00277.png | 0 | 0 | yes |
| low00016.png | 0 | 0 | yes |

Each real low was decoded/forwarded once per path, with exactly eight allowed
image opens in each process and no repeated real smoke forwards. Paired normals,
frozen validation and official test were not decoded. Both processes deny image
reads outside the exact smoke allowlist; existing reference files elsewhere on
the server are not claimed globally absent.

Separate generated32x40 tensor uses both actual pretrained paths under absent,
mutation-A, mutation-B and withheld target-canary conditions. All eight attempted
Python/OpenCV canary reads are denied. Both paths keep invariant output hash
`c56e07432e2f0c90302c9687d7d709d022a63cc41a8b8237e83e27cf7a31a9e7`,
max difference0. Canaries are synthetic bytes, not paired normal images.
Full network state hashes before/after both real paths and synthetic check are
identical: `11d3d667821719bd51e6e608c7876774b001643a28ed85193054bb45428190d4`.

## Validation, environment and runtime

- Independent baseline run `20260914-155420-ttie-t027b-baseline`:8forwards, exit0.
- Exporter run `20260914-155900-ttie-t027b-exporter`:8real forwards plus separate
  synthetic canary checks, all assertions pass, exit0.
- Local focused tests:3passed16.99s. Remote:3passed1.33s. Tests cover native
  right8 reflection/exact unpad, SNR calculation with zero-dark case, CLI surface.
- Local audit verifies deployed-source byte hashes,53 official postrun source
  hashes, checkpoint identity, split exclusion, all8zero-error/equal-hash outputs
  and parameter invariance. Exact commands and logs are committed as receipts.

| Path | Mean s/image | Median s/image | p95 s/image | Peak allocated bytes |
|---|---:|---:|---:|---:|
| Independent native-pad16 adapter | 0.059701769 | 0.021414265 | 0.220400762 | 678074880 |
| TTIE exporter | 0.061464286 | 0.020448629 | 0.233355486 | 669319680 |

Timing covers decoded CPU tensor through native blur, CUDA transfer/pad/SNR/
network and float CPU output; includes cold first-call initialization, excludes
image decoding/model loading. This is a small integration smoke, not a benchmark
speed claim or comparison against another method. A6000 physicalGPU1;
Python3.12.12, PyTorch2.4.0+cu121, CUDA12.1, NumPy1.26.4, OpenCV4.11.0.86,
PyYAML6.0.3, pytest8.3.5. Existing NVML warning did not prevent actual CUDA runs.

Observed setup issue: the venv has the gdown module without a bin/gdown launcher;
`python -m gdown` then exposed server Google Drive network-unreachable. Local
official-member HTTP206 extraction solved transport. No method/math change,
parity tolerance relaxation, alternate mode, or real-image rerun.

## Delivery and next step

Full external source/checkpoint/native float outputs/releases/runs are in
`/media/wenchang/F/wjq/TTIE/shared/t027b/T027B_execution.tar`,208670720bytes,
SHA256`7adb221b4e4715f4c5dabe9839ebc3c0fe394ba8cd3bf05bbdc6dd884e31bade`.
Fetched compact bundle SHA256
`6faef84d5b173b99f9410b65ef82c3a6fb0846101f0bb601aa00233ad6774fe5`.
Only compact TTIE evidence is committed under `T027B_evidence/`.

No PSNR/SSIM/LPIPS, validation/test/normal processing, Retinexformer comparison,
test4 run, alternate preprocessing, retraining or Ours change. Stop for research
review. Any future benchmark execution requires a new task.

Mailbox clarification: T027-A's completion report was already appended directly
to main as `16b7a83491ca8c6c611d861858f0ba07b3de4b10`, separately from PR51.
As requested, the next append supplies a short T027-A completion pointer before
the T027-B report; previous mailbox entries are preserved.

exporter-ready
