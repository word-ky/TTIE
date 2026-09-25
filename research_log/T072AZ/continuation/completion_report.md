# T072-AZ-B — UHDLL_BASELINE_OUTPUTS_FROZEN_RTX4090

The prospectively sealed 512-query-row SNR-Aware execution schedule passed its **one authorized** native-4K low-only smoke. The same paid RTX 4090 then completed one full Retinexformer process and one full SNR-Aware process over the exact 150-image T072-I cohort. Independent per-output verification passed for exactly 150/150 outputs per method. No clean/reference payload was read and no target metric was computed (`reference_reads=0`, `metrics=0`).

## Identity and information boundary

- Fresh sealed-launch GPU snapshot: one `NVIDIA GeForce RTX 4090`, 49140 MiB total, 48510 MiB free, no compute processes; `remote_receipts/continuation_gpu_snapshot.json`.
- Pre-inference port verifier: 70/70 frozen asset hashes and 150/150 native 3840×2160 RGB low-only file hashes/geometry passed. Original accepted SNR checkpoint SHA256 `432d29d370e9f674f1b6763d371b4c24569a86d21f0fd45a5797226274d85781`, config SHA256 `fcb29f50538cfd09ec425c83d7f2072477b24f7c2ab4f23507c3e1b37016b3fb`, binding SHA256 `03ce8bb7f608051ec5c5fd92b7a3e3baea315a2d3978f4c62cc114d226cee875`; Retinexformer accepted binding is unchanged. The query-row schedule source SHA256 is fixed in `prospective_manifest.json` and was checked on the paid host before target-low smoke.
- No resize/crop/tiling, FP16/AMP, checkpoint/config/parameter edit, target tuning, rescue retry, Ours rerun, reference access, or PSNR/SSIM/LPIPS computation. The query-row schedule retains all 32,400 keys/values per query; the official/source audit and synthetic equivalence proof are in `audit.md`, `equivalence_receipt.json`, and `synthetic_model_receipt.json`.

## Smoke and full runs

| Method | Native-4K smoke | Full outputs | Sum per-image inference runtime | Max peak allocated VRAM |
|---|---|---:|---:|---:|
| Retinexformer | Prior T072-AZ smoke passed unchanged | 150/150 | 230.66 s | 20,347,442,176 bytes |
| SNR-Aware | One new query-row smoke passed: `[2160,3840,3]` float32 finite, 3.433 s, peak allocated 14,360,051,200 bytes, peak reserved 20,759,707,648 bytes; output array SHA256 `85813afbc7e783370358744a04bbf6162981478d1f37e42e5bd61d927ed1c493` | 150/150 | 494.14 s | 14,360,051,200 bytes |

Each full exporter process returned 0. Independent `verify_full.py` opened and hashed all 300 output files, checked each row's accepted low SHA256, native shape, float32 dtype, finiteness, array/file SHA256, runtime, peak VRAM, exact coverage, and SNR parameter hash before/after. It returned `UHDLL_BASELINE_OUTPUTS_FROZEN_RTX4090`; the combined manifest is `remote_receipts/continuation_output_manifest.json`, SHA256 `73c8d304c8bb88c4612a13598b5d4dd76378e0c3107bd97011941037a4427001`. The text/JSON receipt archive `continuation_receipts.tar.gz` has SHA256 `9307f02ab0226d432e9c8097701bb3dbb8e22c55a0fd1226c1e0e8f3822c1e0e`.

The 300 `.npy` outputs remain under `/root/autodl-fs/TTIE/T072AZ/runs/full/{retinexformer,snr_aware}/` on the paid host; the 169,518-byte hash manifest and execution evidence are committed in this project's `research_log/T072AZ/continuation/`. Do not open UHD-LL references yet: the expanded main-table methods are not all frozen. No further baseline or analysis task was started. The paid GPU has no remaining compute process from this task; if the instance bills while idle, the user can stop it after preserving the remote outputs.
