# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications/evidence remain in Git history and `research_log/`.

## Research-priority lock — fair evaluation before further method tuning

Do not open a new Ours method-development / heuristic-tuning branch until the fair comparison program is completed. T071-A/T071-B already freeze the official LOL-v2 Real Ours result and matched supervised-baseline table. LSRW remains deferred until the user-provided canonical archive is available and must not block the current line. UHD-LL remains the first cross-domain performance target. No held-out result may be used to tune Final Ours, and test-time adaptation/checkpoint selection must never consume test labels, clean/normal-light targets, PSNR/SSIM, baseline outcomes, or other reference-derived information.

---

# Research-lead review — T072-J accepted as a valid environment `BLOCKED`

I reviewed main report commit `d37366dc420b43689deb32b4e4e7236bed280e35`, PR #177, evidence/head `eb556b561de27bc360065869d4913c11450825de`, the appended `coordination/CODEX_TO_CHATGPT.md` report, and the T072-J receipt/verifier against the T072-J authorization and current information-boundary rules.

The classification is accepted as an environment blocker only. At the pre-launch check, GPU 0 had `2078 MiB` free and GPU 1 had `3499 MiB` free; each was occupied by an unrelated VLLM worker using `44972 MiB`, so neither device met the fixed `>=40960 MiB` free-VRAM / no-unrelated-process-over-`1024 MiB` gate. Codex correctly stopped before opening the smoke payload or launching a model, did not disturb other jobs, and preserved zero-run / zero-reference accounting. The frozen RetinexFormer and SNR-Aware bindings and the T072-I smoke identity are unchanged.

This does **not** establish either success or failure of native-4K baseline inference. It is not a scientific-state change, so `coordination/PROJECT_STATE.md` remains unchanged. Repeating only an instantaneous GPU snapshot each hour has low research value; the next task therefore uses one bounded availability window while preserving exactly the same scientific experiment.

---

# OPEN one-hour task — T072-K: bounded clean-GPU availability watch + native-4K baseline smoke

## Single hypothesis / engineering objective

Resolve the remaining UHD-LL native-4K baseline feasibility question within one bounded work window by waiting non-invasively for a qualifying A6000 and, only if one appears, executing the exact previously authorized RetinexFormer and SNR-Aware smoke once each under the frozen protocol.

## Fixed inputs/settings

Use exactly the T072-I sealed low-only contract and its canonical `1003_UHD_LL.JPG` row (`3840×2160`, RGB, SHA256 `cb89bcd019ac26a34665f07189483a0138e76e9b00714337c5e2919bae9062ca`). Keep the same frozen scientific identities:

- RetinexFormer upstream `1e9a0efce4b306b6701b824768370ff26066c32a`, checkpoint SHA256 `539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b`, config SHA256 `5260d0c65878f6a39712f70948be1936d8583531491d832cb59362fffba894ac`, accepted binding SHA256 `a9a61665602618adf20c5fb6b05af22da5906c293876fe27342c05a5da833d00`, `GT_mean=False`, `self_ensemble=False`.
- SNR-Aware upstream `1113144c82adc8bcc4a9ec27749ed75f196a4e4d`, checkpoint SHA256 `432d29d370e9f674f1b6763d371b4c24569a86d21f0fd45a5797226274d85781`, config SHA256 `fcb29f50538cfd09ec425c83d7f2072477b24f7c2ab4f23507c3e1b37016b3fb`, parameter SHA256 `11d3d667821719bd51e6e608c7876774b001643a28ed85193054bb45428190d4`, accepted binding SHA256 `03ce8bb7f608051ec5c5fd92b7a3e3baea315a2d3978f4c62cc114d226cee875`, mode `ttie_native_pad16`.

For at most **55 minutes**, poll `nvidia-smi` no more frequently than once every **5 minutes**. A device qualifies only if it is an NVIDIA RTX A6000 with at least `40960 MiB` free VRAM and no unrelated process using more than `1024 MiB`. Do not open/stat/hash/decode the smoke image while no device qualifies. Do not kill, pause, evict, renice, or otherwise interfere with other processes.

At the **first** qualifying snapshot, select that device and stop polling. Run RetinexFormer once in a fresh process, then re-check the same GPU gate immediately before SNR-Aware and run SNR-Aware once in a fresh process if the gate still qualifies. Batch size 1; native `3840×2160`; no resize, crop, downsample, tiling, target-specific normalization, precision workaround, checkpoint/config change, or target-specific tuning. Do not run Final Ours.

## Acceptance / stop criteria

Return `UHDLL_NATIVE_BASELINES_PREFLIGHT_PASS` only if both baselines each complete their single allowed native-4K run, produce finite outputs corresponding to the native input, preserve the exact frozen bindings, and have `reference_reads=0`. Record output geometry/dtype/SHA256, wall-clock runtime, peak CUDA allocated/reserved memory, the qualifying pre-launch process ledger, and the re-check before the second baseline.

Return `BLOCKED` if no qualifying GPU appears within the bounded 55-minute window. Also return `BLOCKED` and stop immediately, with exact evidence and no retry/workaround, if: the selected GPU ceases to satisfy the gate before the second method; either method has a genuine OOM or execution failure; any binding/config mismatch occurs; the input/read-scope contract is violated; or an output is non-finite. Do not infer the unrun method's feasibility after a first failure.

## Explicit non-goals

Do not run the full 150-image benchmark. Do not rerun Final Ours. Do not read/open/stat/hash/decode UHD-LL GT/reference/clean payloads. Do not compute PSNR/SSIM/LPIPS or any quality metric. Do not change method code, precision, padding semantics, normalization, checkpoint, config, or image geometry. Do not use another dataset. Do not update `coordination/PROJECT_STATE.md`.

## Expected evidence

Commit one task-owned machine-readable receipt and concise report containing: every GPU-gate timestamp/snapshot in the bounded watch, whether/when a qualifying device first appeared, exact frozen bindings, smoke-input identity from the T072-I seal, one-run accounting for each attempted baseline, output hash/geometry/finiteness, runtime/peak-memory telemetry when run, and explicit `reference_reads=0`. Include a small independent verifier for the receipt and append one concise completion report to `coordination/CODEX_TO_CHATGPT.md` with exactly one classification: `UHDLL_NATIVE_BASELINES_PREFLIGHT_PASS` or `BLOCKED`.

Stop after this bounded task. The complete 150-image UHD-LL inference remains a separate later research-lead cycle.