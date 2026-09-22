# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications/evidence remain in Git history and `research_log/`.

## Research-priority lock — fair evaluation before further method tuning

Do not open a new Ours method-development / heuristic-tuning branch until the fair comparison program is completed. T071-A/T071-B already freeze the official LOL-v2 Real Ours result and matched supervised-baseline table. LSRW remains deferred until the user-provided canonical archive is available and must not block the current line. UHD-LL remains the first cross-domain performance target. No held-out result may be used to tune Final Ours, and test-time adaptation/checkpoint selection must never consume test labels, clean/normal-light targets, PSNR/SSIM, baseline outcomes, or other reference-derived information.

---

# Research-lead review — T072-I accepted as `UHDLL_FULL_LOW_ONLY_DISPATCH_SEALED`

I reviewed main report commit `34e2f1d4b62e183509eee5cf1faeb133c68132ce`, PR #176, evidence/head `4c236e847180164c09b9c802bed530d0ee507d86`, the appended `coordination/CODEX_TO_CHATGPT.md` report, and the T072-I sealing/verifier/tests against the T072-I authorization and current information-boundary rules.

The infrastructure result is accepted. The canonical low-only cohort is fixed at exactly 150 unique native `3840×2160` RGB JPEG inputs, and the dispatch is the exact 450-job Cartesian product across frozen Final Ours, RetinexFormer, and SNR-Aware. The manifest root is `a624d66cbb735bf1483720b1256b63e01d53b55da9f05eb0634d975214f4bd3e`; the low byte-index SHA256 is `95c0a189b406c7dedb247ac2921f332658d76ced0db70a9c301e61d6c9d8d8fd`. The independent verifier anchors frozen method identities, recomputes low-file hashes/geometry from bytes, checks the complete Cartesian job set and deterministic collision-free outputs, and fails on the required adversarial fixtures. The report and receipt preserve `inference_runs=0`, `reference_reads=0`, and `metrics=0`.

This is an execution-contract milestone, not a new performance or native-4K feasibility result. Therefore `coordination/PROJECT_STATE.md` remains unchanged. The next scientific bottleneck is still the same: establish whether the two frozen supervised baselines can run the predeclared native-4K UHD-LL input on a genuinely clean A6000 before authorizing the complete 150-image inference phase.

---

# OPEN one-hour task — T072-J: clean-GPU native-4K baseline feasibility smoke

## Single hypothesis / engineering objective

Establish, with one bounded clean-GPU experiment, whether the exact frozen RetinexFormer and SNR-Aware bindings can each process the predeclared canonical UHD-LL smoke input at native `3840×2160` geometry without any target-specific workaround or reference access.

## Fixed inputs/settings

Use the T072-I sealed low-only contract and its canonical `1003_UHD_LL.JPG` row as the only image input. Use exactly the frozen T072-I/T071-B scientific identities:

- RetinexFormer upstream `1e9a0efce4b306b6701b824768370ff26066c32a`, checkpoint SHA256 `539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b`, config SHA256 `5260d0c65878f6a39712f70948be1936d8583531491d832cb59362fffba894ac`, accepted binding SHA256 `a9a61665602618adf20c5fb6b05af22da5906c293876fe27342c05a5da833d00`, mode `default_no_gt_mean` with `GT_mean=False`, `self_ensemble=False`.
- SNR-Aware upstream `1113144c82adc8bcc4a9ec27749ed75f196a4e4d`, checkpoint SHA256 `432d29d370e9f674f1b6763d371b4c24569a86d21f0fd45a5797226274d85781`, config SHA256 `fcb29f50538cfd09ec425c83d7f2072477b24f7c2ab4f23507c3e1b37016b3fb`, parameter SHA256 `11d3d667821719bd51e6e608c7876774b001643a28ed85193054bb45428190d4`, accepted binding SHA256 `03ce8bb7f608051ec5c5fd92b7a3e3baea315a2d3978f4c62cc114d226cee875`, mode `ttie_native_pad16`.

Before launching any model, require one A6000 with at least `40960 MiB` free VRAM and no unrelated process using more than `1024 MiB`. Record `nvidia-smi` evidence. Run each baseline exactly once, sequentially, in a fresh process on the same qualifying device so allocator state is reset between methods. Re-check the GPU gate immediately before the second baseline.

Input must remain native `3840×2160`; batch size 1. No resize, crop, downsample, tiling, precision workaround, target-specific normalization, target-specific tuning, checkpoint substitution, or config change. The only readable image payload is the degraded low. Do not run Final Ours in this task.

## Acceptance / stop criteria

Return `UHDLL_NATIVE_BASELINES_PREFLIGHT_PASS` only if both baselines, each on its single allowed run, produce a finite output corresponding to the native `3840×2160` input under the frozen binding, with no reference access. For each method record output geometry/dtype, output SHA256, wall-clock runtime, peak CUDA allocated/reserved memory, pre-launch free VRAM/process ledger, and `reference_reads=0`.

Return `BLOCKED` and stop immediately if no GPU satisfies the fixed gate. If a qualifying clean GPU exists but either baseline encounters a genuine OOM, binding/config mismatch, decode/read-scope violation, non-finite output, or other execution failure, preserve the exact failure evidence and stop without retry or workaround. Do not infer feasibility for the unrun method after a first failure.

## Explicit non-goals

Do not run the full 150-image benchmark. Do not rerun Final Ours. Do not open/stat/hash/decode UHD-LL GT/reference/clean payloads. Do not compute PSNR/SSIM/LPIPS or any quality metric. Do not change precision, padding semantics, resize/crop/downsample/tiling behavior, normalization, checkpoint, config, or method code. Do not kill/pause/evict unrelated GPU processes. Do not work on LSRW, SMID, SID, SICE, SDSD, or another dataset. Do not update `coordination/PROJECT_STATE.md`.

## Expected evidence

Commit a task-owned machine-readable receipt plus concise report containing the GPU gate evidence, exact frozen binding hashes, smoke-input identity from the T072-I seal, one-run accounting for each attempted baseline, output hash/geometry/finiteness, runtime and peak-memory telemetry, and explicit read ledger. Include a small independent verifier that confirms the receipt is consistent with the frozen bindings, exactly-one-run allowance, native geometry, and `reference_reads=0`. Append one concise completion report to `coordination/CODEX_TO_CHATGPT.md` with exactly one classification: `UHDLL_NATIVE_BASELINES_PREFLIGHT_PASS` or `BLOCKED`.

Stop after this bounded smoke. The complete 150-image inference remains a separate later research-lead cycle.
