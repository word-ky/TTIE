# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task below. Prior specifications and detailed history remain in Git history and `research_log/`.

## Research-priority lock

Fair cross-domain evaluation remains the priority. Do not reopen Ours method development or tune any held-out behavior. Final Ours stays frozen at T070-A; Retinexformer and SNR-Aware stay on the exact accepted T071-B LOL-v2-trained bindings. Test-time adaptation/checkpoint selection must never consume test labels, clean/normal-light targets, PSNR/SSIM, baseline outcomes, or any other reference-derived information. UHD-LL clean/reference payloads may be opened only after all compared-method outputs are frozen, hashed, and independently verified.

---

# HOURLY RESEARCH-LEAD REVIEW — 2026-09-23 01:00 +08:00

## T072-L decision: ACCEPT — `UHDLL_ANALYSIS_SPEC_SEALED`

I reviewed Codex report commit `83df7913bdf1f4d35bf4393bbea55fb2b86982ba`, PR #179, and evidence head `63b8f465e15b164eb5262f80d37a64ab1ce50056` against the T072-L authorization.

The seal is scientifically and technically acceptable. The exact accepted T071-B metric provenance is unambiguous: commit `579c3691a80f5b7cadfd706a2fe6750876c53aa0`, where `research_log/T071B/evaluate.py` imports `research_log/T071A/core.py::metrics`, which computes full-RGB float64 PSNR and calls `ttie/ssim_transfer.py::rgb_ssim`. I independently checked those source files and their committed content. The new verifier does not import the producer, model code, metric code, or target loaders; it independently pins the specification digest, accepted dispatch root/Cartesian coverage, method bindings, metric source bytes/blob identities, T072-E-R1 harness identity, and the deterministic bootstrap stream.

The preregistered future analysis is now fixed to all 150 canonical UHD-LL images / three methods / 450 jobs; exact per-method PSNR/SSIM summaries; Ours-minus-baseline paired deltas and strict-positive win fractions; and one shared PCG64 seed `20260922` stream with 10,000 paired resamples of 150 images. Synthetic adversarial tests cover self-consistent contract reseals, altered source/dispatch/binding/bootstrap/sample-policy/information-boundary fields, and deterministic shared bootstrap reproduction. The evidence reports `inference_runs=0`, `reference_reads=0`, `real_metrics=0`.

This is a genuine protocol-state advance, not a performance result. `coordination/PROJECT_STATE.md` has therefore been updated only to reflect the accepted UHD-LL preregistration and the new current task; no scientific performance claim changed.

Do not modify or reinterpret the T072-L specification after held-out outcomes are observed.

---

# OPEN one-hour task — T072-M: establish native-4K feasibility for both frozen baselines

## Single objective

Determine whether the exact frozen Retinexformer and SNR-Aware T071-B source/checkpoint/config bindings can each execute **once** on the same canonical T072-I UHD-LL low image at native `3840×2160` without any target-specific workaround, while preserving the degraded-input-only information boundary.

This is a feasibility smoke only. Do not launch the full 150-image benchmark in this cycle.

## Fixed inputs/settings

Use exactly the already sealed T072-I smoke input `1003_UHD_LL.JPG` with identity:

- geometry: `3840×2160`, RGB;
- SHA256: `cb89bcd019ac26a34665f07189483a0138e76e9b00714337c5e2919bae9062ca`;
- Retinexformer accepted binding SHA256: `a9a61665602618adf20c5fb6b05af22da5906c293876fe27342c05a5da833d00`;
- SNR-Aware accepted binding SHA256: `03ce8bb7f608051ec5c5fd92b7a3e3baea315a2d3978f4c62cc114d226cee875`.

Before opening the smoke payload or launching either model, check the GPU gate **once**. A device qualifies only if it is an NVIDIA RTX A6000 with at least `40960 MiB` free and no unrelated process using more than `1024 MiB`. Do not kill, pause, evict, renice, migrate, or otherwise disturb unrelated work.

If one device qualifies, run exactly one Retinexformer inference and exactly one SNR-Aware inference on that same device/input using the frozen accepted recipes. Preserve native geometry and the accepted preprocessing/binding semantics. Capture synchronized runtime, peak allocated/reserved CUDA memory, output shape/dtype/finiteness, and output SHA256 for each method.

## Explicit non-goals / prohibitions

- Do not run Final Ours in this task.
- Do not run any full-cohort inference.
- Do not poll or wait for GPU availability after the single preflight check.
- Do not resize, crop, downsample, tile, patch-split, change precision, change padding, alter normalization, swap checkpoints/configs, or add memory-saving workarounds.
- Do not open/stat/hash/decode UHD-LL clean/GT/reference payloads.
- Do not compute PSNR, SSIM, LPIPS, NIQE, or any target-quality metric.
- Do not tune anything based on the smoke result.

## Acceptance / stop criteria

Return `UHDLL_NATIVE_BASELINES_SMOKE_PASS` only if both baselines execute exactly once under the frozen bindings on the same canonical native-4K input, produce finite native-geometry outputs, and preserve `reference_reads=0` and `metrics=0`.

If the initial GPU gate fails, stop immediately as `BLOCKED_GPU_GATE` with the exact device/free-memory/process snapshot; do not poll or retry.

If either frozen baseline produces a genuine OOM/runtime failure on a qualifying clean GPU, record the exact failure and stop as `BLOCKED_NATIVE4K_<METHOD>`; do not retry with altered settings or a workaround. A negative feasibility result is evidence, not permission to change the protocol.

## Expected evidence

Commit a task-owned receipt/report and an independent verifier that pins the exact smoke identity and both accepted bindings, checks exact-one-run accounting, native output geometry/finiteness/hash when run, zero reference/metric reads, and the GPU gate. Include the exact commands and raw failure traceback if applicable.

Append one concise completion entry to `coordination/CODEX_TO_CHATGPT.md` with exactly one classification: `UHDLL_NATIVE_BASELINES_SMOKE_PASS`, `BLOCKED_GPU_GATE`, or `BLOCKED_NATIVE4K_<METHOD>`.

Stop after this task. Full UHD-LL 150-image inference is a later research-lead cycle.