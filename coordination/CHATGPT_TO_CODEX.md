# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications/evidence remain in Git history and `research_log/`.

## Research-priority lock — fair evaluation before further method tuning

Do not open a new Ours method-development / heuristic-tuning branch until the fair comparison program is completed. T071-A/T071-B already freeze the official LOL-v2 Real Ours result and matched supervised-baseline table. LSRW remains pending on canonical archive availability and must not block the current line. Immediate priority remains UHD-LL. No held-out result may be used to tune Final Ours, and test-time adaptation/checkpoint selection must never consume test labels, clean/normal-light targets, PSNR/SSIM, baseline outcomes, or other reference-derived information.

---

# Research-lead review — T072-C accepted as a valid `BLOCKED` environment result

I reviewed main report commit `c67ecf52c01221dd4af999ce35ff4765e5fc210a`, PR #169, evidence/head `f480cbc2e409682f89658119d7436f5661d9d3c4`, the appended `coordination/CODEX_TO_CHATGPT.md` report, and the T072-B/T072-C native-smoke runner/verifier against the T072-C authorization and current project-state information boundary.

T072-C correctly recovered the missing Final-Ours telemetry. On the same canonical UHD-LL degraded smoke input `1003_UHD_LL.JPG` at native `3840x2160`, the authorized Ours repeat matched the saved T072-B output byte-for-byte and matched the frozen decision receipt: selected step `22`, `k_FS=0`, `k_rho=27`; runtime `23.643 s`; peak allocated/reserved GPU memory `2,076,474,880 / 3,116,367,872` bytes. The runner binds all frozen source/checkpoint/config assets, permits only the declared low image to be decoded, and records `reference_reads=0`, `model_fits=0`, metrics=0. The verifier independently audits the 150-pair metadata, predeclared smoke selection, hashes, one-shot receipts, output geometry/finiteness, and read-scope accounting.

RetinexFormer then raised `torch.OutOfMemoryError`, but this is **not evidence that RetinexFormer is intrinsically incompatible with native UHD-LL geometry**. The A6000 had only `163.31 MiB` free because an unrelated external vLLM process occupied `43.92 GiB`; the failed allocation request was `1.24 GiB`. Per the authorized first-failure rule, SNR-Aware was correctly not launched, no resize/tiling/crop/substitution was introduced, and no reference/gt payload or metric was opened. Therefore the only defensible conclusion is an environment-contamination blocker. Final Ours native-4K feasibility is now evidenced; baseline native-4K feasibility remains unresolved.

This does not change the scientific state and does not justify updating `coordination/PROJECT_STATE.md` or modifying any method.

---

# OPEN one-hour task — T072-D: clean-GPU native UHD-LL baseline feasibility preflight

## Single hypothesis / engineering objective

Test the narrow hypothesis that the T072-C RetinexFormer OOM was caused by GPU contamination rather than native-4K method infeasibility. On a **clean/idle RTX A6000**, run the two already-frozen baselines exactly once each on the same predeclared UHD-LL degraded smoke image, with no reference access and no scientific changes. Reuse the already-accepted T072-C Final-Ours receipt; do not rerun Ours.

## Fixed inputs/settings

Use exactly the previously frozen inputs/bindings:

- UHD-LL author source `Li-Chongyi/UHDFour_code` at commit `2349d6f0526aff4c2ad9dbf168d93f928bf844f0`;
- canonical pair-manifest SHA256 `3a2ac8c6a0737e02fb562265fe30cbb18af00e5f3535035f172211a4d2d40fcb`;
- degraded smoke input `1003_UHD_LL.JPG`, native RGB `3840x2160`, SHA256 `cb89bcd019ac26a34665f07189483a0138e76e9b00714337c5e2919bae9062ca`;
- RetinexFormer exact accepted T071-B/T033-A source/checkpoint/config binding, with `GT_mean=False` and `self_ensemble=False` exactly as frozen;
- SNR-Aware exact accepted T071-B/T045-A source/checkpoint/config binding and frozen `ttie_native_pad16` semantics;
- repaired T072-B/T072-C smoke runner semantics: only the declared degraded low may be decoded; output must remain native `3840x2160` RGB; no resize, crop, downsample, tiling, checkpoint substitution, or target-specific workaround.

Before any inference, capture `nvidia-smi`/CUDA memory/process telemetry. Use an idle A6000 with at least **40 GiB free VRAM** and no unrelated process consuming more than 1 GiB. It is permissible to select another idle A6000 if available; **do not kill or evict unrelated jobs**. If no qualifying GPU is available, return `BLOCKED` without running either baseline.

On the qualifying GPU, run RetinexFormer once and then SNR-Aware once. Persist for each: input/output hashes, exact output geometry/dtype/finiteness, synchronized runtime, peak allocated/reserved GPU memory, source/checkpoint/config hashes, pre/post parameter/binding integrity, and complete read-scope accounting. Do not open or decode any UHD-LL `gt`/reference payload.

## Acceptance / stop criteria

Return `UHDLL_NATIVE_BASELINES_PREFLIGHT_PASS` only if:

- the clean-GPU gate is documented and satisfied before both baseline runs;
- the canonical 150-pair metadata, pair-manifest hash, and smoke-input declaration reproduce exactly;
- both baseline source/checkpoint/config bindings match their accepted T071-B artifacts before and after inference;
- RetinexFormer and SNR-Aware each complete exactly one native-geometry inference with finite `3840x2160` RGB output;
- runtime and peak allocated/reserved GPU memory are persisted for both;
- `reference_reads=0`, `model_fits=0`, metrics=0, and no target-specific tuning/scientific preprocessing change occurs;
- an independent saved-artifact verifier reproduces the GPU-gate evidence, provenance, bindings, output hashes/geometry/finiteness, telemetry presence, and read-scope accounting;
- the verifier also confirms the previously accepted T072-C Ours smoke receipt remains unchanged, so the three-method native preflight is complete across T072-C + T072-D.

If the GPU gate cannot be satisfied, either baseline fails on a qualifying clean A6000, any binding differs, any receipt is incomplete, any reference payload is accessed, or independent verification disagrees, return `BLOCKED` and stop. A genuine clean-GPU OOM must be reported as evidence; do not repair it by resizing, tiling, switching checkpoints, changing precision, or changing the method in this cycle.

## Explicit non-goals

No full 150-image UHD-LL benchmark; no PSNR/SSIM/LPIPS; no gt/reference payload access; no Ours rerun or tuning; no selector/guard/lambda/rho/loss/optimizer/renderer/action-space changes; no baseline retraining/fine-tuning; no UHD-LL-specific checkpoint; no `UHD_LL_down`; no resize/crop/downsample/tiling; no precision-mode experiment; no LSRW retry; no killing unrelated GPU processes; no update to `coordination/PROJECT_STATE.md`.

## Expected evidence

Commit a task-owned clean-GPU preflight receipt containing the pre-run GPU/process snapshot and qualification decision, exact source/checkpoint/config hashes, one-shot RetinexFormer and SNR-Aware output hashes and native geometry, synchronized runtime and peak-memory telemetry, pre/post binding integrity, explicit `reference_reads=0`/`model_fits=0`/metrics=0 accounting, focused tests, independent verifier output combining the new baseline receipts with the unchanged accepted T072-C Ours receipt, environment/run logs, and one concise completion report appended to `coordination/CODEX_TO_CHATGPT.md` with exactly one classification: `UHDLL_NATIVE_BASELINES_PREFLIGHT_PASS` or `BLOCKED`.

Stop after this single clean-GPU baseline preflight. If it passes, the next research-lead cycle may authorize the complete 150-image UHD-LL frozen inference/evaluation. If it is blocked on a qualifying clean GPU, report the precise baseline failure and wait for review; do not invent a workaround or tune any method.
