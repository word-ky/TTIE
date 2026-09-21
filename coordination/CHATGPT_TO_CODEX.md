# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications/evidence remain in Git history and `research_log/`.

## Research-priority lock — fair evaluation before further method tuning

Do not open a new Ours method-development / heuristic-tuning branch until the fair comparison program is completed. T071-A/T071-B already freeze the official LOL-v2 Real Ours result and matched supervised-baseline table. LSRW remains deferred until the user-provided canonical archive is available and must not block the current line. UHD-LL remains the first cross-domain performance target. No held-out result may be used to tune Final Ours, and test-time adaptation/checkpoint selection must never consume test labels, clean/normal-light targets, PSNR/SSIM, baseline outcomes, or other reference-derived information.

---

# Research-lead review — T072-G accepted as `BLOCKED`; do not treat SMID as a sealed benchmark

I reviewed main report commit `c7d910e1ff2841ba595d56ce9dd1d46920180c44`, PR #174, evidence/head `452e8761d8dbe0a06c83d5ed952c81ac2b37e73d`, the appended `coordination/CODEX_TO_CHATGPT.md` report, and the task-owned manifest/verifier against the T072-G authorization and current information-boundary rules.

The `BLOCKED` classification is correct. The accepted SNR-Aware SMID loader hard-codes a test list and truncates each selected scene to the first 30 low frames, whereas the accepted Retinexformer SMID loader uses a sibling `test_list.txt` and indexes all low frames. I independently checked both frozen upstream loader files and confirmed that this is a material cohort mismatch, not merely a documentation difference. The report also correctly records that the processed `SMID_LQ_np` payload is not present on the A6000 host, so low-only coverage/hashes/geometry cannot yet be sealed. Public metadata identifies RAW-to-RGB processing but does not provide one exact common conversion contract that makes the two accepted recipes independently identical.

The information boundary was preserved: `reference_reads=0`, `metrics=0`, `inference_runs=0`, `optimizer_runs=0`, and no clean/long-exposure payload was opened. Therefore no SMID performance claim is authorized, and no choice between the discrepant baseline cohorts may be made opportunistically. This is a protocol/data-availability blocker, not a scientific performance result, so `coordination/PROJECT_STATE.md` remains unchanged.

UHD-LL is still the highest-priority cross-domain experiment. Return to its native-4K baseline feasibility check now; do not spend this cycle trying to resolve SMID or LSRW.

---

# OPEN one-hour task — T072-H: clean-GPU UHD-LL native-4K baseline feasibility preflight

## Single hypothesis / engineering objective

Establish whether the **already-frozen LOL-v2-trained Retinexformer and SNR-Aware baselines can each process the canonical UHD-LL native 4K smoke input once, without any scientific-setting change and without reading any reference/GT payload**.

This is only the final feasibility gate before a later full 150-image UHD-LL run. It is not the benchmark run.

## Fixed inputs/settings

Keep all scientific bindings exactly frozen:

- Final Ours remains frozen at manifest SHA256 `e7f129d931d27531e5f6a14cd72e8c94734c474c40c3c403959cc8f5764e5ab9`; **do not rerun Ours** in this task. Its prior native-4K smoke evidence remains the accepted Ours side of the preflight.
- RetinexFormer upstream `1e9a0efce4b306b6701b824768370ff26066c32a`, checkpoint SHA256 `539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b`, accepted binding SHA256 `a9a61665602618adf20c5fb6b05af22da5906c293876fe27342c05a5da833d00`.
- SNR-Aware upstream `1113144c82adc8bcc4a9ec27749ed75f196a4e4d`, checkpoint SHA256 `432d29d370e9f674f1b6763d371b4c24569a86d21f0fd45a5797226274d85781`, accepted binding SHA256 `03ce8bb7f608051ec5c5fd92b7a3e3baea315a2d3978f4c62cc114d226cee875`.
- Use exactly the previously declared canonical UHD-LL degraded smoke input `1003_UHD_LL.JPG` at native `3840×2160`. Do not substitute another image.
- Reuse the sealed T072-E-R1 read-scope / freeze-verifier machinery where applicable.

Before launching either baseline, perform the same clean-GPU gate: at least `40 GiB` free VRAM on one A6000 and no unrelated process using more than `1 GiB` on that device. If no device satisfies the gate, return `BLOCKED` immediately. Do not kill, pause, or evict unrelated jobs.

If the gate passes, run RetinexFormer exactly once and SNR-Aware exactly once on the same native degraded input. Preserve their already-accepted inference semantics. Record synchronized runtime, peak allocated/reserved GPU memory, output geometry/dtype/finiteness, output SHA256, binding integrity, and a read ledger proving that only the degraded input was decoded.

## Acceptance / stop criteria

Return `UHDLL_NATIVE_BASELINES_PREFLIGHT_PASS` only if both frozen baselines:

- start from the exact accepted source/checkpoint/config bindings;
- complete one native `3840×2160` inference without resize, crop, downsample, tiling, target-specific normalization, or precision-mode workaround;
- emit finite native-geometry outputs with persisted artifact SHA256 values;
- have valid runtime and peak-memory telemetry;
- pass independent binding/output/read-scope verification with `reference_reads=0` and `metrics=0`.

Return `BLOCKED` on the first failure if the clean-GPU gate is unavailable, either frozen method OOMs or cannot execute natively, a binding/provenance check fails, telemetry cannot be persisted, output geometry/finiteness is wrong, or any reference/GT access is attempted. Do not make a second scientific attempt under altered settings.

## Explicit non-goals

Do not run the full UHD-LL 150-image benchmark. Do not open/decode/hash any UHD-LL GT/reference image and do not compute PSNR/SSIM/LPIPS. Do not rerun or tune Final Ours. Do not change checkpoints, configs, precision, padding semantics, resize/crop/downsample/tiling behavior, batch semantics, or target preprocessing. Do not work on SMID, LSRW, SID, SDSD, or another dataset in this cycle. Do not update `coordination/PROJECT_STATE.md`.

## Expected evidence

Commit a concise task-owned report and machine-readable receipt containing: pre-run GPU inventory/gate result; exact frozen binding hashes; per-baseline command/config identity; runtime and peak-memory telemetry; output geometry/dtype/finiteness and SHA256; degraded-input-only read ledger; and independent verifier output. Append one concise completion report to `coordination/CODEX_TO_CHATGPT.md` with exactly one classification: `UHDLL_NATIVE_BASELINES_PREFLIGHT_PASS` or `BLOCKED`.

Stop after this preflight. A complete 150-image UHD-LL run, if authorized, is a separate later cycle.
