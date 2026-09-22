# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications/evidence remain in Git history and `research_log/`.

## Research-priority lock — fair evaluation before further method tuning

Do not open a new Ours method-development / heuristic-tuning branch until the fair comparison program is completed. T071-A/T071-B already freeze the official LOL-v2 Real Ours result and matched supervised-baseline table. LSRW remains deferred until the user-provided canonical archive is available and must not block the current line. UHD-LL remains the first cross-domain performance target. No held-out result may be used to tune Final Ours, and test-time adaptation/checkpoint selection must never consume test labels, clean/normal-light targets, PSNR/SSIM, baseline outcomes, or other reference-derived information.

---

# Research-lead review — T072-H accepted as `BLOCKED`; environment only, no scientific conclusion

I reviewed main report commit `baae9fc4f3d16d14778294523e08dfe76b593ba9`, PR #175, evidence/head `28dc7edd3d7a5e4267a30a78a889b2e642e4e49b`, the appended `coordination/CODEX_TO_CHATGPT.md` report, and the task-owned receipt/verifier against the T072-H authorization and current information-boundary rules.

The `BLOCKED` classification is correct. At the gate check, GPU0/GPU1 had only `3497/3499 MiB` free because unrelated VLLM workers occupied about `44972 MiB` each, far below the fixed `>=40960 MiB` free-VRAM requirement. Per the stop rule, neither RetinexFormer nor SNR-Aware was launched, Final Ours was not rerun, the smoke low was not decoded, and no external process was disturbed. Accounting remains `inference_runs=0`, `optimizer_runs=0`, `model_fits=0`, `reference_reads=0`, `metrics=0`, `smoke_input_decodes=0`.

Therefore this cycle adds no native-4K feasibility evidence for either baseline and changes no scientific conclusion. Do not interpret repeated gate failure as method failure. `coordination/PROJECT_STATE.md` remains unchanged. Because repeated GPU occupancy is currently the only blocker, use this cycle to finish the non-GPU execution contract for the full UHD-LL low-only phase rather than re-running the same gate immediately.

---

# OPEN one-hour task — T072-I: seal the complete UHD-LL 150-low inference dispatch manifest (no real inference)

## Single hypothesis / engineering objective

Create and independently verify a **complete, immutable, low-only execution manifest for the canonical UHD-LL 150-image test input split and all three frozen methods**, so that once a qualifying GPU becomes available the later inference phase can launch without any cohort, binding, path, or setting decisions being made after seeing target results.

This task is infrastructure-only. It must not run any real model.

## Fixed inputs/settings

Use the already accepted canonical UHD-LL `testing_set/input` cohort from T072-B/T072-E-R1 and the existing sealed two-stage evaluation machinery. Read only degraded/input payloads. It is permissible to enumerate, hash, and decode the 150 low images solely to record native geometry/dtype and validate that every low is readable; do not access the GT/reference directory or any clean/normal-light payload in any way.

Freeze the exact existing scientific identities:

- Final Ours manifest SHA256 `e7f129d931d27531e5f6a14cd72e8c94734c474c40c3c403959cc8f5764e5ab9`.
- RetinexFormer upstream `1e9a0efce4b306b6701b824768370ff26066c32a`, checkpoint SHA256 `539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b`, accepted binding SHA256 `a9a61665602618adf20c5fb6b05af22da5906c293876fe27342c05a5da833d00`.
- SNR-Aware upstream `1113144c82adc8bcc4a9ec27749ed75f196a4e4d`, checkpoint SHA256 `432d29d370e9f674f1b6763d371b4c24569a86d21f0fd45a5797226274d85781`, accepted binding SHA256 `03ce8bb7f608051ec5c5fd92b7a3e3baea315a2d3978f4c62cc114d226cee875`.

The manifest must deterministically define exactly `150` unique canonical low inputs and exactly `450` method-image jobs (`150 × 3`), with collision-free output paths, native input geometry, exact frozen method bindings, and no target-specific options. The future executor may contain placeholders for runtime GPU selection, but no scientific parameter may remain unspecified or inferred from outputs.

Add an independent verifier that does not trust candidate self-declarations. It must anchor the frozen method identities independently, recompute low-file SHA256 values from bytes, assert the exact 150-low cohort and 450-job Cartesian product, reject duplicate/missing/extra inputs or jobs, reject altered method bindings, reject output-path collisions, and reject any executable/path field that points into GT/reference/clean targets.

## Acceptance / stop criteria

Return `UHDLL_FULL_LOW_ONLY_DISPATCH_SEALED` only if all of the following hold:

- exactly 150 unique canonical degraded inputs are recorded, each with byte SHA256 and native geometry/dtype;
- exactly 450 jobs exist: every low appears once for each of Final Ours, RetinexFormer, and SNR-Aware, with no extras or omissions;
- all three method bindings exactly match the frozen identities above;
- output paths are deterministic and collision-free;
- an independent verifier passes on the canonical manifest and fails on adversarial fixtures for missing job, extra job, duplicate input, altered binding, altered low hash, output-path collision, and injected reference/GT path;
- accounting is `inference_runs=0`, `optimizer_runs=0`, `model_fits=0`, `reference_reads=0`, `metrics=0`.

Return `BLOCKED` if the canonical 150-low cohort cannot be resolved from low-side data alone, any low cannot be read/hashed, the Cartesian job set cannot be made exact without changing scientific settings, or the verifier cannot fail closed. Do not solve a blocker by reading reference/GT data.

## Explicit non-goals

Do not run Final Ours, RetinexFormer, or SNR-Aware. Do not run the native-4K smoke again. Do not open/stat/hash/decode any UHD-LL GT/reference/clean file. Do not compute PSNR/SSIM/LPIPS or any quality metric. Do not alter checkpoints, configs, precision, padding, resize/crop/downsample/tiling behavior, normalization, batch semantics, or method code. Do not work on SMID, LSRW, SID, SDSD, or another dataset. Do not update `coordination/PROJECT_STATE.md`.

## Expected evidence

Commit a task-owned machine-readable full-dispatch manifest, a concise provenance/read-ledger receipt, the independent verifier and adversarial tests, plus a short report stating the 150-low count, 450-job count, manifest/root hash, low-only read accounting, and exact frozen binding hashes. Append one concise completion report to `coordination/CODEX_TO_CHATGPT.md` with exactly one classification: `UHDLL_FULL_LOW_ONLY_DISPATCH_SEALED` or `BLOCKED`.

Stop after sealing this low-only dispatch contract. Native-4K baseline feasibility and the real 150-image inference remain separate later cycles and still require the clean-GPU gate.
