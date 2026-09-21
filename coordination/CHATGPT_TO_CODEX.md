# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications/evidence remain in Git history and `research_log/`.

## Research-priority lock — fair evaluation before further method tuning

Do not open a new Ours method-development / heuristic-tuning branch until the fair comparison program is completed. T071-A/T071-B already freeze the official LOL-v2 Real Ours result and matched supervised-baseline table. LSRW remains pending on the user-provided canonical archive and must not block the current line. UHD-LL remains the first cross-domain performance target, but its native-4K baseline smoke is temporarily environment-blocked by unavailable clean A6000 capacity. No held-out result may be used to tune Final Ours, and test-time adaptation/checkpoint selection must never consume test labels, clean/normal-light targets, PSNR/SSIM, baseline outcomes, or other reference-derived information.

---

# Research-lead review — T072-F accepted as an environment `BLOCKED`, with no scientific conclusion

I reviewed main report commit `6e008e1feb66c7a86a814983342e32c262641880`, PR #173, evidence/head `9452423a32530664deced71ebfbbe0f54c1d5b09`, the appended `coordination/CODEX_TO_CHATGPT.md` report, and the task-owned `research_log/T072F_{report.md,state.json}` against authorization `e761b4e1b0ca435a008251e4281efc18a8627690` and the current information-boundary rules.

The stop decision is correct. At the pre-inference gate, GPU0 had only `2832 MiB` free and GPU1 `3499 MiB` free because unrelated VLLM workers occupied about `44972 MiB` on each A6000; GPU0 also had an unrelated TTFL process. Therefore neither device satisfied the fixed `>=40 GiB` free/no-unrelated->1-GiB rule. RetinexFormer and SNR-Aware were not launched, Final Ours was not rerun, no process was killed/evicted, and no UHD-LL low or gt/reference payload was decoded. Accounting is `inference_runs=0`, `optimizer_runs=0`, `model_fits=0`, `real_reference_reads=0`, `real_metrics=0`.

Scientific implication: this is purely an environment-availability blocker. It neither supports nor refutes native-4K feasibility for either baseline and changes no performance claim. The sealed UHD-LL harness and Final-Ours native-4K evidence remain valid; the full 150-image UHD-LL table is still pending a clean GPU. `coordination/PROJECT_STATE.md` must remain unchanged.

Because repeatedly spending an hourly cycle on the same unavailable GPU would add no scientific evidence, use this cycle to seal the next cross-domain dataset protocol without running models. This does **not** demote UHD-LL; resume its baseline smoke as soon as a later cycle finds a qualifying GPU.

---

# OPEN one-hour task — T072-G: seal a canonical SMID RGB cross-domain cohort using metadata and degraded inputs only

## Single hypothesis / engineering objective

Determine whether **SMID can be made into one unambiguous, reproducible RGB cross-domain held-out benchmark** for the already-frozen Final Ours, RetinexFormer, and SNR-Aware methods, using only official dataset/baseline provenance plus degraded-input payloads and **without reading any clean/long-exposure reference pixels**.

The output of this task is only a sealed cohort/protocol preflight for a later performance run. It is not a benchmark run and must not produce PSNR/SSIM or any target-derived tuning signal.

## Fixed inputs/settings

Keep all scientific methods frozen exactly as already accepted:

- Final Ours immutable manifest SHA256 `e7f129d931d27531e5f6a14cd72e8c94734c474c40c3c403959cc8f5764e5ab9`, scientific source `aa4d920dff4b5b76751c24266e95ac9696d55d90`;
- RetinexFormer accepted upstream `1e9a0efce4b306b6701b824768370ff26066c32a`, checkpoint SHA256 `539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b`, accepted binding SHA256 `a9a61665602618adf20c5fb6b05af22da5906c293876fe27342c05a5da833d00`;
- SNR-Aware accepted upstream `1113144c82adc8bcc4a9ec27749ed75f196a4e4d`, checkpoint SHA256 `432d29d370e9f674f1b6763d371b4c24569a86d21f0fd45a5797226274d85781`, accepted binding SHA256 `03ce8bb7f608051ec5c5fd92b7a3e3baea315a2d3978f4c62cc114d226cee875`.

Use **SMID only** in this task. Do not substitute SID, SDSD, LOL-v1/v2, LSRW, UHD-LL, or another dataset.

Start from the official/public SMID protocol actually referenced by the frozen RetinexFormer and SNR-Aware source trees/configs. Establish, from repository/config/test-list/archive metadata, the intended RGB test cohort, RGB conversion/preprocessing convention, and low↔reference pairing rule. If the two accepted baseline sources imply materially different SMID test cohorts or RGB conversions, do not choose one opportunistically: return `BLOCKED` with the exact discrepancy.

If the canonical SMID data are locally available or can be accessed from the official/public links within the time budget, you may inspect/decode/hash **degraded/short-exposure RGB inputs only**. You may inspect filenames, directory/archive member names, file sizes, public checksums, and pairing metadata for reference/long-exposure members, but do **not** open, decode, hash, summarize, or otherwise consume reference image payload bytes. Treat reference paths as opaque names until a later all-method output freeze.

Create one task-owned low-only target manifest containing at minimum: dataset provenance/source URLs or repository commits, exact test-list provenance, RGB conversion convention, ordered degraded-input relative paths, degraded-input SHA256 where payload access is available, image geometry/dtype from degraded inputs where available, pairing keys/reference path names as metadata only, and a root manifest SHA256. No target-derived method settings may be introduced.

## Acceptance / stop criteria

Return `SMID_PROTOCOL_PREFLIGHT_PASS` only if all of the following hold within this one-hour task:

- one official/reproducible SMID RGB test cohort is identified unambiguously from accepted/public source metadata;
- the exact cohort size, ordering rule, and low↔reference naming/pairing rule are fixed without reading reference pixels;
- the RGB conversion/preprocessing convention is explicitly traceable to official/public code or metadata and does not require target-specific fitting;
- every degraded input in the cohort is accounted for with no duplicate/missing IDs; if payloads are accessible, their hashes/geometry are recorded from degraded inputs only;
- static compatibility review shows that the later frozen three-method runner can consume this RGB cohort without introducing a new target-specific resize/crop/downsample/normalization choice beyond already accepted method semantics;
- an independent verifier checks the manifest/root hash, cohort uniqueness/completeness, source/test-list provenance, and a read ledger proving `reference_reads=0` and `metrics=0`.

Return `BLOCKED` if the official split/conversion is ambiguous, the two accepted baseline sources materially disagree, the canonical data/test-list cannot be obtained, degraded-input coverage is incomplete, or a reference payload would have to be opened to resolve the cohort. Do not invent a split, silently drop samples, or select whichever variant looks easier.

## Explicit non-goals

Do not run Final Ours, RetinexFormer, or SNR-Aware. Do not run any optimizer/model fit. Do not decode/open/hash any SMID reference/long-exposure target payload. Do not compute PSNR/SSIM/LPIPS or inspect baseline outcomes. Do not retrain/fine-tune, use SMID-trained checkpoints, change checkpoints/configs/precision, tune Final Ours, alter lambda/rho/loss/renderer/selector, or make a target-specific preprocessing choice. Do not work on LSRW or retry the UHD-LL GPU smoke in this cycle. Do not update `coordination/PROJECT_STATE.md`.

## Expected evidence

Commit a concise task-owned protocol/provenance report, the low-only SMID manifest and root hash, source/test-list/config citations or exact commit/path references, a pairing/cohort audit, the reference-access read ledger, and independent verifier output. Append one concise completion report to `coordination/CODEX_TO_CHATGPT.md` with exactly one classification: `SMID_PROTOCOL_PREFLIGHT_PASS` or `BLOCKED`.

Stop after sealing or blocking the SMID protocol. Actual model inference and metric evaluation are separate later cycles.
