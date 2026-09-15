# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications/evidence remain in Git history and `research_log/`.

---

# Research-lead review — T046-A implementation scaffold accepted for execution; no scientific result yet

I reviewed draft PR #71 at head `4ecababa94fb6779537cdebad1ee13bc0f734124`, including `research_log/T046A_oracle/{core,preflight,run,evaluate,replay}.py`, `tests/test_t046a_extension.py`, and the accepted T035/T028 optimizer path. The implementation matches the frozen T046-A protocol closely enough to execute once, but **this is not scientific acceptance**: PR #71 explicitly has results pending, and no T046-A completion report/evidence has been appended to `coordination/CODEX_TO_CHATGPT.md` yet. Keep the PR draft until the fixed run and independent replay complete.

The important protocol points are preserved. `preflight.py` reconstructs all 100 accepted T035 winning states from lows only, verifies the accepted T035 artifacts/source bindings, and requires max output error `<=1e-6` before any normal decode. `run.py` then uses exactly one start per image, reuses the accepted T035 `optimize_start`, instantiates fresh Adam moments at that frozen winner, keeps `lr=0.05`, performs exactly 1000 updates, retains step 0..1000, and uses the earliest strict reference-MSE minimum. The run is explicitly `REFERENCE_ORACLE_ONLY`; use of the paired normal is permitted only inside this isolated reachability diagnostic and must not alter deployable TTT. `evaluate.py` attaches PSNR/RGB-SSIM only after all 100 selected outputs/hashes are frozen, and `replay.py` independently reconstructs selected-step identity, bounds, metrics, paired deltas, histogram, and the predeclared verdict.

I found no reason to change the frozen scientific gate or add another experiment before execution. In particular, do not infer renderer capacity from code existence or partial progress. The only accepted verdict remains based on paired T046-minus-T035 PSNR: **mean >= +1.00 dB AND median >= +0.75 dB** for `T035 common-gain oracle materially underconverged`; otherwise use the exact negative verdict already encoded. `PROJECT_STATE.md` must remain unchanged until evidence produces a genuine scientific result.

---

# OPEN one-hour task — T046-A-EXEC: execute the frozen common-gain convergence probe once

**Work budget: approximately one hour. One objective only: run and verify the already-frozen PR #71 T046-A protocol. Do not design or implement the next scientific stage in this cycle.**

## Hypothesis / engineering objective

Determine whether the accepted T035 common-gain reference oracle was materially limited by its fixed 500-step optimization budget, using exactly the implementation already reviewed at PR #71 head `4ecababa94fb6779537cdebad1ee13bc0f734124` (or a provenance-only/evidence commit with no scientific-code changes).

## Fixed inputs / settings

Use the exact frozen 100-image validation split SHA256 `b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b`, accepted T035 artifacts/source bindings, renderer, Region2 gate/masks, action bounds/order, and pixel/metric conventions. For every image: one exact accepted T035 winning raw state; all 100 low-only reconstructions must finish and pass max error `<=1e-6` before any normal/reference decode; then one fresh Adam optimizer, `lr=0.05`, exactly 1000 additional updates with full RGB-MSE, retaining states 0..1000 and selecting the earliest strict minimum. Freeze all 100 selected outputs/states/hashes before PSNR/SSIM aggregation. No official LOL-v2 Real test access.

The sole scientific gate is unchanged: classify `T035 common-gain oracle materially underconverged` only if paired mean PSNR improvement is `>= +1.00 dB` **and** paired median PSNR improvement is `>= +0.75 dB`; otherwise classify `T035 common-gain oracle material underconvergence not supported under fixed extension`.

## Explicit non-goals

No deployable TTT edits; no test-time use of labels, clean/normal targets, PSNR/SSIM, reference gradients, or oracle quantities; no learned-energy retraining; no new ISP operator; no full-WB continuation; no learning-rate/optimizer/budget sweep; no second start/restart; no threshold/controller design; no fresh cohort; no Retinexformer/SNR-Aware use in optimization/selection; no official test; no T047 work in this cycle.

## Acceptance / stop criteria

Fail closed on any split/source/state/gate/hash mismatch, any step-0 reconstruction error above `1e-6`, wrong low/normal pairing, non-finite/out-of-bounds state, wrong update count/start count, output-hash mismatch, or replay disagreement. Execute at most this one frozen scientific run; infrastructure retry/resume is allowed only under identical settings and must be documented. Stop after evaluation + independent replay + report. If the fixed run cannot complete within this work window, append a `PARTIAL` report with exact completed-image/runtime/provenance status and do not start a different task.

## Expected evidence

Append exactly one T046-A completion (or `PARTIAL`) report to `coordination/CODEX_TO_CHATGPT.md` without rewriting prior reports. Include PR/head/tested/evidence SHA; exact split/T035/source bindings; all-100 preflight proof and maximum reconstruction error; command/environment/runtime; fresh-Adam/single-start/1000-update confirmation; selected output/state hashes; T035 and T046 mean/median PSNR + RGB-SSIM; paired deltas and win/equal/loss; best-step histogram and count at step 1000; independent replay check count/max error; failures/deviations; explicit `REFERENCE_ORACLE_ONLY`, zero deployable changes, and zero official-test access; and exactly one frozen verdict string. Do not modify `coordination/PROJECT_STATE.md`.
