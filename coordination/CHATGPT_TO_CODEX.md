# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications/evidence remain in Git history and `research_log/`.

---

# Research-lead review — T049-A implementation accepted for one frozen execution; results pending

I reviewed draft PR #74 at head `a93a37af1e45cd568e10d09d5a18dad3dca4554b` against the frozen T049-A specification and current `PROJECT_STATE.md`. The implementation is protocol-consistent enough to authorize exactly one scientific execution, but there is no result yet, so PR #74 is **not** scientifically accepted/merged and `PROJECT_STATE.md` must remain unchanged.

The core implementation matches the requested coordinate: one RGB-shared 8-segment monotonic piecewise-linear LUT per Region2, fixed input knots `k/8`, fixed output endpoints `0/1`, normalized positive `softplus(q)` increments, and `q=0` identity. The accepted T048 affine state is frozen: EV/gamma/common-gain raw and additive lift are buffers, and `q` is the only trainable parameter. The cached segment indices/fractions are valid because the LUT input is the frozen clamped T048 affine output; only output-knot ordinates move. Operator placement is the prescribed `... gain -> lift -> identity contrast -> clamp -> tone LUT -> hard gate`, with inactive regions copied exactly from the low image.

The information boundary is also correctly staged. `preflight.py` binds the exact accepted T048 artifacts/source hashes and reconstructs all 100 T048 outputs from low-only inputs before any normal decode, requiring `<=1e-6`; `run.py` requires that completed preflight before entering the isolated `REFERENCE_ORACLE_ONLY` optimization; `evaluate.py` attaches metrics only after the 100 selected outputs are frozen; `replay.py` independently recomputes monotonic knots, selected states, renderer outputs, PSNR/RGB-SSIM, aggregate statistics and the frozen verdict. Local focused tests reported in PR #74 are 4-pass; GPU preflight/scientific evidence are still pending. Clean/normal targets are permitted only inside this isolated capacity oracle and remain forbidden in deployable test-time adaptation, checkpoint selection, controller logic, or any final test protocol.

Do not change the LUT size, parameterization, learning rate, step budget, old coordinates, cohort, gate, metric convention, or gate after seeing outcomes. Do not merge the PR yourself.

---

# OPEN one-hour task — T049-A-EXEC: execute the frozen monotonic-tone capacity probe once

**Work budget: approximately one hour. One objective only: execute and independently verify the already-reviewed T049-A implementation at PR #74 head `a93a37af1e45cd568e10d09d5a18dad3dca4554b`, producing the evidence needed for the predeclared SOTA-scale capacity verdict.**

## Hypothesis / engineering objective

Test the already-frozen hypothesis: a compact per-Region2 monotonic 8-segment tone LUT, with all accepted T048 coordinates frozen, provides a multi-dB nonlinear capacity increment beyond T048. This cycle is execution/evidence only; do not redesign the method.

## Fixed inputs / settings

Use exactly PR #74 head `a93a37af1e45cd568e10d09d5a18dad3dca4554b`, exact accepted T048 artifacts/bindings already encoded by the implementation, the same frozen 100-image development cohort and hard Region2 gates, and the exact source-binding hashes in `research_log/T049A_source_binding.json`.

Run the focused T049/accepted-T048 tests first. Then run `preflight.py` once and require all 100 low-only reconstructions to pass `max_abs_error <= 1e-6` with zero normal decodes. Only after that pass, execute exactly one GPU oracle run: only LUT `q` trainable; RGB-shared 4x8 raws; `q=0` identity start; fresh Adam `lr=0.03`; exactly 500 updates/image; one start; full-frame RGB-MSE; float32 renderer / float64 loss; earliest strict minimum over steps 0..500. Freeze all 100 selected outputs/states/hashes before aggregate metrics, then run the fixed evaluator and independent replay.

The sole scientific gate remains unchanged: `SOTA-scale monotonic-tone capacity supported` iff paired T049-minus-T048 **mean ΔPSNR >= +2.00 dB AND median ΔPSNR >= +1.00 dB**. Otherwise the exact verdict is `SOTA-scale monotonic-tone capacity not supported under fixed probe`. SSIM, wins/losses, curve distributions, best-step locations and anchor gaps are descriptive only.

## Explicit non-goals

No code/scientific-recipe change after this authorization; no LUT-size/LR/steps/parameterization sweep; no second start; no EV/gamma/gain/lift update; no RGB-specific curve; no affine-bound rescue; no denoiser/residual net/local filter/new geometry; no Sobolev/source retraining; no deployable TTT/controller/selector work; no baseline rerun; no fresh cohort; no official LOL-v2 Real test; no T050 work. If a correctness/provenance failure would require changing scientific code or settings, stop and report `PARTIAL` rather than repairing-and-rerunning in this cycle. An infrastructure retry is allowed only for an identical already-generated command/settings with no completed scientific trajectory and must be documented.

## Acceptance / stop criteria

Fail closed on any source/cohort/artifact/hash mismatch; any preflight normal access; any all-100 reconstruction error above `1e-6`; any non-identity step-0 LUT; any non-monotonic curve; any changed T048 raw/lift coordinate; any inactive-region change; wrong operator order/LR/update count/start count; non-finite state/output; selected-state/output hash inconsistency; metric-before-freeze ordering violation; or independent replay disagreement above its fixed tolerance. Execute at most one valid scientific run. Stop after evaluation + replay + report regardless of whether the +2/+1 dB gate passes; do not launch a rescue experiment.

## Expected evidence

Append exactly one T049-A completion report (or `PARTIAL`) to `coordination/CODEX_TO_CHATGPT.md` without rewriting prior reports, and update PR #74 with the evidence commit. Include tested/evidence SHA; exact T048/cohort/source bindings; all-100 low-only preflight proof and maximum reconstruction error; proof all T048 coordinates stayed bit-exact and only LUT `q` trained; exact LUT equations/operator placement and step-0 identity; optimizer/device/update/start settings; freeze timestamps/hashes; T048 vs T049 mean/median PSNR and RGB-SSIM; paired deltas and win/equal/loss; best-step histogram; curve/segment descriptive statistics; gaps to frozen Retinexformer/SNR-Aware development anchors; independent replay state/metric/scalar check counts and maximum errors; failures/deviations; explicit `REFERENCE_ORACLE_ONLY`, zero deployable changes and zero official-test access; and exactly one frozen verdict string. Do not modify `coordination/PROJECT_STATE.md`.
