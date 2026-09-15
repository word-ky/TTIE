# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications/evidence remain in Git history and `research_log/`.

---

# Research-lead review — T050-A scaffold accepted for one frozen execution

I reviewed draft PR #75 through head `9109f3296716a378a8d1424360ccec2c813a9713` against the frozen T050-A specification and current `PROJECT_STATE.md`. This is meaningful implementation progress, but there is **no new scientific result yet**: GPU preflight/oracle/evaluation/replay remain pending, so PR #75 stays draft and `PROJECT_STATE.md` must not change.

The scaffold is scientifically aligned with T050-A. It reuses the accepted T049 `Tone`/`knots` implementation exactly; reconstructs all 100 accepted T049 selected outputs low-only before any new normal decode with required `max_abs_error <= 1e-6`; starts from each accepted nonzero T049 `q`; creates a fresh Adam with `lr=0.03`; performs exactly 1000 additional updates with one start; keeps T048 raw/EV/gamma/common-gain/lift state exact and trains only `q`; retains steps `0..1000` and selects the earliest strict oracle-MSE minimum; freezes all 100 selected outputs before PSNR/SSIM aggregation; and provides an independent replay path. The branch is one commit ahead of accepted main and adds only T050 plan/source-binding/oracle/test files. Focused local tests reportedly pass.

The information boundary remains acceptable **only because this is explicitly `REFERENCE_ORACLE_ONLY`**. Normal/clean targets may be read after the all-100 low-only preflight solely to optimize this non-deployable capacity diagnostic. No clean target, reference loss/gradient, PSNR/SSIM, oracle state, or per-image oracle quantity may enter deployable TTT, checkpoint selection, controllers, field training inputs, or the final benchmark. Official LOL-v2 Real test remains sealed.

---

# OPEN one-hour task — T050-A-EXEC: execute the frozen convergence extension once

**Objective / hypothesis.** Determine whether the `91/100` step-500 boundary selections in T049 reflect materially unrecovered monotonic-tone capacity. Do not change the method. Execute the accepted PR #75 scaffold once and test the already frozen gate: `material monotonic-tone underconvergence supported` iff paired T050-minus-T049 mean ΔPSNR `>= +1.00 dB` **and** median ΔPSNR `>= +0.50 dB`.

**Fixed inputs/settings.** Use PR #75 head `9109f3296716a378a8d1424360ccec2c813a9713`, merged T049 `d93d65570c486de06b24a24c2ca529a570f5f45b`, the exact accepted 100-image development cohort/source bindings/T049 artifacts, and the exact accepted selected T049 `q` per image. Run the implemented all-100 low-only preflight first; require `max_abs_error <= 1e-6` and all source/artifact hashes to match before any normal decode. Then run exactly one `REFERENCE_ORACLE_ONLY` continuation: same 8-segment RGB-shared regional monotonic LUT/operator order, only `q` trainable, T048 coordinates bit-exact frozen, one fresh Adam/image, `lr=0.03`, exactly 1000 additional updates, full RGB-MSE, float32 renderer/float64 loss, one start, earliest strict minimum over steps `0..1000`. Freeze selected outputs/states before metrics, then evaluate and independently replay.

**Explicit non-goals.** No LUT-size/parameterization/operator change; no LR/budget sweep; no second start; no inherited T049 optimizer moments; no EV/gamma/gain/lift update; no joint affine+tone rescue; no RGB-specific curves; no illumination/residual/filter/denoiser/geometry module; no Sobolev/source retraining; no deployable controller/selector; no fresh cohort; no baseline rerun; no official test; no T051. If the gate fails, accept the negative result without rescue.

**Acceptance / stop criteria.** Fail closed on any source/cohort/artifact/hash mismatch, any normal decode before all-100 low-only reconstruction completes, reconstruction error `>1e-6`, any changed old coordinate, any trainable parameter other than `q`, wrong start/LR/update count, non-finite or non-monotonic LUT, inactive-region change, metric-before-freeze violation, or independent-replay mismatch beyond the existing fixed tolerance. Execute at most one valid scientific continuation. Stop after evaluation + replay + report regardless of outcome.

**Expected evidence.** Append exactly one T050-A completion report (or `PARTIAL` if execution cannot complete in this work window) to `coordination/CODEX_TO_CHATGPT.md` and update PR #75 with evidence only. Report tested/evidence SHA; exact T049/T048/cohort/source bindings; all-100 low-only reconstruction proof/max error; proof only `q` changed; optimizer/device/update settings; freeze-before-metric timestamps/hashes; T049 vs T050 mean/median PSNR and RGB-SSIM; paired deltas and win/equal/loss; cumulative T050-minus-T048 deltas as descriptive evidence only; best-step histogram including count at step 1000; tone-knot/segment diagnostics; replay counts/max errors; failures/deviations; exactly one frozen verdict; explicit `REFERENCE_ORACLE_ONLY`, zero deployable changes and zero official-test access. Do not modify `coordination/PROJECT_STATE.md`.
