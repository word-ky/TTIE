# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications/evidence remain in Git history and `research_log/`.

---

# Research-lead review — T048-A accepted; affine coupling is real, but not SOTA-scale capacity

I reviewed PR #73, the completion mailbox, and the fixed implementation/evidence (`preflight.py`, `core.py`, `run.py`, `evaluate.py`, `replay.py`, tests), then squash-merged PR #73 as `152ae5b757df2423a96e003d7185192720a0fd8e`.

The predeclared T048 verdict is correctly positive. With accepted T046 EV/gamma frozen and only common gain + additive lift jointly re-optimized from the exact T047 state, mean PSNR rises from `20.0559720402` to `21.0649799860 dB`; paired T048-minus-T047 improvement is `+1.0090079459 dB` mean / `+0.5583504734 dB` median, passing the frozen `+0.50 / +0.25 dB` gate. Mean RGB-SSIM rises from `0.4142735258` to `0.4566638129`; PSNR improves on 100/100 and SSIM on 97/100. This proves that post-gamma scale/offset interaction is materially important and that T047's frozen-gain marginal probe understated affine-family capacity.

The result still does not satisfy the SOTA sprint. T048 remains `0.4138064 dB` below the training-exposed Retinexformer development anchor and `2.3313499 dB` below SNR-Aware in mean PSNR, while the SSIM gap is much larger. Also, 87/392 selected lift coordinates hit the `+0.20` bound and 12/100 winners are at step 500, so T048 is not a certified affine ceiling. However, do not spend this cycle sweeping affine bounds/budgets: the user's target is a clear 2–3 dB advantage over strong fair baselines, which requires a higher-capacity nonlinear fast-state rather than incremental affine tuning.

The information boundary is accepted. All 100 T047 outputs were reconstructed low-only before any new reference decode with max error `0.0`; EV/gamma stayed bit-exact; only gain raw + lift were trainable with the frozen `0.05 / 0.01` Adam groups; exactly 500 updates/one start were used; outputs were frozen before aggregate metrics; independent replay checked 50,100 states, 200 metrics and 924 scalar quantities with max discrepancy `7.1054e-15` and independent-render error `2.3842e-7`. The run is `REFERENCE_ORACLE_ONLY`; no deployable TTT change and no official-test access occurred. Clean/normal targets, PSNR/SSIM, reference gradients and oracle states remain forbidden in deployable test-time adaptation or checkpoint selection.

---

# OPEN one-hour task — T049-A: compact monotonic tone-LUT marginal-capacity oracle

**Work budget: approximately one hour. One hypothesis only: test whether a compact nonlinear monotonic tone coordinate adds a SOTA-scale capacity increment beyond the accepted T048 affine state.**

## Hypothesis / engineering objective

The remaining gap is unlikely to be closed by another scalar affine adjustment. Test one interpretable nonlinear fast-state: an RGB-shared, per-active-Region2 monotonic piecewise-linear tone LUT. This is a capacity diagnostic only. Do not retrain the learned field or make a deployable claim in this cycle.

## Fixed inputs / settings

Use exactly the accepted T048 100-image cohort, source/provenance bindings, hard Region2 gates/masks, and exact accepted T048 selected EV/gamma/common-gain/lift states. Before any new normal/reference decode, reconstruct all 100 accepted T048 selected outputs from low image + frozen state and require max absolute error `<=1e-6`; freeze all baseline identities/hashes.

Keep **all T048 coordinates bit-exact frozen**. Add exactly one RGB-shared monotonic piecewise-linear LUT per active Region2, with fixed input knots `x_k = k/8`, `k=0..8` (8 segments, 9 knots). Fix output endpoints `y_0=0`, `y_8=1`. Parameterize the eight positive segment increments by unconstrained raws `q_j` as `d_j = softplus(q_j)`, normalize `d_j / sum(d)`, and set interior output knots by cumulative sum. Initialize all eight increments equally so the LUT is exactly identity at step 0. No per-channel LUT and no spatial interpolation beyond the existing hard Region2 assignment.

Apply the LUT **after the accepted T048 corrected value is clamped to `[0,1]` and before hard-gate compositing**:

`exposure -> shifted gamma -> common gain -> additive lift -> identity contrast -> clamp -> monotonic tone LUT -> hard-gate compositing`.

Use linear interpolation between adjacent fixed input knots. Inactive regions must remain exactly the original low image. Step 0 must reproduce accepted T048 output bit-exactly or within `1e-6`.

Optimize only the tone-LUT raws with one fresh Adam per image, fixed `lr=0.03`, exactly `500` updates, one start, full-frame RGB-MSE with the same float32-render / float64-loss conventions as T048. Retain steps `0..500` and select the earliest strict MSE minimum. Freeze all selected LUTs/outputs/hashes before aggregate PSNR/RGB-SSIM.

The **sole scientific gate** is paired T049-minus-T048 PSNR. Call `SOTA-scale monotonic-tone capacity supported` only if **mean ΔPSNR >= +2.00 dB AND median ΔPSNR >= +1.00 dB**. Otherwise call `SOTA-scale monotonic-tone capacity not supported under fixed probe`. Absolute T049 PSNR/SSIM, RGB-SSIM deltas, wins/losses, knot/segment distributions, boundary behavior, best-step histogram, and remaining anchor gaps are descriptive only. Do not relax the gate after seeing results.

## Explicit non-goals

No deployable TTT change; no use of test labels/clean targets/reference metrics/oracle values in deployable inference; no EV/gamma/gain/lift update; no RGB-specific curves; no learned residual network; no denoiser; no local filtering; no geometry change; no source/Sobolev retraining; no controller/selector; no LUT-size, lr, step, bound, or parameterization sweep; no second start; no fresh cohort; no baseline rerun; no official LOL-v2 Real test; no T050 work in this cycle.

## Acceptance / stop criteria

Fail closed on any cohort/source/gate/state/hash mismatch, failure to reconstruct all 100 T048 outputs within `1e-6` before new reference access, any change to T048 coordinates, non-identity step-0 LUT, non-monotonic LUT, wrong 8-segment parameterization/operator order/lr/update count, non-finite state/output, inactive-region change, output/state hash mismatch, or independent-replay disagreement. Execute at most one fixed scientific run. Infrastructure retry is allowed only with identical generated commands/settings and must be documented. Stop after evaluation + independent replay + report regardless of outcome.

## Expected evidence

Append exactly one T049-A completion (or `PARTIAL`) report to `coordination/CODEX_TO_CHATGPT.md` without rewriting prior reports. Include PR/head/tested/evidence SHA; exact T048/cohort/source bindings; all-100 pre-reference reconstruction proof/max error; proof all T048 coordinates stayed bit-exact; exact LUT equations, knot positions, identity initialization and operator placement; trainable-parameter inventory; Adam lr/start/update count; selected LUT/output/state hashes; monotonicity and inactive-region checks; T048 and T049 mean/median PSNR + RGB-SSIM; paired PSNR/SSIM deltas and win/equal/loss; descriptive gaps to the frozen Retinexformer/SNR-Aware anchors; best-step histogram; independent replay check count/max error; failures/deviations; explicit `REFERENCE_ORACLE_ONLY`, zero deployable changes, zero official-test access; and exactly one frozen verdict string. Do not modify `coordination/PROJECT_STATE.md`.
