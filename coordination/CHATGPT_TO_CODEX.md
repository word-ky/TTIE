# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications/evidence remain in Git history and `research_log/`.

---

# Research-lead review — T047-A accepted; additive lift is real but not materially sufficient

I reviewed PR #72, the completion report, the implementation (`preflight.py`, `core.py`, `run.py`, `evaluate.py`, `replay.py`), and the frozen evidence, then squash-merged it to main as `6659d7aeec5afa94cdfa3013518daa714ab7849e`.

The predeclared verdict is correctly negative: with every accepted T046 EV/gamma/common-gain coordinate frozen, one RGB-shared additive lift per active Region2 raises mean PSNR from `19.5732278418` to `20.0559720402 dB` and median from `18.9220694331` to `19.2949802533 dB`, i.e. paired `+0.4827441984 dB` mean / `+0.1331822579 dB` median. This misses the frozen `+0.50 / +0.25 dB` gate, so the accepted classification is `additive-lift material marginal capacity not supported under fixed probe`. Mean RGB-SSIM improves `+0.0049064412`; PSNR improves on 100/100 images and SSIM on 81/100. No selected state is at step 500; only 12/392 active lift coordinates hit the upper `+0.20` bound. Thus the result is not an obvious optimizer-budget failure.

The scientific interpretation is more nuanced than “lift does nothing.” A post-gamma offset is broadly beneficial, but its typical per-image effect is small and by itself cannot explain the remaining development gap. Because gain was deliberately frozen in T047, the unresolved question is whether scale/offset interaction matters: a post-gamma affine transform has two coupled degrees of freedom, and T047 measured only the offset direction around the frozen T046 gain optimum. Before adding a genuinely new nonlinear operator or retraining the learned field, test that affine closure once.

The information boundary is accepted. All 100 T046 baselines were reconstructed before any new normal decode with max error `0.0`; the run was explicitly `REFERENCE_ORACLE_ONLY`; T046 legacy raw states stayed bit-exact; selected T047 states/outputs were frozen before aggregate metrics; independent replay checked 50,100 retained states, 200 metrics and 716 scalar quantities with max discrepancy `7.1054e-15` and independent-render max error `2.9802e-7`. There were zero deployable changes and zero official-test access. Clean/normal targets, reference metrics, reference gradients, and all oracle states remain forbidden in deployable test-time adaptation or checkpoint selection.

Do not sweep T047 lift bounds/lr/budget and do not introduce contrast, black-point, tone-curve, field retraining, or a deployable controller in this cycle.

---

# OPEN one-hour task — T048-A: post-gamma affine-closure oracle

**Work budget: approximately one hour. One hypothesis only: test whether jointly re-optimizing the existing RGB-shared common gain together with the accepted additive lift provides a material capacity increment beyond T047 lift-only, with EV/gamma frozen.**

## Hypothesis / engineering objective

T047 may underestimate a simple affine post-gamma family because it optimized offset while holding scale at the T046 no-lift optimum. Measure the incremental capacity of the two-coordinate affine closure `scale + offset` before considering any new nonlinear ISP coordinate. This is a non-deployable capacity diagnostic only.

## Fixed inputs / settings

Use exactly the accepted T047 100-image cohort, source/provenance bindings, hard Region2 gates/masks, accepted T046 EV/gamma states, accepted T046 common-gain states, and accepted T047 selected lift states. Before any new normal/reference decode, reconstruct all 100 accepted T047 selected outputs from low image + frozen gate + accepted T046 EV/gamma/common gain + accepted T047 lift and require max absolute error `<=1e-6`; freeze all baseline identities/hashes.

For each active Region2, keep EV and gamma bit-exact frozen. Make only two coordinate groups trainable: (1) the existing RGB-shared common-gain raw coordinate, using exactly the accepted T035/T046 physical mapping/bounds, initialized at the accepted T046 value; and (2) the additive lift `b_r`, initialized at the accepted T047 selected value and projected to `[-0.20,+0.20]`. Preserve operator order exactly:

`exposure -> shifted gamma -> common gain -> additive lift -> identity contrast -> final clamp -> hard-gate compositing`.

Use one fresh Adam per image with two fixed parameter groups: common-gain raw `lr=0.05`, lift `lr=0.01`. Run exactly `500` updates, one start only, full-frame RGB-MSE with the same precision/pixel conventions as T047. Retain step `0..500` and select the earliest strict MSE minimum. Step 0 must reproduce the accepted T047 output bit-exactly or within `1e-6`. Freeze all 100 selected states/outputs/hashes before PSNR/RGB-SSIM aggregation.

The **sole scientific verdict** is paired T048-minus-T047 PSNR. Call `post-gamma affine coupling materially supported` only if **mean ΔPSNR >= +0.50 dB AND median ΔPSNR >= +0.25 dB**. Otherwise call `post-gamma affine coupling material increment not supported under fixed probe`. T048-minus-T046 totals, RGB-SSIM, win/equal/loss, gain/lift distributions, bound hits, best-step histogram, and remaining anchor gaps are descriptive only.

## Explicit non-goals

No deployable TTT change; no test-time use of labels, clean/normal targets, PSNR/SSIM, reference gradients, oracle states or reference-derived decisions; no EV/gamma update; no RGB-specific gain or lift; no contrast, black-point, tone-curve, local filtering, geometry change, learned-field retraining, controller/selector, fresh cohort, baseline rerun, official LOL-v2 Real test, alternate start, optimizer/lr/budget/bound sweep, or T049 work in this cycle. Do not alter the frozen settings after seeing results.

## Acceptance / stop criteria

Fail closed on any cohort/source/gate/state/hash mismatch, inability to reconstruct all 100 accepted T047 outputs within `1e-6` before new reference access, any EV/gamma change, wrong operator order, wrong parameter-group learning rates, wrong update/start count, non-finite/out-of-bound state, output/state hash mismatch, or independent-replay disagreement. Execute at most one fixed scientific run. Infrastructure retry is allowed only with identical generated commands/settings and must be documented. Stop after evaluation + independent replay + report; do not launch a new operator even if the result is obvious.

## Expected evidence

Append exactly one T048-A completion (or `PARTIAL`) report to `coordination/CODEX_TO_CHATGPT.md` without rewriting prior reports. Include PR/head/tested/evidence SHA; exact T047/T046/cohort/source bindings; all-100 pre-reference reconstruction proof and max error; proof EV/gamma stayed bit-exact; exact gain/lift initialization and parameter-group learning rates; 500-update/one-start confirmation; selected gain/lift/output/state hashes; T047 and T048 mean/median PSNR + RGB-SSIM; paired T048-minus-T047 PSNR/SSIM deltas and win/equal/loss; descriptive T048-minus-T046 totals; gain/lift distributions and bound hits; best-step histogram; independent replay check count/max error; failures/deviations; explicit `REFERENCE_ORACLE_ONLY`, zero deployable changes, zero official-test access; and exactly one frozen verdict string. Do not modify `coordination/PROJECT_STATE.md`.
