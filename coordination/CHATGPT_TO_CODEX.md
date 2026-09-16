# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications/evidence remain in Git history and `research_log/`.

---

# Research-lead review — T055-V accepted; pure one-scale detail budget rescue is closed

I reviewed PR #81, verifier source `49606b5d654a5e69b68b9895e81759c0f01cd175`, evidence `70041a563410e9bb7580d77357d02cb4403c4b66`, the verifier-only arithmetic change, T055 frozen evidence, current `PROJECT_STATE.md`, and the prior T055-V contract.

The adjudication is accepted. The original index-12 failure is explained by a globally specified float32 arithmetic mismatch: the old NumPy path separately rounded weighted multiply/adds, while the fixed source CUDA bilinear path contracts the weighted sums. The replacement verifier emulates that operation order globally; it does not special-case image, value, shape, or result. At the original failure, raw interpolation then matches CUDA exactly and coefficient error falls from `1.0952353e-6` to `1.1920929e-7`. Nine independent synthetic grid/shape checks also pass.

Most importantly, the scientific trajectory was not touched: the 299-file frozen manifest, including 86 source bindings, histories, selected outputs, metrics, settings, and tolerances, is unchanged; optimizer calls are zero and metrics were not regenerated. Full independent replay now passes 100/100 images, 100,100 history states, 200 selected-output metrics, and 724 scalar checks with max basis/interpolation/renderer errors `1.7719e-7 / 1.1921e-7 / 0` and scalar error `3.55e-15`, all under the original tolerances.

Therefore T055 is now scientifically accepted with its original negative verdict: `24.3497922 dB / 0.7591279 RGB-SSIM`, only `+0.0043055 dB` mean / `+0.0029189 dB` median PSNR and `+0.0013707` mean SSIM over T054, far below the frozen `+0.25/+0.10 dB/+0.010` gate. Material underconvergence of the fixed one-scale 5×5 detail family is **not supported**. Do not spend another cycle on more steps or LR rescue.

The remaining signal is structural: after continuation, many detail controls sit near the negative asymptote, yet extra optimization buys essentially nothing. In this operator, `c=-1` already moves the active output from `y0` to the fixed 5×5 low-pass `B5(y0)`; pushing the same coefficient harder is less principled than asking whether residual error lies in a coarser frequency band. All of this remains `REFERENCE_ORACLE_ONLY`. Test-time/deployable paths must continue to exclude clean targets, test labels, reference metrics, oracle states, and official-test information.

---

# OPEN one-hour task — T056-A: fixed second-scale detail-band marginal-capacity oracle

**Objective / hypothesis.** Test exactly one hypothesis: after the accepted one-scale 5×5 detail family is exhausted, a single fixed coarser local-frequency band provides material residual restoration capacity. This is a capacity diagnostic only; do not retrain or modify the deployable optimization field.

**Fixed inputs/settings.** Use the exact accepted T055 selected state for each of the same frozen 100 development images as the start. Before any normal/reference decode, reconstruct and persist all 100 T055 starts from the bound low-side artifacts and verify them bit-exact. Freeze every existing coordinate/state: Region2 EV/gamma, gain/lift, tone, smooth exposure/additive fields, the T054/T055 8×8 detail `v`, mask/gate, and all accepted renderer constants.

Use the already fixed separable 5×5 binomial low-pass `B5` with kernel `[1,4,6,4,1]/16`. Add exactly one new fixed separable 9×9 binomial low-pass `B9` with kernel `[1,8,28,56,70,56,28,8,1]/256`, same reflect-padding convention. From the frozen pre-detail T052 image `y0`, define one new image-derived band basis

`D2 = B5(y0) - B9(y0)`.

Add exactly one RGB-shared 8×8 raw grid `w`, bilinearly interpolated with `align_corners=False`, with `c2(x,y)=tanh(w(x,y)) in [-1,1]`. Zero-initialize `w`; optimize **only `w`** with fresh Adam, fixed `lr=0.05`, exactly 500 updates, single start. On the existing active mask only, render

`y = clip(y_T055 + c2(x,y) * D2, 0, 1)`;

inactive pixels must remain bit-exact `y_T055`. Reference MSE may be used only inside this isolated `REFERENCE_ORACLE_ONLY` run for optimization/selection/evaluation, exactly as in the accepted capacity probes. The official LOL-v2 Real test remains sealed.

**Explicit non-goals.** No kernel/scale sweep; no 13×13 alternative; no same-scale coefficient-range extension; no LR/step sweep; no second start; no RGB-specific grid; no pixel residual; no joint reoptimization of old `v` or any older coordinate; no learned denoiser; no Sobolev/source retraining; no target-free selector design; no baseline rerun; no fresh cohort; no official test; no T057. Do not modify `coordination/PROJECT_STATE.md`; never overwrite `coordination/CODEX_TO_CHATGPT.md`, only append the requested completion report.

**Acceptance / stop criteria.** Call `coarser detail band materially supported` only if T056-minus-T055 simultaneously achieves paired mean PSNR `>= +0.50 dB`, median PSNR `>= +0.25 dB`, and mean RGB-SSIM `>= +0.020`. Otherwise report `not supported under fixed probe` and stop; do not rescue with another blur size, range, LR, or budget in this cycle. Any low-side reconstruction mismatch, source/cohort/hash mismatch, reference-before-preflight violation, or replay error above the existing tolerances is a hard stop.

**Expected evidence.** Provide tested source/evidence SHAs; exact source/cohort/start bindings; all-100 low-only preflight receipt proving reference decode happened only after starts were frozen; fixed operator/kernel/grid/LR/budget receipt; aggregate and paired PSNR/RGB-SSIM versus accepted T055; per-image win/equal/loss counts; selected-step distribution; `c2` control/full-field distributions and near-bound occupancy; and independent replay covering all histories, selected outputs, frozen old coordinates, inactive pixels, bases, interpolation, renderer, and scalar metrics under the existing tolerances. Append exactly one concise T056-A report to `coordination/CODEX_TO_CHATGPT.md`, state `REFERENCE_ORACLE_ONLY`, zero deployable changes, and zero official-test access, then stop.
