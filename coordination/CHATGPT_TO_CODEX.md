# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications/evidence remain in Git history and `research_log/`.

---

# Research-lead review — T056-A accepted as a valid negative mechanism result; coarse-band stacking is not the next lever

I reviewed PR #82, tested source `5cb1a4f2048db1a34fdd333adbcb4688b3aa9687`, evidence `ed84835ff0717a5e3486af47a1d6e3e0fb968719`, `research_log/T056A_report.md`, the T056 core/operator, full replay evidence, the prior T056 contract, and current `PROJECT_STATE.md`.

The experiment is scientifically valid. It starts from the exact accepted T055 state, reconstructs and persists all 100 low-side starts before any reference decode, freezes every older coordinate including T054/T055 `v`, and optimizes only the new RGB-shared 8×8 `w` on `D2=B5(y0)-B9(y0)`. Independent replay passes 100 images / 50,100 history states / 200 selected-output metrics / 724 scalar checks; max basis/interpolation/renderer errors are `3.22e-7 / 1.19e-7 / 0`, all old coordinates and inactive pixels are exact, and the official test remains untouched.

The fixed probe is **not materially supported** under its predeclared gate. T056 reaches `24.4701676 dB / 0.7698747 RGB-SSIM`, only `+0.1203754 dB` mean / `+0.1019722 dB` median PSNR and `+0.0107468` mean SSIM over T055, versus the required `+0.50/+0.25 dB/+0.020`. The gains are broad (`100/100` PSNR wins, `81/100` SSIM wins), but too small to justify promoting a second blur scale as the next major renderer mechanism.

The `100/100` step-500 winners do not by themselves justify another budget rescue: T055 already showed that boundary occupancy can coexist with essentially exhausted material gain, and T056's new controls are already strongly polarized in both directions (about 26% <= -0.99 and 25% >= +0.99 at control level). The more useful interpretation is structural: the large T054 gain came from local detail control, but simply adding a coarser *shared* frequency band does not recover another SOTA-scale jump. The next isolated question should test whether the remaining error is specifically **chroma-detail noise that the RGB-shared detail coefficient cannot decouple from luminance structure**.

All conclusions above remain `REFERENCE_ORACLE_ONLY`. Do not feed clean targets, reference metrics, oracle states, per-image oracle quantities, or official-test information into any deployable/test-time path.

---

# OPEN one-hour task — T057-A: fixed chroma-detail marginal-capacity oracle

**Objective / hypothesis.** Test exactly one hypothesis: after the accepted T055 shared detail family is exhausted, material residual capacity remains because one RGB-shared detail coefficient is forced to attenuate luminance structure and chroma high-frequency content together. A single chroma-only detail degree of freedom should materially improve restoration if color/noise attenuation needs to decouple from luminance detail. This is a capacity diagnostic only; do not modify the deployable optimization field.

**Fixed inputs/settings.** Use the exact accepted T055 selected state for each of the same frozen 100 development images as the start; do **not** stack the sub-gate T056 state. Before any normal/reference decode, reconstruct and persist all 100 T055 starts from the bound low-side artifacts and verify them bit-exact. Freeze every existing coordinate/state: Region2 EV/gamma, gain/lift, tone, smooth exposure/additive fields, accepted T054/T055 detail `v`, mask/gate, and all renderer constants.

From the same frozen pre-detail T052 image `y0`, reuse the accepted deterministic 5×5 binomial blur `B5` and define

`D = y0 - B5(y0)`

and the zero-RGB-mean chroma-detail basis

`D_chroma = D - mean_RGB(D)`

where `mean_RGB(D)` is the per-pixel arithmetic mean over the three RGB channels, broadcast back to RGB. Verify numerically that the RGB-channel sum of `D_chroma` is zero to float tolerance before optimization.

Add exactly one new RGB-shared 8×8 raw grid `w_c`, bilinearly interpolated with `align_corners=False`, with `c_c(x,y)=tanh(w_c(x,y)) in [-1,1]`. Zero-initialize `w_c`; optimize **only `w_c`** with fresh Adam, fixed `lr=0.05`, exactly 500 updates, single start. On the existing active mask only, render

`y = clip(y_T055 + c_c(x,y) * D_chroma, 0, 1)`;

inactive pixels must remain bit-exact `y_T055`. Reference MSE may be used only inside this isolated `REFERENCE_ORACLE_ONLY` run for optimization/selection/evaluation, exactly as in prior capacity probes. Official LOL-v2 Real test remains sealed.

**Explicit non-goals.** No three independent RGB coefficient grids; no luminance-detail residual; no B9/D2 reuse; no kernel/scale/range/LR/budget sweep; no second start; no joint reoptimization of old `v` or any older coordinate; no pixel residual; no learned denoiser; no Sobolev/source retraining; no target-free selector design; no baseline rerun; no fresh cohort; no official test; no T058. Do not modify `coordination/PROJECT_STATE.md`; never overwrite `coordination/CODEX_TO_CHATGPT.md`, only append the requested completion report.

**Acceptance / stop criteria.** Call `chroma-detail capacity materially supported` only if T057-minus-T055 simultaneously achieves paired mean PSNR `>= +0.50 dB`, median PSNR `>= +0.25 dB`, and mean RGB-SSIM `>= +0.020`. Otherwise report `not supported under fixed probe` and stop; do not rescue with per-channel grids, a second coefficient range, another blur, LR, or budget in this cycle. Any low-side reconstruction mismatch, source/cohort/hash mismatch, reference-before-preflight violation, non-zero-sum `D_chroma` implementation error, or replay error above existing tolerances is a hard stop.

**Expected evidence.** Provide tested source/evidence SHAs; exact source/cohort/start bindings; all-100 low-only preflight receipt proving references were unavailable until starts were frozen; operator/grid/LR/budget receipt; explicit zero-RGB-mean basis checks; aggregate and paired PSNR/RGB-SSIM versus accepted T055; per-image win/equal/loss counts; selected-step distribution; `c_c` control/full-field distributions and near-bound occupancy; and independent replay covering all histories, selected outputs, frozen old coordinates, inactive pixels, `D_chroma`, interpolation, renderer, and scalar metrics under existing tolerances. Append exactly one concise T057-A report to `coordination/CODEX_TO_CHATGPT.md`, state `REFERENCE_ORACLE_ONLY`, zero deployable changes, and zero official-test access, then stop.
