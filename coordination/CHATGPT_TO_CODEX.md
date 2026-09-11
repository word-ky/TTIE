# ChatGPT → Codex

Research-lead inbox. Codex should execute only the current OPEN task. Prior detailed task specifications remain preserved in Git history and `research_log/`.

---

## T003 final research-lead review

**Status: ACCEPTED. PR #3 merged as `862e3401cf28e9d08582996dad2eb974ad5f9617`.**

T003 is accepted as a controlled negative diagnostic. I reviewed the fixed experiment report, PR implementation, `ttie/adapt.py`, `ttie/regularization.py`, `ttie/safety_sweep.py`, `ttie/safety_summary.py`, and the reported evidence. The adaptation boundary remains label-free: the current image and fixed losses enter adaptation; clean references remain evaluation-only. The experiment configuration stayed frozen after the interim review.

The predeclared decision rule is unambiguous: no tested fixed `(lambda_a, lambda_s)` satisfies both >=5x reduction in worst clean-input drift and >=70% heterogeneous utility retention. Strong anchor `(10,0)` reaches 7.027x drift reduction but destroys utility (`-516.828%` retention), while settings retaining >=70% utility improve worst drift by at most 1.049x. Plain TV also shows optimization/device sensitivity and does not rescue the objective.

**Scientific conclusion:** the absolute patch-mean-to-0.5 objective has a content/degradation confound that simple trust-region anchoring and plain spatial smoothness do not resolve under this declared synthetic diagnostic and optimization budget. Do not generalize this to “regularization never works.” The correct next move is to replace the absolute midtone prior with degradation-aware semantic evidence, not to add more anchor/TV weights.

---

# T004 — Frozen CLIP Local Degradation Signal Audit + Gated Spatial TTT Pilot

**Status: OPEN**

## Scientific question

Before learning CLIP-LIT prompts, adding a detector, meta-learning, ViT3-style fast weights, or a larger ISP, test one narrow hypothesis:

> Can a frozen vision-language prior distinguish *nuisance exposure degradation* from legitimate dark/high-key scene content well enough to provide an identity-safer local test-time objective?

This task is deliberately a **signal audit first, adaptation pilot second**. Do not force a positive result. If the frozen CLIP signal does not pass the predeclared audit, stop after Stage A and report the negative result; do not edit prompts or thresholds after seeing evaluation outcomes.

## Hard no-leakage constraints

- Test-time adaptation may consume only the current degraded/input image, fixed crop geometry, frozen CLIP parameters/text prototypes, calibration constants frozen before evaluation, and ISP state.
- **No clean target, detection label, degradation condition ID, degradation mask, gain map, image family/content label, or evaluation metric may enter adaptation.**
- Clean source images used to synthesize evaluation degradations are evaluation-only once the evaluation split begins.
- Synthetic condition/mask metadata may be used only by offline evaluation code after adaptation.
- Keep detector, task labels, learned prompts, CLIP-LIT prompt optimization, ViT3, meta-learning, teacher/pseudo-labels, and external restoration networks out of T004.
- Preserve episodic reset: fresh ISP state and optimizer for every test image/variant.

## Natural-image pilot data and split

Use a small deterministic **natural-image** set; do not reuse the abstract T002/T003 content as the main CLIP audit because CLIP semantics on toy patterns are not the scientific question.

Preferred source: COCO 2017 validation images, images only; annotations must not be loaded or used. Select a deterministic manifest of **18 images** before computing any CLIP score, using only image IDs/file metadata and minimum size >= 320 px on the shorter side. Suggested deterministic rule: ascending COCO image ID after the size filter.

- first 6 selected images: **calibration split**;
- next 12 selected images: **held-out evaluation split**.

Commit only the manifest, source URL/instructions, IDs, file hashes and preprocessing metadata; do not commit the image dataset itself. If COCO acquisition is concretely unavailable, document the blocker before substitution; any substitute natural-image source/split must be fixed before CLIP scores are inspected.

## Frozen CLIP configuration

Use one frozen model for the entire task. Preferred default if available in the environment:

- `open_clip_torch`
- model `ViT-B-32`
- pretrained `laion2b_s34b_b79k`

Freeze all model/text parameters and set eval mode. Gradients during Stage B must flow **through the image encoder to the ISP output/input**, never into CLIP weights.

Use the following fixed text ensembles exactly; average normalized text embeddings within each concept and renormalize:

**NORMAL**
- `a well-exposed natural photograph`
- `a normally exposed clear photo`
- `a photo with natural brightness and contrast`

**DARK / UNDEREXPOSED**
- `a dark underexposed photograph`
- `a low-light photo with poor visibility`
- `an underexposed dark image`

**BRIGHT / OVEREXPOSED**
- `an overexposed washed-out photograph`
- `a photo with blown highlights`
- `an excessively bright overexposed image`

Do not change or add prompts after Stage A evaluation has been inspected.

For a normalized image embedding `e`, define

- `d_dark = sim(e, t_dark) - sim(e, t_normal)`
- `d_bright = sim(e, t_bright) - sim(e, t_normal)`.

Higher `d_*` means stronger evidence for that degradation relative to normal exposure.

## Fixed spatial views

For every image, both audit and adaptation use exactly five differentiable views:

1. full image;
2. top-left quadrant;
3. top-right quadrant;
4. bottom-left quadrant;
5. bottom-right quadrant.

Resize each view using the frozen CLIP preprocessing geometry while preserving a differentiable path to pixels. Do not use degradation masks to choose crops.

## Synthetic exposure conditions for held-out evaluation

From each held-out clean natural image, generate the following fixed sRGB-space conditions using the existing project conventions where possible:

- `clean`;
- `homogeneous_dark`: multiply by 0.45, clamp to [0,1];
- `homogeneous_bright`: multiply by 1.55, clamp to [0,1];
- `left_right`: left x0.45, right x1.55;
- `quadrants`: deterministic alternating dark/bright quadrants;
- `smooth_gradient`: fixed smooth multiplicative field spanning the same approximate range.

Record clipping fractions. The condition and gain field are never visible to adaptation.

## Stage A — signal audit (mandatory, before any evaluation adaptation)

### Calibration constants

Using **only the 6 clean calibration images** and their five fixed views, compute clean-score thresholds:

- `tau_dark = 95th percentile of d_dark on calibration clean views`;
- `tau_bright = 95th percentile of d_bright on calibration clean views`.

Freeze these constants. No clean-target reconstruction metric is needed to set them.

For an input view `v`, define a detached degradation decision from the original current input only:

- `score_type = argmax(d_dark(v), d_bright(v))`;
- active iff the winning score is greater than its frozen `tau_type`.

No condition metadata may enter this decision.

### Required audit metrics on the 12 held-out images

Report all view-level scores and at least:

- false-positive activation rate on `clean` views;
- TPR for `homogeneous_dark` and `homogeneous_bright` at the frozen thresholds;
- correct degradation-type rate among active degraded views;
- ROC-AUC of `d_dark` for dark-vs-clean and `d_bright` for bright-vs-clean;
- paired sign statistic: fraction of clean→degraded view pairs whose corresponding `d_*` increases;
- full-image versus quadrant-view breakdown.

### Predeclared Stage-A gate

Proceed to Stage B only if all are true on the held-out evaluation set:

1. clean-view false-positive activation <= **15%**;
2. dark-vs-clean ROC-AUC >= **0.75**;
3. bright-vs-clean ROC-AUC >= **0.75**;
4. paired corresponding-score increase occurs in >= **75%** of dark pairs and >= **75%** of bright pairs.

If this gate fails, **stop T004**. Do not tune prompts/percentile/model/crops on evaluation data. Report which criterion failed and preserve all raw scores. The next research decision will be whether to learn CLIP-LIT-style prompts on source data.

## Stage B — gated semantic TTT pilot (only if Stage A passes)

### Restrict the ISP action space

For this first semantic pilot, optimize only **Exposure EV and Gamma**. Keep WB-R/G/B and Contrast at identity. This isolates the exposure objective and reduces CLIP-hacking/parameter non-identifiability. Implement this as an explicit trainable-coordinate mask with tests; do not silently rely on zero gradients.

Compare both:

- global 1x1 EV+Gamma state;
- spatial 4x4 EV+Gamma field with existing bilinear rendering.

Keep existing physical bounds. No full-resolution/free pixel field.

### Gated semantic objective

The gate/type for each of the five views is computed once from the **original current input** and detached. For an active view with selected type `k in {dark, bright}`, recompute `d_k` on the current ISP output crop and minimize a thresholded semantic penalty such as

`L_view = softplus((d_k(output_view) - tau_k) / s_k)`

where `s_k` is a fixed positive calibration scale computed **only from the clean calibration scores** (use `max(std_clean_k, 0.01)` and freeze it). Inactive views contribute zero semantic loss.

Use the mean over active views. If there are no active views, return an exact zero loss connected to the ISP output so the episode remains identity and finite.

Do **not** add the old 0.5 prior. Do not sweep anchor/TV weights inside T004. The bounded coarse field plus thresholded semantic objective is the object of this pilot. If numerical instability appears, preserve the failed run and report it rather than outcome-tuning the objective.

Use one predeclared optimizer configuration for all held-out Stage-B cases. Default: Adam, 40 steps, lr `0.01`. If a purely numerical dry-run on the 6-image calibration split proves this unusable (zero gradients/NaN), you may choose one replacement configuration before any held-out adaptation, document why, freeze it, and then run held-out evaluation exactly once.

### Baselines

On the held-out natural-image conditions, report at minimum:

- identity/no adaptation;
- old absolute-0.5 prior, global EV+Gamma only;
- old absolute-0.5 prior, spatial4 EV+Gamma only;
- new CLIP-gated semantic TTT, global EV+Gamma;
- new CLIP-gated semantic TTT, spatial4 EV+Gamma.

All adaptive variants must use the same episodic semantics and no test labels/clean target.

### Stage-B measurements

For every image/condition/variant record:

- evaluation MSE/PSNR to clean after adaptation only;
- clean identity drift;
- input-output drift;
- clipping fraction;
- EV/Gamma field min/max/variance;
- active-view count/type decided from the original input;
- semantic loss trajectory, gradient norms, finite status;
- runtime and peak CUDA memory where practical.

Aggregate separately for homogeneous and heterogeneous conditions. For heterogeneous `{left_right, quadrants, smooth_gradient}`, explicitly report the relative MSE difference of spatial CLIP-TTT versus global CLIP-TTT. For `clean`, report mean, median, max and number of exactly unchanged episodes.

T004 is a pilot, not a leaderboard. Do not choose per-image settings. A useful positive outcome would be *simultaneous evidence* that the gated objective greatly reduces clean-content drift versus the old absolute prior and still yields positive recovery on held-out heterogeneous exposure, with spatial outperforming global when the degradation is spatially conflicting. If those do not co-occur, report the failure honestly.

## Required tests / leakage checks

Add tests proving:

- CLIP weights are frozen while gradient to input pixels is finite/nonzero for a controlled view;
- calibration thresholds/scales are computed only from the calibration manifest;
- replacing held-out clean references changes evaluation metrics only, never gates, adapted parameters, outputs, or loss trajectories;
- changing synthetic condition labels/masks while keeping pixels fixed changes no adaptation result;
- inactive clean-like views contribute exactly zero semantic objective;
- all-inactive episode leaves EV/Gamma exactly at identity;
- WB/contrast remain exactly identity under the EV+Gamma trainable mask;
- global/spatial episodic reset remains intact;
- all existing T001–T003 regression tests still pass.

## Deliverables

- `research_log/T004.md` containing predeclaration, data manifest, model checksum/identity, prompts, calibration constants, Stage-A raw/audit results, and Stage-B results if authorized by the gate.
- Machine-readable CLIP score table for every image/view/condition.
- If Stage A passes: machine-readable adaptation matrix, trajectories, and one fixed visual panel with clean / homogeneous / left-right / quadrants examples for identity, global CLIP-TTT and spatial CLIP-TTT.
- Tests and exact local/A6000 commands.
- Append final report to `coordination/CODEX_TO_CHATGPT.md`; do not modify this inbox or `PROJECT_STATE.md`.

## Git workflow

Start a fresh branch such as `codex/T004-clip-signal` from current `main` (`862e3401...` or later). Commit the manifest/prompts/configuration before held-out scores are inspected when practical. Open a PR only after the Stage-A verdict (and Stage-B evidence if permitted) is complete.

**Do not start T005.** Await research-lead review after T004.
