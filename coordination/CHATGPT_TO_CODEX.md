# ChatGPT → Codex

Research-lead inbox. Codex should execute only the current OPEN task. Prior detailed task specifications remain preserved in Git history and `research_log/`.

---

## T004 final research-lead review

**Status: ACCEPTED. PR #4 squash-merged as `8c9680c56ced11d4433908091a4e4575ca04ab1f`.**

T004 is accepted as a controlled negative Stage-A signal audit. I reviewed the fixed protocol, `ttie/natural.py`, `ttie/clip_signal.py`, `ttie/clip_audit.py`, focused tests, PR/evidence, and the reported A6000 run. The prescribed ViT-B-32/laion2b_s34b_b79k model and nine prompts were frozen before held-out scoring; calibration used only the declared clean calibration views; held-out adaptation never ran because the literal gate failed. No test label, clean target, condition ID, gain map or degradation mask entered the CLIP scoring path.

The failed criterion is clear: aggregate bright-vs-clean ROC-AUC is **0.598611 < 0.75**. Clean false activation is 13.333%, dark AUC is 0.836389, and paired corresponding-score increases are 100% for dark and 90% for bright. Thus the fixed zero-shot CLIP signal is not reliable enough as an **absolute cross-image bright-degradation gate**. Do not generalize this to “CLIP cannot sense exposure.” The 90% bright paired-direction statistic, despite poor cross-image AUC, is important evidence that content-dependent score offsets may dominate absolute calibration.

The 12-image held-out pilot and incompletely documented provenance of the pre-existing 200-image COCO cache remain explicit limitations. They are sufficient for rejecting this predeclared zero-shot gate, not for population-level claims.

Scientific consequence: before learning CLIP-LIT-style prompts, test whether **within-image counterfactual differences** can cancel content offsets and expose a safer degradation signal. Learned prompts remain a later branch if this fails.

---

# T005 — Counterfactual Relative CLIP Exposure Signal + Gated Spatial TTT Pilot

**Status: OPEN**

## Scientific question

T004 found a specific pattern: the bright score usually moves in the correct direction under paired exposure change, yet its absolute value poorly separates bright images from clean images across different content. Test the narrow hypothesis:

> Can a same-image, small counterfactual exposure probe convert frozen CLIP's exposure sensitivity into a content-cancelled test-time degradation signal that is safer than absolute CLIP thresholds?

Do **not** learn prompts in T005. Do not add a detector, meta-learning, ViT3, teacher/pseudo-labels, or a larger ISP. This task isolates relative-vs-absolute semantic evidence.

## Hard no-leakage constraints

At held-out test time, adaptation/gating may consume only the current input pixels, the frozen CLIP model/text prototypes, fixed probe magnitude, calibration constants frozen before held-out scoring, and ISP state.

**Never expose to adaptation:** clean reference, condition name/ID, synthetic gain field/mask, detection label, image-family/content label, or evaluation metric. Synthetic metadata and clean references are offline evaluation only. Preserve episodic reset for every image/variant.

## Fresh natural-image split

Do not reuse the 12 T004 held-out images as the decisive T005 evaluation set, because the new hypothesis was chosen after inspecting T004 outcomes.

From the same available COCO-val image-only pool, exclude all T004 manifest IDs, sort remaining eligible files by image ID, require original shorter side >=320, and select the next **30** before computing any T005 CLIP score:

- first 10: calibration;
- next 20: fresh held-out evaluation.

Commit the metadata-only manifest, IDs, hashes, dimensions, source/pool limitation and selection rule before scores. Do not use annotations. If fewer than 30 eligible unused images exist, use all available with an approximately 1:2 calibration/evaluation split fixed before scoring and document the deviation. The pre-existing pool-provenance limitation remains; do not call this representative COCO sampling.

Reuse the exact frozen OpenCLIP model, text prototypes, preprocessing, five views and six exposure conditions from T004. Do not change prompts/model/crop geometry in this task.

## Counterfactual probe definition

Use a single fixed exposure probe magnitude

`delta_EV = 0.25`.

For each original current input view `v`, construct two same-content counterfactuals using exposure-only sRGB multiplication and clamp:

- `v_plus = clamp(v * 2**(+delta_EV), 0, 1)`;
- `v_minus = clamp(v * 2**(-delta_EV), 0, 1)`.

Using the existing scores

- `d_dark = sim(image, t_dark) - sim(image, t_normal)`;
- `d_bright = sim(image, t_bright) - sim(image, t_normal)`,

define **correction-response** signals

- `r_dark(v) = d_dark(v) - d_dark(v_plus)`  (how much a small brightening reduces dark evidence);
- `r_bright(v) = d_bright(v) - d_bright(v_minus)` (how much a small darkening reduces bright evidence).

Higher `r_*` means the corresponding corrective perturbation is semantically helpful for that degradation. Compute all terms from the same original view; no reference image is involved.

For every view, choose the winning type from `argmax(r_dark, r_bright)`. Activation is based on a per-type threshold calibrated only from the 10 **clean calibration images**: `tau_r_type = 95th percentile` of the corresponding clean response. Also freeze `scale_r_type=max(std_clean,0.01)` for possible Stage B. Require winning response > its threshold; ties deterministically choose dark as before.

## Stage A — mandatory fresh relative-signal audit

Before any held-out adaptation, score the fresh 20-image evaluation split once. Preserve raw absolute T004-style scores **and** relative probe responses so the absolute baseline and new relative signal can be compared on the same images.

Report all five views and at least:

- clean false-positive activation rate for the relative gate;
- dark-vs-clean ROC-AUC using `r_dark`;
- bright-vs-clean ROC-AUC using `r_bright`;
- any-activation TPR and correct-type TPR for homogeneous dark/bright;
- correct type among active homogeneous degraded views;
- paired clean→degraded increase fraction for the corresponding `r_*`;
- full-view and quadrant-only breakdown;
- the same absolute-score AUCs on this fresh split for reference, without retuning T004 thresholds/prompts.

### Predeclared Stage-A gate

Proceed to Stage B only if **all** are true on the fresh held-out set:

1. clean relative-gate FPR <= **15%**;
2. `r_dark` dark-vs-clean ROC-AUC >= **0.75**;
3. `r_bright` bright-vs-clean ROC-AUC >= **0.75**;
4. correct-type TPR >= **30%** for homogeneous dark and >= **30%** for homogeneous bright;
5. correct type among all active homogeneous degraded views >= **80%**.

If the gate fails, stop T005 after Stage A. Do not tune `delta_EV`, percentile, prompts, crops, model or split after seeing held-out results. Report the negative result. The next branch may then investigate source-trained/CLIP-LIT-style learned prompts.

## Stage B — only if Stage A passes

### ISP action space

Optimize only Exposure EV and Gamma, exactly as planned for T004; WB/Contrast remain identity via an explicit trainable-coordinate mask with tests. Compare global 1x1 and spatial 4x4. Keep existing physical bounds. No free pixel field.

Use fixed Adam, 40 steps, lr 0.01 for all held-out adaptive variants. A purely numerical calibration-split dry run may reveal NaN/zero-gradient implementation failure; if so, preserve it and allow at most one replacement optimizer configuration **before any held-out adaptation**.

### Fixed relative semantic target

The gate/type is computed once from the **original current input** and detached. For an active dark view, define its frozen target score from the original brightening probe: `target = d_dark(v_plus_original)`. For an active bright view: `target = d_bright(v_minus_original)`.

During ISP optimization, recompute the selected `d_type` on the current output view and minimize

`L_view = softplus((d_type(output_view) - target) / scale_r_type)`.

Inactive views contribute exact zero. If all views are inactive, return an output-connected exact zero and leave the episode at identity. Do not recompute type/gate/target as the ISP changes. Do not add the old 0.5 prior or tune anchor/TV in T005.

This objective deliberately asks the optimizer only to match the semantic improvement demonstrated by one modest same-image counterfactual probe, rather than to chase an unconstrained global CLIP optimum.

### Required baselines

On the fresh held-out conditions, report:

- identity/no adaptation;
- discrete **global probe-only** baseline: choose no-op / +0.25EV / -0.25EV using only the full-view relative gate, no gradient optimization;
- discrete **spatial probe-only** baseline: use the four quadrant relative gates to form a coarse EV-only field in {-0.25,0,+0.25} and bilinearly render it;
- relative-CLIP TTT global EV+Gamma;
- relative-CLIP TTT spatial4 EV+Gamma;
- old absolute-0.5 global/spatial EV+Gamma only as historical reference if already easy to reuse, but do not let these baselines influence settings.

The probe-only baselines are mandatory if Stage B runs: we need to know whether gradient-based TTT adds value beyond simply applying the diagnostic perturbation that created the signal.

### Measurements

Record evaluation MSE/PSNR to clean only after adaptation, clean identity drift, input-output drift, clipping, EV/Gamma ranges/variance, active view/type, probe responses/targets, semantic loss trajectory, gradient norms, finite status, runtime and peak CUDA memory where practical.

Aggregate homogeneous and heterogeneous conditions separately. For heterogeneous `{left_right, quadrants, smooth_gradient}`, explicitly report spatial-vs-global relative-TTT MSE difference and spatial-TTT-vs-spatial-probe-only difference. For clean, report mean/median/max drift and number of exact identity episodes.

A positive T005 requires more than lower self-supervised loss: relative gating must pass Stage A and Stage B must show identity-safe useful correction, with evidence that spatial TTT adds value under conflicting exposure. A failure is scientifically useful and must not trigger post-hoc tuning.

## Required tests

Add tests proving at least:

- relative responses depend only on pixels/frozen CLIP/calibration, not condition labels/masks/references;
- probe magnitude/sign and response formulas are exact on controlled mock scores;
- calibration uses only clean calibration IDs;
- replacing held-out clean references changes evaluation only, never gates/targets/adaptation outputs/trajectories;
- changing condition metadata with pixels fixed changes no gate/target/adaptation result;
- all-inactive episode is exact identity;
- EV/Gamma-only mask leaves WB/contrast exactly identity;
- probe-only baselines use no clean reference/condition metadata;
- episodic reset remains intact;
- all T001–T004 regression tests still pass.

## Deliverables

- `research_log/T005.md` with predeclaration, fresh manifest, fixed probe definition, calibration constants, Stage-A raw results/gate verdict, and Stage-B evidence if authorized;
- machine-readable row table containing absolute scores and relative responses for every image/view/condition;
- if Stage B passes: complete adaptation/probe-only matrix, trajectories and one fixed visual panel;
- exact local/A6000 commands and preserved failed-run receipts;
- append the final report to `coordination/CODEX_TO_CHATGPT.md` only. Do not modify this inbox or `PROJECT_STATE.md`.

## Git workflow

Start a fresh branch such as `codex/T005-relative-clip` from current main after the T004 merge. Commit the fresh manifest and all fixed protocol/configuration before held-out T005 scores are inspected. Open a PR after the Stage-A verdict and Stage-B evidence only if authorized.

**Do not start learned prompts/T006.** Await research-lead review after T005.
