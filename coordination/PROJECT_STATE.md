# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Processing for Task-Oriented Vision**

## Core scientific question

Can a vision system adapt a **compact spatial correction field per test image**, without test labels or clean targets, so that spatially heterogeneous nuisance degradation is corrected before a frozen downstream task model processes the image?

## Main hypothesis

A single global enhancement/ISP state is insufficient when degradation varies across an image. A compact spatial field can help when heterogeneous degradation is spatially resolvable, but useful test-time adaptation requires an objective that distinguishes nuisance degradation from legitimate scene content. Spatial capacity alone is not enough and can amplify a bad objective.

## Current method abstraction

For a test image `x_t`:

1. obtain fixed local views/tokens describing current-image appearance;
2. use a compact global or spatial ISP state to produce bounded correction parameters;
3. derive a label-free degradation/correction signal from the current image only;
4. adapt only the fast ISP state from that signal;
5. render a coarse spatial correction field and corrected image;
6. later, after objective safety is established, evaluate with a frozen detector.

The current codebase still uses direct episodic optimization of a coarse ISP grid. Learned degradation tokens, a learned fast network `F_W`, detector coupling, meta-learned initialization and ViT3-style fast weights remain later targets.

## Established controlled findings

- **T001:** episodic label-free optimization of a bounded spatial ISP field is stable and can strongly outperform one global ISP vector on a favorable heterogeneous exposure toy.
- **T002:** the gain is conditional. On prior-compatible midtone content, spatial freedom strongly helps low-frequency left/right, quadrant and smooth-gradient shifts; it provides essentially no advantage on homogeneous shifts and can become neutral/harmful for high-frequency or content-confounded cases.
- `uniform96` rules out a trivial raw-latent-count explanation, but it is not an equal effective-function-capacity baseline; no claim against arbitrary equally expressive global nonlinear correction is accepted.
- The absolute patch-mean-to-0.5 self-supervised objective is **not identity-safe**. It edits valid intrinsic-dark and high-key content substantially while its own loss decreases; more spatial freedom can amplify this failure.
- **T003:** simple normalized identity anchoring and plain spatial TV do not rescue that absolute prior under the fixed synthetic diagnostic. No one predeclared setting achieves both >=5x reduction of worst clean-content drift and >=70% retention of heterogeneous utility. Strong anchoring can make the field safer only by destroying useful correction. This is a negative result about the tested objective/regularizers/budget, not a universal impossibility result.
- **T004:** a fixed zero-shot frozen-CLIP exposure gate is insufficient in its **absolute cross-image** form. Dark-vs-clean AUC is 0.836389 but bright-vs-clean AUC is 0.598611. Paired corresponding-score changes remain 100% dark / 90% bright, indicating exposure sensitivity but strong content offsets.
- **T005:** the prescribed same-image ±0.25 EV counterfactual correction-response does **not** cancel the content problem into a useful severity signal. On a fresh held-out split, relative dark/bright AUC is 0.5806 / 0.2945, correct-type TPR 10% / 5%, and active-type precision 71.43%, despite clean FPR of 7%. On those exact same images, the unchanged absolute zero-shot score is materially stronger (AUC 0.8339 / 0.6040). Thus a local finite-difference response under one fixed probe is not automatically a reliable degradation variable. No semantic ISP adaptation has yet been authorized.

## Non-negotiable design principles

- Test-time adaptation must never consume test labels, clean targets, degradation masks, gain maps or evaluation metrics.
- Clean references may be used only after adaptation for controlled evaluation, except that disjoint source/calibration data may be used to train/freeze development constants before held-out evaluation.
- Keep identity/no-adaptation and a global counterpart in every major spatial claim.
- Do not infer task utility from lower self-supervised loss alone.
- Spatial capacity should remain compact/coarse rather than a free full-resolution correction tensor.
- Separate **objective quality** from **spatial parameterization**: first establish a trustworthy degradation signal, then add larger fast models/meta-learning.
- After a held-out result motivates a new hypothesis, use a fresh deterministic held-out split for the next decisive audit; do not silently recycle inspected examples as a pristine test set.
- Source-trained degradation signals are allowed only with strict source/calibration/evaluation separation; source synthetic labels must never become test-time metadata inputs.

## Milestone M0 — mechanism scaffold

Status: **COMPLETED / T001 ACCEPTED**

Evidence: deterministic heterogeneous toy, 13 CPU tests, A6000 CPU/CUDA validation, no-label adaptation boundary verified. Spatial 4x4 reduced evaluation MSE by 75.1% relative to global on the predeclared T001 toy. Controlled mechanism result only.

## Milestone M1 — isolate spatiality and prior confounds

Status: **COMPLETED / T002 ACCEPTED**

Evidence: fixed 504-row matrix, homogeneous/heterogeneous/grid-frequency/content controls and a 96-latent spatially uniform control. Conclusion: low-frequency spatial heterogeneity can justify a spatial field under a compatible objective, but the absolute midtone prior is unsafe and became the dominant bottleneck.

## Milestone M2 — can simple regularization rescue the absolute prior?

Status: **COMPLETED / T003 ACCEPTED — NEGATIVE DIAGNOSTIC**

Evidence: fixed 936-row / 864-episode A6000-host CPU sweep with focused CUDA validation, exact 11-setting anchor/TV sweep and predeclared 5x-drift/70%-utility conjunction. No setting qualifies. PR #3 merged as `862e3401cf28e9d08582996dad2eb974ad5f9617`.

Conclusion: do not spend the next milestone tuning the same absolute 0.5 prior. Move to degradation-aware evidence.

## Milestone M3 — degradation-aware semantic test-time objective

Status: **ACTIVE**

### T004 — absolute frozen-CLIP gate

Status: **COMPLETED / ACCEPTED — NEGATIVE STAGE-A DIAGNOSTIC**

PR #4 squash-merged as `8c9680c56ced11d4433908091a4e4575ca04ab1f`.

Conclusion: fixed zero-shot absolute scores retain useful dark sensitivity but do not provide a reliable two-sided exposure gate, especially for bright/overexposed content.

### T005 — counterfactual relative frozen-CLIP signal

Status: **COMPLETED / ACCEPTED — NEGATIVE STAGE-A DIAGNOSTIC**

PR #5 squash-merged as `0d2146257135c2b0c3568cbf579d6c78d2354114`.

Conclusion: the fixed within-image ±0.25 EV correction-response hypothesis is rejected on its fresh predeclared audit. Relative responses underperform the absolute zero-shot score on the same split and do not justify semantic TTT. Do not post-hoc tune this probe on T005 examples.

### T006 — source-trained CLIP exposure prototypes

Status: **OPEN**

Goal: use disjoint source images with paired synthetic clean/dark/bright exposure to train only three normalized semantic prototypes initialized from existing CLIP text prototypes, while freezing the CLIP image encoder. Then freeze the learned signal and audit it on a fresh held-out split before any ISP adaptation.

The decisive addition is a **mixed spatial localization audit** on left/right and quadrant exposure: a signal that works only for whole-image homogeneous degradation is not sufficient for Spatially Varying TTT-ISP. If T006 passes its predeclared identity, homogeneous discrimination and mixed-region localization gates, the next task may finally run a learned-signal global-vs-spatial EV+Gamma TTT pilot. If it fails, the next branch should test true CLIP-LIT/CoOp prompt-token learning or a non-CLIP degradation encoder rather than tuning T006 on the held-out set.

## Open task

`T006 — Source-Trained CLIP Exposure Prototypes + Fresh Spatial Signal Audit` in `coordination/CHATGPT_TO_CODEX.md`.

## Candidate later components (not approved unless an OPEN task says otherwise)

- True CLIP-LIT/CoOp-style learned prompt tokens on disjoint source data if minimal learned prototypes are insufficient.
- Learned-signal global vs spatial EV+Gamma TTT, with direct/discrete correction baselines.
- Frozen detector / object-feature consistency for task-semantic preservation.
- Edge-aware spatial regularization after the semantic objective is validated.
- Stronger effective-function-capacity global baseline.
- Learned low-rank spatial regions instead of a fixed grid.
- Meta-learned fast initialization `W_0`.
- ViT3-style fast-weight model mapping local degradation tokens to ISP actions.
