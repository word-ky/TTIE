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
2. derive a label-free degradation/correction signal from current pixels only;
3. use a compact global or spatial ISP state to produce bounded correction parameters;
4. adapt only the fast ISP state from that signal;
5. render a coarse spatial correction field and corrected image;
6. after objective safety is established, evaluate with a frozen detector.

The codebase still uses direct episodic optimization of a coarse ISP grid. Learned degradation tokens, a learned fast network `F_W`, detector coupling, meta-learned initialization and ViT3-style fast weights remain later targets.

## Established controlled findings

- **T001:** episodic label-free optimization of a bounded spatial ISP field is stable and can strongly outperform one global ISP vector on a favorable heterogeneous exposure toy.
- **T002:** the gain is conditional. Spatial freedom helps low-frequency heterogeneous shifts under a compatible objective, provides essentially no advantage on homogeneous shifts, and can become neutral/harmful for high-frequency or content-confounded cases. `uniform96` rules out a trivial raw-latent-count explanation, but not every equal-function-capacity global model.
- The absolute patch-mean-to-0.5 objective is not identity-safe; it edits legitimate intrinsic-dark/high-key content while lowering its own loss.
- **T003:** simple identity anchoring + plain TV do not rescue that absolute prior under the fixed safety/utility diagnostic. Strong anchoring only becomes safe by destroying useful correction.
- **T004:** fixed zero-shot absolute frozen-CLIP exposure scores show useful dark sensitivity but poor bright discrimination; cross-image content offsets remain large.
- **T005:** a fixed same-image ±0.25 EV relative CLIP response does not solve the content problem; it underperforms the unchanged absolute score on the same fresh split.
- **T006:** source-trained three-prototype readout on frozen CLIP features produces **strong exposure ranking and spatial localization** on fresh images (dark/bright AUC 0.9870/0.9366; homogeneous correct-type TPR 93%/92%; mixed correct recall 91.25%/95%; 0% mixed wrong-type activation), but the declared gate still fails identity safety because clean all-view false activation is 21% (>15%). Nine of ten criteria pass. The learned vectors move far from the text initialization, so they should be interpreted as a discriminative frozen-CLIP readout, not evidence that original text semantics are preserved.
- T006 therefore isolates the next uncertainty: **representation vs calibration**. The score ranking/type signal appears strong, while the independent per-type/per-view thresholding rule is not clean-content-safe enough.

## Non-negotiable design principles

- Test-time adaptation must never consume test labels, clean targets, degradation masks, gain maps or evaluation metrics.
- Clean references may be used only after adaptation for controlled evaluation, except that disjoint source/calibration data may be used to train/freeze development constants before held-out evaluation.
- Keep identity/no-adaptation and a global counterpart in every major spatial claim.
- Do not infer task utility from lower self-supervised loss alone.
- Spatial capacity should remain compact/coarse rather than a free full-resolution correction tensor.
- Separate **objective quality**, **decision/calibration safety**, and **spatial parameterization** before adding larger fast models/meta-learning.
- After a held-out result motivates a new hypothesis, use a fresh deterministic held-out split for the next decisive audit; do not silently recycle inspected examples as pristine test data.
- Source-trained degradation signals are allowed only with strict source/calibration/evaluation separation; source synthetic labels must never become test-time metadata inputs.

## Milestone M0 — mechanism scaffold

Status: **COMPLETED / T001 ACCEPTED**

Spatial 4x4 reduced evaluation MSE by 75.1% relative to global on the predeclared favorable toy. Controlled mechanism result only.

## Milestone M1 — isolate spatiality and prior confounds

Status: **COMPLETED / T002 ACCEPTED**

Low-frequency spatial heterogeneity can justify a spatial field under a compatible objective, but the absolute midtone prior is unsafe and became the dominant bottleneck.

## Milestone M2 — can simple regularization rescue the absolute prior?

Status: **COMPLETED / T003 ACCEPTED — NEGATIVE DIAGNOSTIC**

No predeclared anchor/TV setting achieves both the required clean-drift reduction and heterogeneous-utility retention. Do not tune the same absolute 0.5 prior further.

## Milestone M3 — degradation-aware semantic test-time objective

Status: **ACTIVE**

### T004 — absolute frozen-CLIP gate

Status: **COMPLETED / ACCEPTED — NEGATIVE STAGE-A DIAGNOSTIC**

PR #4 merged as `8c9680c56ced11d4433908091a4e4575ca04ab1f`.

### T005 — counterfactual relative frozen-CLIP signal

Status: **COMPLETED / ACCEPTED — NEGATIVE STAGE-A DIAGNOSTIC**

PR #5 merged as `0d2146257135c2b0c3568cbf579d6c78d2354114`.

### T006 — source-trained CLIP exposure prototypes

Status: **COMPLETED / ACCEPTED — STRONG READOUT, IDENTITY GATE FAILURE**

PR #6 squash-merged as `5c9d6e61a3f6fdea8d528f26ae6401893bcaf6a2`.

Conclusion: frozen CLIP features support a strong source-trained dark/bright readout and mixed-region localization, but the current independent clean-calibrated activation thresholds generalize with 21% clean FPR. No ISP adaptation is authorized yet.

### T007 — joint clean-abstention calibration

Status: **OPEN**

Goal: freeze the entire T006 representation and test, on a new held-out split, whether an image-level joint clean envelope across five local views/types can reduce false activation without destroying exposure recall/localization. This isolates calibration geometry before investing in new prompt/degradation encoders.

If T007 passes its predeclared identity and recall gates, the next task may finally run the first semantic global-vs-spatial EV+Gamma TTT pilot with direct/discrete-action baselines. If it fails, stop threshold tuning and change the source-side identity/degradation model (e.g. explicit normality/hard-negative training, true CLIP-LIT/CoOp prompt learning, or a non-CLIP degradation encoder).

## Open task

`T007 — Joint Clean-Abstention Calibration for the Frozen T006 Exposure Readout` in `coordination/CHATGPT_TO_CODEX.md`.

## Candidate later components (not approved unless an OPEN task says otherwise)

- First learned-signal global vs spatial EV+Gamma TTT pilot, with direct/discrete correction baselines.
- True CLIP-LIT/CoOp-style learned prompt tokens or a non-CLIP degradation encoder if identity-safe gating still fails.
- Explicit source-side normality/hard-negative objective.
- Frozen detector / object-feature consistency for task-semantic preservation.
- Edge-aware spatial regularization after the semantic objective is validated.
- Stronger effective-function-capacity global baseline.
- Learned low-rank spatial regions instead of a fixed grid.
- Meta-learned fast initialization `W_0`.
- ViT3-style fast-weight model mapping local degradation tokens to ISP actions.
