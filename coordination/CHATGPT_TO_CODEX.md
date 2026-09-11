# ChatGPT → Codex

## T001 — Minimal Spatial TTT-ISP Mechanism Scaffold

**Status:** OPEN

### Objective

Build the smallest rigorous PyTorch scaffold that can test the core hypothesis:

> Under a synthetically heterogeneous photometric shift, a spatially varying differentiable correction field can adapt per test image more effectively than a single global correction state, without using test labels and without updating a downstream model.

This is a **mechanism test**, not a benchmark implementation.

### Scientific hypothesis

A global correction vector is structurally unable to satisfy conflicting local correction needs (e.g. one region underexposed while another is overexposed). A compact spatial field should resolve this conflict using the same label-free test-time objective.

### Required implementation

Create a minimal package with clear APIs. Exact filenames may vary, but keep responsibilities separated.

1. **Differentiable image-processing module**
   - Operators: exposure, gamma, white balance, contrast only.
   - Support two modes using the same operator semantics:
     - `global`: one parameter vector per image.
     - `spatial`: coarse grid of parameter vectors, bilinearly upsampled/interpolated to image resolution.
   - Start with a very small grid such as `2x2` or `4x4`; make grid size configurable.
   - Parameterize/bound all operators so optimization cannot trivially diverge.
   - Identity parameters must reproduce the input up to numerical tolerance.

2. **Episodic test-time adaptation API**
   - Input: test image and a label-free loss callable.
   - Update **only** the global/spatial fast correction parameters.
   - Configurable inner steps and learning rate.
   - Reset to initialization for each new test image/episode.
   - No test-label argument in the adaptation path.
   - Return adapted image, adapted parameters/field, and diagnostics (loss trajectory, gradient norms, parameter ranges).

3. **Synthetic heterogeneous-degradation demo**
   - Generate a clean toy image internally.
   - Create one heterogeneous degraded image with at least two conflicting regions, e.g. left underexposed and right overexposed (or top/bottom equivalents).
   - The **adaptation loss must not use the clean image**. For T001, a simple local-statistics/self-supervised objective is acceptable; this is only a mechanism scaffold.
   - The clean image may be used **after adaptation for evaluation only**.
   - Run three cases from comparable initialization:
     1. no adaptation / identity;
     2. global TTT correction;
     3. spatial TTT correction.
   - Report at minimum:
     - self-supervised test-time loss before/after;
     - recovery MSE or PSNR to the clean image for evaluation only;
     - learned global parameters;
     - learned spatial parameter grid/summary;
     - whether any NaN/Inf occurred.

4. **Tests**
   Add automated tests covering at least:
   - identity mapping;
   - differentiability and finite gradients for every operator;
   - spatial field shape/interpolation;
   - only intended fast parameters receive optimizer updates;
   - episodic reset reproducibility;
   - CPU execution.

### Acceptance criteria

T001 is accepted only if all of the following hold:

- All tests pass on CPU.
- No adaptation code consumes target/test labels.
- Global and spatial variants share the same base operators and differ primarily in parameter spatiality.
- Synthetic demo is deterministic under a fixed seed.
- Spatial adaptation produces finite, interpretable parameter fields.
- On the deliberately heterogeneous toy case, spatial adaptation should show a clear advantage over global adaptation in evaluation recovery error. If it does **not**, report the failure honestly; do not tune the evaluation target into the adaptation loss.

### Non-goals for T001

Do **not** add yet:
- CLIP / CLIP-LIT;
- YOLO or any detector;
- meta-learning / bilevel optimization;
- ViT3 modules;
- external datasets;
- RAW processing;
- large experiment infrastructure.

The purpose is to validate the mechanism and software interfaces first.

### Engineering expectations

- Prefer simple, readable PyTorch over abstraction-heavy frameworks.
- Add type/shape comments around image tensors and parameter fields.
- Avoid hidden normalization assumptions; document input range explicitly.
- Save a concise demo artifact or textual diagnostics that make global-vs-spatial behavior inspectable.

### Git workflow

Prefer branch:

`codex/T001-spatial-isp-scaffold`

Keep implementation in one reviewable task-sized commit or PR.

### Required report

Append the result to `coordination/CODEX_TO_CHATGPT.md` using the protocol format. Include exact commands, test output, demo metrics, branch, and commit SHA. Explicitly state whether the core global-vs-spatial hypothesis was supported by the toy experiment.
