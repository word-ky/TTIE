# TTIE

Research workspace for **Spatially Varying Test-Time Image Enhancement / ISP Adaptation**.

This repository also serves as a coordination channel between ChatGPT (research lead) and Codex (engineering lead).

## Coordination

Read these files first:

- `coordination/PROTOCOL.md` — communication and ownership rules.
- `coordination/PROJECT_STATE.md` — current research hypothesis and milestone state.
- `coordination/CHATGPT_TO_CODEX.md` — tasks/instructions from ChatGPT to Codex.
- `coordination/CODEX_TO_CHATGPT.md` — implementation reports from Codex to ChatGPT.

## Current research direction

The core hypothesis is that real visual degradation can be **spatially heterogeneous within a single image**, so a single global enhancement/ISP state may be fundamentally insufficient. We investigate a compact, differentiable, spatially varying correction field that is **adapted per test image without test labels**, while the downstream task model remains frozen.

## T001 mechanism scaffold

From the repository root, with Python 3.12 and PyTorch/Pillow installed:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m ttie.demo --device cpu
python -m ttie.demo --device cuda:0 --output research_log/artifacts/T001_cuda
```

The API processes one floating RGB image `[1,3,H,W]` in `[0,1]`, with no hidden normalization or linear/sRGB conversion. The toy uses dimensions divisible by eight. The same `ISP` class implements global (1x1) and spatial (default 4x4) physical parameter grids. `--grid 2 2`, `--steps`, `--lr` and `--seed` configure the demo.

The six coordinates represent four operators, in this order:

1. Exposure: multiply by `2**EV`, `EV = 2*tanh(raw_EV)`.
2. Gamma: `(x+eps)**gamma - eps**gamma`, `eps=1e-6`.
3. White balance: multiply each RGB channel by its gain.
4. Contrast: `0.5 + contrast*(x-0.5)`, then clamp output to `[0,1]`.

Gamma, three WB gains, and contrast use `exp(log(2)*tanh(raw))`, giving bounds `[0.5,2]`. All raw zeros give identity. Physical grid values are bilinearly interpolated with `align_corners=False`; global and constant spatial grids have identical operator semantics. Clipping is differentiable almost everywhere but saturated pixels can have zero gradients.

```python
from ttie.adapt import adapt, local_statistics_loss
result = adapt(image, local_statistics_loss, mode="spatial", grid_size=(4, 4), steps=200, lr=0.03)
```

Every call creates fresh identity fast parameters and a fresh Adam optimizer (default betas/epsilon, no weight decay). Only those parameters receive optimizer updates; the input is detached. The caller supplies a scalar, label-free differentiable loss callable. Returned results include image, raw grid, physical grid, full field and all step losses, gradient norms, physical ranges, and finiteness diagnostics. No detector is implemented in T001.

The fixed objective is `mean((avg_pool2d(output, 8)-0.5)**2)`: patch RGB means toward a gray midtone prior. Global and spatial share this exact objective, initialization, bounds, optimizer, 200 updates and learning rate 0.03; only the grid size/parameter count differs (6 versus 96). The toy is seeded periodic RGB texture around 0.5, multiplied by 0.45 on the left and 1.55 on the right, then clipped. Both adaptation runs finish before clean-reference MSE/PSNR evaluation. A test replaces the clean evaluation reference while keeping degraded input fixed and verifies unchanged adaptation outputs/trajectories.

The demo saves `metrics.json` (full diagnostics and parameter grids), `tensors.pt` (float images and fields), and `comparison.png`. PNGs are rounded for viewing only; MSE/PSNR use float RGB with data range 1. Evaluation does not select checkpoints: final fixed-step outputs are reported.

This is one deliberately controlled mechanism experiment. The statistical prior suits this synthetic image and can flatten real texture, remove genuine color, or brighten genuinely dark objects. Gamma/exposure/WB/contrast can compensate for each other, so individual learned operator values do not uniquely identify the degradation. Bilinear fields cannot exactly reproduce the abrupt exposure boundary. No natural-image, detector, benchmark, or general research-hypothesis validation is claimed.

Remote setup, run IDs and recovery instructions are recorded in `research_log/REMOTE.md`; test receipts and experiment artifacts remain under `research_log/`.

## T002 controls

```bash
python -m ttie.suite --device cpu --output research_log/artifacts/T002_cpu
python -m ttie.summarize research_log/artifacts/T002_cpu
```

The full fixed suite uses seeds 7/11/23, three intrinsic content families, eight illumination conditions and seven variants, for 504 rows. Configuration is predeclared in `research_log/T002.md`: the same 200-step Adam, lr 0.03, bounds and label-free patch prior apply everywhere. `--device cuda:0` runs on GPU; `--seeds`, `--steps` and `--lr` are explicit diagnostic overrides, not condition-dependent selectors.

`ISP("uniform_control", (4,4))` and `adapt(..., mode="uniform_control")` optimize 16 raw six-vectors initialized identically at zero. Their arithmetic mean is taken before the bounded physical mapping and then rendered uniformly over the image. Returned `raw_parameters` has 96 coordinates; `parameter_grid` is the single rendered physical vector. The global/spatial APIs and defaults are unchanged. The uniform control has 96 optimization scalars but only six effective output degrees of freedom; symmetric latents remain equal. Adam epsilon permits small numerical differences from the six-scalar global baseline. This comparison isolates raw parameter count, not arbitrary model expressivity.

The suite writes compact `metrics.csv` / `metrics.json`, config.json, seed-7 representative tensors and full adaptation diagnostics, fixed-scale output/EV-field panels, and seed-mean summary.json/summary.md. Every row includes self-supervised losses, clean-reference recovery, region errors, clipping, input-output drift, physical field variances and parameter count. `identity_drift_mse` is populated only for the undegraded clean condition; exact-zero MSE has infinite PSNR represented as JSON null / CSV blank. Evaluation is performed after every variant has completed adaptation for that input. Input-only `adapt_variants` receives no clean reference, masks, family identifier or degradation parameters.

Three seeds vary toy noise only. High-frequency stripes can exceed both the field grid's resolution and the fixed 8x8 loss's spatial resolution. Interpret those effects together; the suite does not separate them causally and does not claim real-image generalization.

## T003 identity-anchor / TV diagnostic

```bash
python -m ttie.safety_sweep --device cpu --output research_log/artifacts/T003_cpu
python -m ttie.safety_summary research_log/artifacts/T003_cpu --plot
```

`adapt` accepts optional `lambda_a` and `lambda_s`, both zero by default. The coarse physical grid is normalized as EV/2 and log2 of gamma/WB/contrast. The anchor is mean squared normalized correction; TV sums mean absolute horizontal and vertical coarse-grid differences (zero for absent axes). Global mode's rendered grid is 1x1, so TV has no effect. The loss is the original callable plus weighted anchor and TV; no clean pixels enter any term. Diagnostics include separate prior/anchor/TV trajectories and the existing total/gradient trajectories. Old zero-weight behavior is covered by numerical regression.

The fixed T003 sweep retains the exact T002 inputs and includes all 11 research-lead-specified weight settings on spatial4, plus unregularized global and identity: 936 rows / 864 reset adaptation episodes. The complete matrix runs on CPU on the A6000 host; separate CUDA repeat/device-sensitivity tests are reported explicitly. Matplotlib is used only for local plotting from completed metrics. No per-example configuration is selected.

Every run's component, total and gradient trajectories are saved under `trajectories/`; metrics.csv/metrics.json preserve all input-level measurements. `summary.json`/`summary.md` implement the predeclared conjunction: worst drift across nine clean inputs must fall at least 5x, and the ratio of mean heterogeneous improvements over global must be at least .70. Mean drift and worst family-mean drift are also reported. Negative utility retention is retained. `--plot` produces labeled Pareto PNG/SVG figures. The image/EV panel always shows the predeclared (0,0), (.1,.1), (1,.1), (10,0) settings on the same three fixed cases, irrespective of outcomes. Full rationale and outcomes are in `research_log/T003.md`.

## T004 frozen CLIP signal audit

```bash
python -m ttie.clip_audit --manifest research_log/T004_manifest.json \
  --images /path/to/selected/images --model-identity /path/to/model_identity.json \
  --output research_log/artifacts/T004 --device cuda:0
```

`requirements-clip.txt` pins OpenCLIP and torchvision for the existing remote torch2.4 environment. The manifest fixes18 COCO images from an available200-image cache, split6calibration/12held-out using file metadata alone. Images/checkpoints remain outside Git. See `research_log/T004.md` for selection/provenance and `T004_model_identity.json` for the official checkpoint identity and hash.

Frozen ViT-B-32/laion2b_s34b_b79k uses the nine task-specified prompts and five fixed differentiable views. Only clean calibration IDs set the95th-percentile thresholds and standard-deviation scales. Held-out scoring covers six exposure conditions, writes every view score/activation to CSV/JSON, and reports aggregate/full/quadrant AUC, paired score changes, false activation and detection/type rates. A real pretrained gradient/freeze check uses a calibration image before held-out scoring. The literal Stage-A conjunction controls whether a later adaptation pilot is authorized; a failed gate stops the task without prompt/threshold/model tuning. The audit entry point never launches adaptation automatically.

## T005 relative CLIP audit

```bash
python -m ttie.relative_audit --manifest research_log/T005_manifest.json \
  --images /path/to/t005/images --model-identity /path/to/model_identity.json \
  --absolute-calibration /path/to/completed/T004/calibration.json \
  --output research_log/artifacts/T005 --device cuda:0
```

The fresh 30-image metadata-only manifest excludes every T004 image, with10clean calibration/20held-out sources. Model, prompts, views and conditions remain fixed. A single±0.25EV pixel probe defines dark response as d_dark(original)-d_dark(brightened), and bright response as d_bright(original)-d_bright(darkened). Only fresh clean calibration responses determine relative thresholds/scales. The audit preserves all original/probe scores, responses, fixed targets, clipping and activations, and compares relative versus absolute signals on the same held-out images using unchanged T004 absolute thresholds. The all-view gate requires clean FPR<=15%, both AUC>=.75, both correct-type TPR>=30%, and combined active-type precision>=80%. Failure stops before adaptation; no magnitude/prompt/threshold tuning. Full protocol and evidence are in research_log/T005.md.
