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
