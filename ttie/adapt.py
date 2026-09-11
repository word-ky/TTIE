"""Episodic adaptation: only fresh ISP fast parameters are optimized."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .isp import ISP, physical_parameters


def local_statistics_loss(image: Tensor) -> Tensor:
    """Fixed gray/midtone prior: RGB means of nonoverlapping 8x8 patches -> 0.5.

    This is a label-free toy prior, not an estimate from a clean target.
    It does not preserve texture or distinguish dark objects from underexposure.
    """
    return (F.avg_pool2d(image, 8) - 0.5).square().mean()


@dataclass
class AdaptationResult:
    image: Tensor
    raw_parameters: Tensor
    parameter_grid: Tensor
    parameter_field: Tensor
    diagnostics: dict


def adapt(image: Tensor, loss_fn: Callable[[Tensor], Tensor], *, mode: str = "global",
          grid_size: tuple[int, int] = (4, 4), steps: int = 200,
          lr: float = 0.03) -> AdaptationResult:
    """Reset ISP and Adam to identity/empty state for each [1,3,H,W] episode.

    Only image and a label-free callable enter this path. autograd.grad requests
    only the raw ISP derivative, leaving input/loss-module .grad fields untouched.
    """
    source = image.detach()
    model = ISP(mode, grid_size).to(device=source.device, dtype=source.dtype)
    optimizer = torch.optim.Adam([model.raw], lr=lr)
    losses, gradients, ranges = [], [], []
    finite = True
    for step in range(steps + 1):
        output = model(source)
        loss = loss_fn(output)
        grid = physical_parameters(model.raw)
        losses.append(loss.detach().item())
        ranges.append({"min": grid.detach().amin(dim=(0, 2, 3)).tolist(),
                       "max": grid.detach().amax(dim=(0, 2, 3)).tolist()})
        finite &= bool(torch.isfinite(output).all() & torch.isfinite(loss) & torch.isfinite(grid).all())
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        gradient, = torch.autograd.grad(loss, model.raw)
        gradients.append(gradient.norm().item())
        finite &= bool(torch.isfinite(gradient).all())
        model.raw.grad = gradient
        optimizer.step()
    return AdaptationResult(output.detach(), model.raw.detach().clone(), grid.detach(),
                            model.parameter_field(source.shape[-2:]).detach(),
                            {"loss_trajectory": losses, "gradient_norms": gradients,
                             "parameter_ranges": ranges, "all_finite": finite,
                             "steps": steps, "lr": lr, "optimizer": "Adam",
                             "reset": "fresh identity ISP and fresh optimizer per call",
                             "parameter_count": model.raw.numel()})
