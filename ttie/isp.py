"""Shared bounded ISP for one float RGB image [1, 3, H, W] in [0, 1]."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


PARAMETER_NAMES = ("exposure_ev", "gamma", "wb_r", "wb_g", "wb_b", "contrast")


def physical_parameters(raw: Tensor) -> Tensor:
    """Map unconstrained [1,6,h,w] values to EV [-2,2], others [0.5,2]."""
    bounded = raw.tanh()
    return torch.cat((2 * bounded[:, :1], (math.log(2) * bounded[:, 1:]).exp()), dim=1)


class ISP(nn.Module):
    def __init__(self, mode: str = "global", grid_size: tuple[int, int] = (4, 4)):
        super().__init__()
        if mode not in ("global", "spatial", "uniform_control"):
            raise ValueError("mode must be global, spatial or uniform_control")
        self.mode = mode
        size = (1, 1) if mode == "global" else grid_size
        self.raw = nn.Parameter(torch.zeros(1, 6, *size))

    def physical_grid(self) -> Tensor:
        """Uniform control averages raw latent vectors before applying bounds."""
        raw = self.raw.mean(dim=(-2, -1), keepdim=True) if self.mode == "uniform_control" else self.raw
        return physical_parameters(raw)

    def parameter_field(self, size: tuple[int, int]) -> Tensor:
        """Bilinearly interpolate bounded physical grid values to [1,6,H,W]."""
        return F.interpolate(self.physical_grid(), size=size, mode="bilinear", align_corners=False)

    def forward(self, image: Tensor) -> Tensor:
        field = self.parameter_field(image.shape[-2:])
        exposure, gamma = field[:, :1], field[:, 1:2]
        wb, contrast = field[:, 2:5], field[:, 5:6]
        corrected = image * torch.exp2(exposure)
        # Shifted power keeps the black point at zero with finite gamma gradients.
        eps = 1e-6
        corrected = (corrected + eps).pow(gamma) - torch.pow(eps, gamma)
        corrected = corrected * wb
        corrected = 0.5 + contrast * (corrected - 0.5)
        return corrected.clamp(0, 1)
