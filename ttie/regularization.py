"""Input-only penalties on the coarse physical ISP correction state."""

import torch
from torch import Tensor


def normalized_correction(physical_grid: Tensor) -> Tensor:
    """[1,6,h,w]: EV/2, log2(gamma/WB/contrast), identity exactly zero."""
    return torch.cat((physical_grid[:, :1] / 2, physical_grid[:, 1:].log2()), dim=1)


def correction_penalties(physical_grid: Tensor) -> tuple[Tensor, Tensor]:
    correction = normalized_correction(physical_grid)
    anchor = correction.square().mean()
    # 1x1/global grids have no adjacent pairs; their TV is identically zero.
    smooth = correction.sum() * 0
    if correction.shape[-1] > 1:
        smooth = smooth + correction.diff(dim=-1).abs().mean()
    if correction.shape[-2] > 1:
        smooth = smooth + correction.diff(dim=-2).abs().mean()
    return anchor, smooth
