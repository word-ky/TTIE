"""Fixed physical actions; only separable spatial weights vary."""
from itertools import product
import torch
from torch import nn
from ..isp import ISP

CANDIDATES = tuple(product((.4, .5, .6), (.4, .5, .6), (0., .05, .1)))
HARD = CANDIDATES.index((.5, .5, 0.))


def weights(size, candidate, like):
    h, w = size
    bx, by, tau = candidate
    x = torch.arange(w, device=like.device, dtype=like.dtype)
    y = torch.arange(h, device=like.device, dtype=like.dtype)
    if tau == 0:
        hx = (x >= int(bx*w)).to(like.dtype)
        hy = (y >= int(by*h)).to(like.dtype)
    else:
        hx = torch.sigmoid(((x+.5)/w-bx)/tau)
        hy = torch.sigmoid(((y+.5)/h-by)/tau)
    return torch.stack(((1-hy)[:, None]*(1-hx)[None, :],
                        (1-hy)[:, None]*hx[None, :],
                        hy[:, None]*(1-hx)[None, :],
                        hy[:, None]*hx[None, :]))


class FixedCorners(ISP):
    def __init__(self, corners, candidate):
        nn.Module.__init__(self)
        self.register_buffer('corners', corners.detach().clone())
        self.candidate = candidate

    def parameter_field(self, size):
        w = weights(size, self.candidate, self.corners)
        identity = self.corners.new_tensor((0., 1.))[None, :, None, None]
        pair = identity + torch.einsum('khw,bck->bchw', w,
                                      (self.corners-identity).flatten(2))
        return torch.cat((pair, torch.ones_like(pair).repeat(1, 2, 1, 1)), dim=1)

    def forward(self, image):
        result = super().forward(image)
        pair = self.parameter_field(image.shape[-2:])[:, :2]
        identity = (pair[:, :1] == 0) & (pair[:, 1:2] == 1)
        return torch.where(identity, image, result)
