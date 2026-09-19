"""One low-only post-gamma RGB-shared fast coordinate per fixed Region2."""
import torch
from types import SimpleNamespace
from .semantic_ttt import Region2
from .gamma_range_box import Gamma05Box
from .isp import physical_parameters

class CommonRegion2(Region2):
    def __init__(self,active):
        super().__init__(active)
        self.raw=torch.nn.Parameter(torch.zeros(1,3,2,2))

    def physical_grid(self):
        return physical_grid(self.raw)

def physical_grid(raw):
    return physical_parameters(torch.cat([raw[:,:2],raw[:,2:3].expand(-1,3,-1,-1),raw.new_zeros(raw.shape[0],1,2,2)],1))

class CommonBox(Gamma05Box):
    def __call__(self,model):
        super().__call__(SimpleNamespace(raw=model.raw[:,:2]))
        with torch.no_grad():model.raw[:,2:].masked_fill_(~self.active.reshape(1,1,2,2),0.)
