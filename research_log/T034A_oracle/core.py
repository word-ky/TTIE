"""REFERENCE_ORACLE_ONLY: WB capacity extension; no deployable module changes."""
import importlib.util
from pathlib import Path
import torch
from types import SimpleNamespace
from ttie.semantic_ttt import Region2
from ttie.gamma_range_box import Gamma05Box
from ttie.isp import physical_parameters

spec=importlib.util.spec_from_file_location('t028_oracle_core',Path(__file__).resolve().parents[1]/'T028A_oracle/core.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
optimize_start=prior.optimize_start

class WBRegion2(Region2):
    def __init__(self,active):
        super().__init__(active)
        self.raw=torch.nn.Parameter(torch.zeros(1,5,2,2))

    def physical_grid(self):
        # Exact existing ISP map: EV, gamma, WB-R/G/B; contrast remains identity.
        return physical_parameters(torch.cat([self.raw,self.raw.new_zeros(1,1,2,2)],dim=1))

class WBBox(Gamma05Box):
    def __call__(self,model):
        # Reuse the exact old EV/gamma projection without changing Adam moments.
        super().__call__(SimpleNamespace(raw=model.raw[:,:2]))
        with torch.no_grad():model.raw[:,2:].masked_fill_(~self.active.reshape(1,1,2,2),0.)

def frozen_model(decision,image):
    obj=SimpleNamespace(active=torch.tensor(decision['gate']['active'],device=image.device,dtype=torch.bool),
                        winner=torch.tensor(decision['gate']['winner'],device=image.device))
    model=WBRegion2(obj.active).to(image);box=WBBox(obj,2)
    saved=decision['diagnostics']['action_box']
    assert torch.equal(box.lower.cpu(),torch.tensor(saved['lower'])) and torch.equal(box.upper.cpu(),torch.tensor(saved['upper']))
    return model,box

def extend_start(raw):return torch.cat([raw,raw.new_zeros(1,3,2,2)],dim=1)
