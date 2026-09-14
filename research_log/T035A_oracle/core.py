"""REFERENCE_ORACLE_ONLY: one RGB-shared gain per existing Region2 cell."""
import importlib.util
from pathlib import Path
import torch
from types import SimpleNamespace
from ttie.isp import physical_parameters

spec=importlib.util.spec_from_file_location('t034_wb_core',Path(__file__).resolve().parents[1]/'T034A_oracle/core.py')
wb=importlib.util.module_from_spec(spec);spec.loader.exec_module(wb)
prior=wb.prior
optimize_start=prior.optimize_start

class CommonRegion2(wb.WBRegion2):
    def __init__(self,active):
        super().__init__(active)
        self.raw=torch.nn.Parameter(torch.zeros(1,3,2,2))

    def physical_grid(self):
        return common_physical(self.raw)

def common_physical(raw):
    return physical_parameters(torch.cat([raw[:,:2],raw[:,2:3].expand(-1,3,-1,-1),raw.new_zeros(raw.shape[0],1,2,2)],1))

def frozen_model(decision,image):
    obj=SimpleNamespace(active=torch.tensor(decision['gate']['active'],device=image.device,dtype=torch.bool),
                        winner=torch.tensor(decision['gate']['winner'],device=image.device))
    model=CommonRegion2(obj.active).to(image);box=wb.WBBox(obj,2)
    saved=decision['diagnostics']['action_box']
    assert torch.equal(box.lower.cpu(),torch.tensor(saved['lower'])) and torch.equal(box.upper.cpu(),torch.tensor(saved['upper']))
    return model,box

def extend_start(raw):return torch.cat([raw,raw.new_zeros(1,1,2,2)],dim=1)
