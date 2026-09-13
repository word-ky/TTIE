"""Single T022-C change; historical ActionBox remains untouched."""
import torch
from .projected_ttt import ActionBox


class DarkEV2Box(ActionBox):
    def __init__(self,objective,size):
        super().__init__(objective,size)
        dark=(self.active & (self.winner==0)).reshape(2,2)
        self.upper[0,0]=torch.where(dark,2.,self.upper[0,0])
