"""T026-A: one active-gamma lower-bound change; accepted T022-C stays unchanged."""
import torch
from .ev_range_box import DarkEV2Box


class Gamma05Box(DarkEV2Box):
    def __init__(self, objective, size):
        super().__init__(objective, size)
        self.lower[0,1]=torch.where(self.active.reshape(2,2),.5,self.lower[0,1])
