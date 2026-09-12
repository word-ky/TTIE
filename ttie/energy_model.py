"""T013 smooth energy, with the unchanged T012 source-only training recipe."""
import torch
from torch import nn
from .stop_quality import QualityHead,train_head,RECIPE as STOP_RECIPE
from .stop_trajectory import FEATURE_NAMES

SCHEMA=dict(version='T013-28-v1',names=FEATURE_NAMES[:28],dimension=28)
RECIPE=dict(STOP_RECIPE,input_dim=28,activation='SiLU')


class EnergyHead(QualityHead):
    def __init__(self):super().__init__(28,nn.SiLU)


def train_energy(features,mse):return train_head(features,mse,head_factory=EnergyHead)


def save_energy(head,path):torch.save(dict(state_dict=head.state_dict(),schema=SCHEMA,recipe=RECIPE),path)


def load_energy(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    assert saved['schema']==SCHEMA and saved['recipe']==RECIPE
    head=EnergyHead();head.load_state_dict(saved['state_dict']);return head.eval().requires_grad_(False)


def features(objective,scores,grid):
    # Only original gate constants detach. Current CLIP evidence and ISP state
    # retain their complete autograd graph, including device/dtype conversion.
    active=objective.active.to(scores);signed=active*torch.where(objective.winner==0,1.,-1.)
    z=(scores.double()-scores.new_tensor(objective.calibration['tau'],dtype=torch.float64))/scores.new_tensor(objective.calibration['scale'],dtype=torch.float64)
    state=grid.expand(1,2,2,2)
    return torch.cat((active,signed,objective.evidence,z[:,0],z[:,1],state[0,0].flatten(),state[0,1].flatten())).float()
