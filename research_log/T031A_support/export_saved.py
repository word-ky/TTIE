"""Separate Torch process for portable read-only compact evidence verification."""
import argparse
from pathlib import Path
import torch,numpy as np
p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
b=torch.load(a.result/'support/bound_features.pt',weights_only=True,map_location='cpu')
g=torch.load(a.result/'REFERENCE_GRADIENT_DIAGNOSTIC_ONLY/gradients.pt',weights_only=True,map_location='cpu')
a.out.parent.mkdir(parents=True,exist_ok=True)
np.savez(a.out,**{k:v.numpy() for k,v in b.items()},gradients=g.numpy())
