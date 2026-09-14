"""Torch-only export for a separate NumPy/SciPy compact replay process."""
import argparse
from pathlib import Path
import torch,numpy as np
p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
source=torch.load(a.result/'radius/source.pt',weights_only=True,map_location='cpu')
arrays={k:v.numpy() for k,v in source.items()};features=[];distances=[];lengths=[]
for i in range(100):
    t=torch.load(a.result/'audit'/f'{i:03d}'/'trajectory.pt',weights_only=True,map_location='cpu')
    features.append(t['features'].numpy());distances.append(t['distances'].numpy());lengths.append(len(t['features']))
a.out.parent.mkdir(parents=True,exist_ok=True)
np.savez(a.out,**arrays,features=np.concatenate(features),distances=np.concatenate(distances),lengths=np.asarray(lengths))
