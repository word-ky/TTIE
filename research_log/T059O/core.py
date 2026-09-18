import torch
from research_log.T059G.core import nearest
LIMIT=.07650849781930447
LABELS=['explicit anchor-context 56-D locality fails the source control; stop','explicit bank-anchor context is insufficient for source-image scalar localization under fixed 1-NN','explicit bank-anchor context restores source-selector scalar locality under fixed 1-NN; context-conditioned representation is supported for a later parametric test']
def classify(f,s):return LABELS[0] if f>LIMIT else LABELS[1] if s>LIMIT else LABELS[2]
def context(x,mean,scale,bank,state):
 a=torch.empty(len(x),dtype=torch.long)
 for b in bank.unique():
  q=torch.where((bank==b)&(state==0))[0];assert len(q)==1;a[bank==b]=q[0]
 u=(x-mean)/scale;u0=u[a];return dict(anchor=a,u=u,u0=u0,z=torch.cat((u-u0,u0),dim=1))
