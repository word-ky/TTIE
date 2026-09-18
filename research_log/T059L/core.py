import torch
from research_log.T059G.core import nearest
LIMIT=.07650849781930447
LABELS=['bank-relative feature displacement is not even source-LOO consistent under fixed 1-NN; stop','bank-relative feature displacement restores image-held scalar localization under fixed 1-NN; relative conditioning is supported as a mechanism','bank-relative feature displacement preserves train-LOO consistency but does not resolve unseen-image scalar localization']
def classify(t,h):return LABELS[0] if t>LIMIT else LABELS[1] if h<=LIMIT else LABELS[2]
def displacement(x,bank,state):
 a=torch.empty(len(x),dtype=torch.long)
 for b in bank.unique():
  rows=torch.where(bank==b)[0];anchors=rows[state[rows]==0];assert len(anchors)==1
  a[rows]=anchors[0]
 return a,x-x[a]
