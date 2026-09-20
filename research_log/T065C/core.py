"""Fixed five-neighbor safety bank and exact rollback rule."""
import numpy as np
import torch
from research_log.T065A.core import features,FEATURES,base_choose,summarize,RHO
from research_log.T065B.core import choose,confusion
K=5


def build_bank(training,normalization):
    mean=np.asarray(normalization['mean']);scale=np.asarray(normalization['scale'])
    return dict(mean=mean.tolist(),scale=scale.tolist(),normalized_features=((np.asarray([r['features'] for r in training])-mean)/scale).tolist(),
        labels=[r['safe'] for r in training],image_indices=[r['index'] for r in training],steps=[r['step'] for r in training],
        k=5,threshold=.5,metric='Euclidean float64',weighting='unweighted',tie_order=['distance','development_image_index','step'])


def predict(queries,bank,exclude_image=None,device='cpu'):
    q=(np.asarray(queries,dtype=np.float64)-np.asarray(bank['mean']))/np.asarray(bank['scale'])
    reference=torch.tensor(bank['normalized_features'],dtype=torch.float64,device=device)
    query=torch.tensor(q,dtype=torch.float64,device=device)
    distances=(query[:,None,:]-reference[None,:,:]).square().sum(dim=-1).sqrt().cpu().numpy()
    ids=np.asarray(bank['image_indices']);steps=np.asarray(bank['steps']);labels=np.asarray(bank['labels'])
    permitted=np.flatnonzero(ids!=exclude_image) if exclude_image is not None else np.arange(len(ids))
    probabilities=[];neighbors=[]
    for row in distances:
        chosen=permitted[np.lexsort((steps[permitted],ids[permitted],row[permitted]))[:5]]
        probabilities.append(float(np.mean(labels[chosen])))
        neighbors.append([dict(bank_row=int(j),index=int(ids[j]),step=int(steps[j]),distance=float(row[j])) for j in chosen])
    return probabilities,neighbors
