"""T016-D: unweighted within-episode logistic ranking, unchanged scalar-head size."""
import hashlib
import torch
from torch.nn import functional as F
from .stop_quality import QualityHead

RECIPE=dict(hidden=[64,64],activation='SiLU',loss='unweighted pairwise logistic',
    optimizer='AdamW',lr=1e-3,weight_decay=1e-4,betas=[.9,.999],eps=1e-8,
    pair_batch_size=256,epochs=100,seed=7,checkpoint='final_epoch',device='cpu',
    normalization='all training candidate rows only; population std; constant scales=1',
    output_standardization=False,pair_order='saved episode order, then lexicographic i<j',
    ties='skip exact reference MSE equality only')


def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def pairs(mse):
    i,j=torch.triu_indices(9,9,offset=1)
    episode,position=torch.where(mse[:,i]!=mse[:,j])
    left,right=i[position],j[position]
    signs=torch.where(mse[episode,left]<mse[episode,right],1.,-1.)
    return torch.stack((episode,left,right),dim=1),signs


def pair_loss(left,right,sign):return F.softplus(sign*(left-right)).mean()


def train_rank(features,mse):
    torch.manual_seed(7);torch.set_num_threads(1)
    x=features.detach().cpu().float();y=mse.detach().cpu().double()
    pair,sign=pairs(y);head=QualityHead(x.shape[-1],torch.nn.SiLU)
    flat=x.reshape(-1,x.shape[-1])
    with torch.no_grad():
        head.x_mean.copy_(flat.double().mean(0));scale=flat.double().std(0,unbiased=False)
        head.x_scale.copy_(torch.where(scale==0,torch.ones_like(scale),scale))
    # y_mean=0 and y_scale=1 remain their constructor values: no target standardization.
    optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
    generator=torch.Generator().manual_seed(7);history=[]
    for epoch in range(100):
        order=torch.randperm(len(pair),generator=generator);total=0.;seen=0
        for batch in order.split(256):
            e,i,j=pair[batch].T
            loss=pair_loss(head(x[e,i]),head(x[e,j]),sign[batch])
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            total+=float(loss.detach())*len(batch);seen+=len(batch)
        history.append(dict(epoch=epoch+1,train_pairwise_loss=total/len(pair),pairs_seen=seen,
                            permutation_sha256=tensor_sha(order)))
    head.zero_grad(set_to_none=True);head.eval().requires_grad_(False)
    return head,history,dict(non_tied_pairs=len(pair),exact_ties_skipped=len(x)*36-len(pair),
        pair_index_sha256=tensor_sha(pair),pair_sign_sha256=tensor_sha(sign))
