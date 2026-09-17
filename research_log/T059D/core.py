"""Deterministic bank-relative scalar/ranking diagnostics; no fitting."""
import torch
LIMIT=.07650849781930447

def average_ranks(v):
    order=torch.argsort(v,stable=True);rank=torch.empty(len(v),dtype=torch.float64);start=0
    while start<len(v):
        end=start+1
        while end<len(v) and v[order[end]]==v[order[start]]:end+=1
        rank[order[start:end]]=(start+1+end)/2;start=end
    return rank

def spearman(p,t):
    if len(p)<2:return None,'singleton'
    a=average_ranks(p);b=average_ranks(t);a-=a.mean();b-=b.mean();den=a.norm()*b.norm()
    if den==0:return None,'constant_prediction_or_target'
    return float((a*b).sum()/den),'defined'

def directional(cos,mask):
    v=cos[mask]
    return dict(eligible=len(v),ineligible=len(cos)-len(v),positive_fraction=float((v>0).float().mean()) if len(v) else None,median_cosine=float(torch.quantile(v,.5)) if len(v) else None)

def bank_metrics(p,t,legacy_cos,detail_cos,legacy_mask,detail_mask,bank_indices,state_indices,global_indices):
    assert all(torch.isfinite(v).all() for v in [p,t,legacy_cos,detail_cos])
    dp=torch.empty_like(p);dt=torch.empty_like(t);anchors=torch.empty(len(p),dtype=torch.int64);banks=[]
    for bi in dict.fromkeys(bank_indices.tolist()):
        ids=(bank_indices==bi).nonzero().flatten();assert torch.equal(state_indices[ids],torch.arange(len(ids)))
        anchor_ids=ids[state_indices[ids]==0];assert len(anchor_ids)==1;anchor=int(anchor_ids[0]);anchors[ids]=anchor
        pp=p[ids];tt=t[ids];dp[ids]=pp-p[anchor];dt[ids]=tt-t[anchor]
        rho,status=spearman(pp,tt);choice=int(torch.argmin(pp));target_choice=int(torch.argmin(tt))
        banks.append(dict(bank_index=bi,rows=len(ids),anchor_state_index=0,anchor_global_index=int(global_indices[anchor]),anchor_residual=float(p[anchor]-t[anchor]),unanchored_huber=float(torch.nn.functional.huber_loss(pp,tt,delta=1.)),relative_huber=float(torch.nn.functional.huber_loss(dp[ids],dt[ids],delta=1.)),spearman=rho,spearman_status=status,predicted_argmin_state_index=int(state_indices[ids[choice]]),target_argmin_state_index=int(state_indices[ids[target_choice]]),argmin_regret=float(tt[choice]-tt.min()),legacy=directional(legacy_cos[ids],legacy_mask[ids]),detail=directional(detail_cos[ids],detail_mask[ids])))
    return banks,dp,dt,anchors

def distribution(values):
    vals=[v for v in values if v is not None];t=torch.tensor(vals,dtype=torch.float64)
    if not vals:return dict(defined=0,undefined=len(values))
    return dict(defined=len(vals),undefined=len(values)-len(vals),mean=float(t.mean()),minimum=float(t.min()),p10=float(torch.quantile(t,.1)),median=float(torch.quantile(t,.5)),p90=float(torch.quantile(t,.9)),maximum=float(t.max()))
