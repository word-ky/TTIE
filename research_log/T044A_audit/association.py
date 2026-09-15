import numpy as np
from scipy.stats import rankdata
SUPPORTED='legacy-extrapolation risk association supported'
NEGATIVE='legacy-extrapolation risk association not supported / mixed'
def verdict(auc,rho):return SUPPORTED if auc>=.75 and rho<=-.35 else NEGATIVE
def summarize(scores,deltas):
    x=np.asarray(scores,dtype=np.float64);y=np.asarray(deltas,dtype=np.float64);loss=y<0
    n=int(loss.sum());k=len(x)-n
    rx,ry=rankdata(x),rankdata(y)
    auc=float((rx[loss].sum()-n*(n+1)/2)/(n*k))
    dx,dy=rx-rx.mean(),ry-ry.mean()
    rho=float(np.sum(dx*dy)/np.sqrt(np.sum(dx*dx)*np.sum(dy*dy)))
    groups={}
    for name,mask in [('loss',loss),('non_loss',~loss)]:
        v=x[mask];groups[name]=dict(count=len(v),mean=float(v.mean()),q25=float(np.quantile(v,.25)),median=float(np.median(v)),q75=float(np.quantile(v,.75)))
    return dict(roc_auc=auc,spearman=rho,loss_count=n,non_loss_count=k,groups=groups,classification=verdict(auc,rho))
