"""Fixed support geometry and predeclared diagnostic categories; never selects outputs."""
import numpy as np
import torch
from research_log.T065B.core import confusion


def geometry(queries,bank,labels,query_ids=None,device='cpu'):
    q=torch.as_tensor(queries,dtype=torch.float64,device=device)
    b=torch.as_tensor(bank,dtype=torch.float64,device=device)
    labels=np.asarray(labels,dtype=bool);rows=[]
    for start in range(0,len(q),28):
        ds=(q[start:start+28,None,:]-b[None,:,:]).square().sum(-1).sqrt().cpu().numpy()
        for j,d in enumerate(ds):
            if query_ids is not None:d[np.repeat(np.arange(len(bank)//28),28)==query_ids[start+j]]=np.inf
            safe=np.flatnonzero(labels);unsafe=np.flatnonzero(~labels)
            si=int(safe[np.argmin(d[safe])]);ui=int(unsafe[np.argmin(d[unsafe])])
            rows.append(dict(d_safe=float(d[si]),d_unsafe=float(d[ui]),margin=float(d[ui]-d[si]),safe_bank_row=si,unsafe_bank_row=ui))
    return rows


def classification(recall,safe_fraction,false_safe_bases):
    if recall<.5:return 'TRANSFER_SUPPORT_SHIFT' if safe_fraction>=.75 else 'BOUNDARY_MISMATCH_WITH_UNSAFE_SUPPORT'
    return 'SELECTED_TAIL_SPECIFIC_FAILURE' if false_safe_bases else 'NO_DIAGNOSTIC_FAILURE'


def scores(labels,probabilities,mask):
    cm=confusion(np.asarray(labels)[mask],np.asarray(probabilities)[mask]);n=cm['unsafe_pred_safe']+cm['unsafe_pred_unsafe']
    return dict(states=int(np.sum(mask)),confusion=cm,unsafe_recall=cm['unsafe_pred_unsafe']/n if n else None)


def distribution(rows,mask):
    selected=[r for r,m in zip(rows,mask) if m]
    out=dict(states=len(selected),fraction_m_positive=sum(r['margin']>0 for r in selected)/len(selected) if selected else None)
    for key in ['d_safe','d_unsafe','margin']:
        x=np.asarray([r[key] for r in selected]);out[key]=dict(zip(['min','q25','median','q75','max'],np.quantile(x,[0,.25,.5,.75,1]).tolist())) if len(x) else None
        if len(x):out[key]['mean']=float(x.mean())
    return out
