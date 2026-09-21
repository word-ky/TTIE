import math
import numpy as np
TOTAL_ATOL=2e-7
TOTAL_RTOL=2e-5

def score(gradients):
    g=np.asarray(gradients,dtype=np.float64);assert g.shape==(3,12)
    norms=np.linalg.norm(g,axis=1);total=g.sum(axis=0);tn=float(np.linalg.norm(total));den=float(norms.sum())
    r=0. if den<=1e-12 else 1-tn/max(den,1e-12)
    return dict(gradients=g.tolist(),gradient_norms=norms.tolist(),summed_gradient=total.tolist(),summed_gradient_norm=tn,sum_component_norms=den,R_cancel=r,reason='zero_component_gradient' if den<=1e-12 else 'formula')

def weighted_gradients(parts,raw):
    import torch
    return [torch.autograd.grad(parts[c]*w,raw,retain_graph=True)[0].detach() for c,w in enumerate([1.,10.,5.])]

def threshold(rows):
    values=sorted(float(r['R_cancel']) for r in rows);assert len(values)==100 and all(math.isfinite(v) for v in values)
    return values[98]

def diagnose(rows):
    unsafe=[r for r in rows if r['unsafe']];assert unsafe,'Intended unsafe diagnostic target missing'
    tp=sum(r['above_T99_cancel'] for r in unsafe);fp=sum(r['above_T99_cancel'] for r in rows if not r['unsafe'])
    return dict(classification='GRADIENT_CANCELLATION_SIGNAL_PRESENT' if tp==len(unsafe) and fp<=5 else 'GRADIENT_CANCELLATION_SIGNAL_ABSENT',unsafe_endpoints=len(unsafe),unsafe_above_T99=tp,safe_endpoints=len(rows)-len(unsafe),safe_above_T99=fp)
