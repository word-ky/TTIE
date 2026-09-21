import math
import numpy as np

def score(states,pre_box,fs,k):
    end=np.asarray(states[k],dtype=np.float64)[:,:2]
    if k==0:return dict(k_FS=fs,k_lambda=k,s_prev=None,p=None,s_end=end.tolist(),norm_prop=0.,norm_clip=0.,R_proj=0.,reason='no_transition')
    prev=np.asarray(states[k-1],dtype=np.float64)[:,:2];p=np.asarray(pre_box[k-1],dtype=np.float64)[:,:2]
    prop=p-prev;clip=p-end;pn=float(np.linalg.norm(prop));cn=float(np.linalg.norm(clip))
    return dict(k_FS=fs,k_lambda=k,s_prev=prev.tolist(),p=p.tolist(),s_end=end.tolist(),norm_prop=pn,norm_clip=cn,R_proj=0. if pn<=1e-12 else cn/max(pn,1e-12),reason='zero_proposal' if pn<=1e-12 else 'formula')

def threshold(rows):
    values=sorted(float(r['R_proj']) for r in rows);assert len(values)==100 and all(math.isfinite(v) for v in values)
    return values[98]

def diagnose(rows):
    unsafe=[r for r in rows if r['unsafe']];assert unsafe,'Intended unsafe diagnostic target missing'
    tp=sum(r['above_T99_proj'] for r in unsafe);fp=sum(r['above_T99_proj'] for r in rows if not r['unsafe'])
    return dict(classification='PROJECTION_PRESSURE_SIGNAL_PRESENT' if tp==len(unsafe) and fp<=5 else 'PROJECTION_PRESSURE_SIGNAL_ABSENT',unsafe_endpoints=len(unsafe),unsafe_above_T99=tp,safe_endpoints=len(rows)-len(unsafe),safe_above_T99=fp)
