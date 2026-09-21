import math
import numpy as np
EPS=1e-12

def score(values,fs,k1,motions):
    k0=max(fs,k1-3);d=np.asarray(motions,dtype=np.float64);assert len(d)==k1-fs+1 and d[0]==0
    loss=np.asarray(values,dtype=np.float64);dt=float(loss[fs]-loss[k1]);mt=float(np.sum(d,dtype=np.float64));tail=float(np.sum(d[k0-fs+1:],dtype=np.float64))
    p=float(np.clip((loss[k0]-loss[k1])/max(dt,EPS),0.,1.));q=tail/max(mt,EPS);ratio=math.log((q+EPS)/(p+EPS))
    if fs==k1:p=q=ratio=0.
    return dict(k_FS=fs,k_lambda=k1,k0=k0,D_total=dt,p_tail=p,m_j=d.tolist(),M_total=mt,M_tail=tail,q_tail=q,R=ratio,reason='singleton' if fs==k1 else 'formula')

def threshold(rows):
    values=sorted(float(r['R']) for r in rows);assert len(values)==100 and all(math.isfinite(v) for v in values)
    return values[98]

def diagnose(rows):
    unsafe=[r for r in rows if r['unsafe']];assert unsafe,'Intended unsafe diagnostic target missing'
    tp=sum(r['above_T99'] for r in unsafe);fp=sum(r['above_T99'] for r in rows if not r['unsafe'])
    return dict(classification='TAIL_INEFFICIENCY_SIGNAL_PRESENT' if tp==len(unsafe) and fp<=5 else 'TAIL_INEFFICIENCY_SIGNAL_ABSENT',unsafe_endpoints=len(unsafe),unsafe_above_T99=tp,safe_endpoints=len(rows)-len(unsafe),safe_above_T99=fp)
