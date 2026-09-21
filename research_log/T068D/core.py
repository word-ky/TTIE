import math
import numpy as np

def score(components,fs,kl):
    z=np.asarray(components,dtype=np.float64)[fs:kl+1]*np.array([1.,10.,5.],dtype=np.float64)
    zmin=z.min(axis=0);zmax=z.max(axis=0);a=z[-1]-zmin;e=zmax-zmin;den=float(e.sum())
    if fs==kl:r=0.;reason='singleton'
    elif den<=1e-12:r=0.;reason='zero_excursion'
    else:r=float(a.sum())/max(den,1e-12);reason='formula'
    return dict(k_FS=fs,k_lambda=kl,Z=z.tolist(),endpoint_Z=z[-1].tolist(),zmin=zmin.tolist(),zmax=zmax.tolist(),a=a.tolist(),e=e.tolist(),sum_a=float(a.sum()),sum_e=den,R_comp=r,reason=reason)

def threshold(rows):
    values=sorted(float(r['R_comp']) for r in rows);assert len(values)==100 and all(math.isfinite(v) for v in values)
    return values[98]

def diagnose(rows):
    unsafe=[r for r in rows if r['unsafe']];assert unsafe,'Intended unsafe diagnostic target missing'
    tp=sum(r['above_T99_comp'] for r in unsafe);fp=sum(r['above_T99_comp'] for r in rows if not r['unsafe'])
    return dict(classification='COMPONENT_REGRET_SIGNAL_PRESENT' if tp==len(unsafe) and fp<=5 else 'COMPONENT_REGRET_SIGNAL_ABSENT',unsafe_endpoints=len(unsafe),unsafe_above_T99=tp,safe_endpoints=len(rows)-len(unsafe),safe_above_T99=fp)
