from research_log.T059S.core import *
import numpy as np
U=ROOT/'runs/20260919-013345-ttie-t059u-outer/artifacts/T059U'
def errors(online,cached):
    a=np.asarray(online,dtype=np.float64);b=np.asarray(cached,dtype=np.float64);diff=float(np.linalg.norm(a-b));an=float(np.linalg.norm(a));bn=float(np.linalg.norm(b))
    return dict(max_abs=float(np.max(np.abs(a-b))),relative=diff/max(bn,1e-30),cosine=1. if an==bn==0 else 0. if an*bn==0 else float(np.sum(a*b)/(an*bn)),online_norm=an,cached_norm=bn)
def passed(row):return row['x']['max_abs']<=5e-5 and row['J']['relative']<=2e-3 and row['g']['cosine']>=.999 and row['g']['relative']<=1e-2 and row['y1']['max_abs']<=1e-4
