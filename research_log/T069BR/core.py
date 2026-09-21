import numpy as np
from research_log.T069B.core import score,threshold,diagnose,TOTAL_ATOL,TOTAL_RTOL
DOUBLE_ATOL=1e-12
DOUBLE_RTOL=1e-10

def check(actual,anchor,double=False):
    a=np.asarray(actual,dtype=np.float64);b=np.asarray(anchor,dtype=np.float64)
    assert a.shape==b.shape==(12,) and np.isfinite(a).all() and np.isfinite(b).all()
    diff=np.abs(a-b);atol,rtol=(DOUBLE_ATOL,DOUBLE_RTOL) if double else (TOTAL_ATOL,TOTAL_RTOL)
    return dict(abs_residual=diff.tolist(),max_abs=float(diff.max()),passes=bool(np.all(diff<=atol+rtol*np.abs(b))))
