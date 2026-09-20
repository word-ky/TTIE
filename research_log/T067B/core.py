import math
from research_log.T063C.core import progress,choose
RHO=.9857470621423519
GRID=[0,.125,.25,.375,.5,.625,.75,.875,1]
def choices(values,probabilities,base):
    best,reduction,r=progress(values);assert math.isfinite(reduction) and reduction>1e-12
    assert choose(values,RHO)==base
    fs=next((k for k in range(base+1) if probabilities[k]>=.5),None);assert fs is not None
    rows=[]
    for lam in GRID:
        target=float(r[fs]+lam*(RHO-r[fs]));selected=next((k for k in range(fs,base+1) if r[k]>=target),None);assert selected is not None,'Invalid interval'
        rows.append(dict(lambda_value=lam,k_FS=fs,k_rho=base,r_FS=float(r[fs]),r_target=target,selected_step=selected))
    assert rows[0]['selected_step']==fs and rows[-1]['selected_step']==base
    return rows

def select(table):
    passing=[r for r in table if all(r['gates'].values())]
    order=sorted(passing,key=lambda r:(-r['worst_delta_t026'],-r['mean_delta_psnr'],r['lambda_value']))
    best=order[0] if order else None
    verdict='INTERIOR_PROGRESS_DEV_CANDIDATE_FROZEN' if best is not None and 0<best['lambda_value']<1 else 'INTERIOR_PROGRESS_DEV_NEGATIVE'
    return best,verdict,[r['lambda_value'] for r in order]
