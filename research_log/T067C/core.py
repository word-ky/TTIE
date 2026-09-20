from research_log.T063C.core import progress,choose
RHO=.9857470621423519
LAMBDA=.875
def select(values,probabilities,base):
    import math
    _,reduction,r=progress(values);assert math.isfinite(reduction) and reduction>1e-12 and choose(values,RHO)==base
    fs=next((k for k in range(base+1) if probabilities[k]>=.5),None);assert fs is not None
    target=float(r[fs]+LAMBDA*(RHO-r[fs]));k=next((k for k in range(fs,base+1) if r[k]>=target),None);assert k is not None
    return dict(k_FS=fs,k_rho=base,r_FS=float(r[fs]),r_target=target,selected_step=k)
