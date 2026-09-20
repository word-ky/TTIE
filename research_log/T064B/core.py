"""One fixed antithetic photometric-sensitivity rollback statistic."""
import math,statistics
import torch
from research_log.T063B.core import summarize
from research_log.T063C.core import choose as base_choose

SEED=64064
DELTA=1/255
RHO=0.9857470621423519


def perturb(low):
    generator=torch.Generator(device='cpu').manual_seed(SEED)
    e=(torch.randint(0,2,low.shape,generator=generator,dtype=torch.int64)*2-1).to(device=low.device,dtype=low.dtype)
    return (low+DELTA*e).clamp(0,1),(low-DELTA*e).clamp(0,1)


def sensitivity(plus,minus,input_plus,input_minus):
    numerator=float((plus-minus).abs().mean())
    denominator=float((input_plus-input_minus).abs().mean())
    return numerator/max(denominator,1e-8)


def choose(values,base_step,tau):
    return max([0]+[k for k in range(base_step+1) if math.isfinite(values[k]) and values[k]<=tau])


def candidates(rows):
    return sorted({float(v) for r in rows for v in r['sensitivity'][:r['base_step']+1] if math.isfinite(v)})


def calibrate(rows,quality):
    table=[]
    for tau in candidates(rows):
        steps=[choose(r['sensitivity'],r['base_step'],tau) for r in rows]
        metrics=summarize(quality,steps)
        metrics['eligible']=all(metrics['gates'].values())
        table.append(dict(tau=tau,mean_psnr=statistics.fmean(q['psnr'][k] for q,k in zip(quality,steps)),**metrics))
    eligible=[r for r in table if r['eligible']]
    best=max(eligible,key=lambda r:(r['mean_psnr'],-r['tau'])) if eligible else None
    return table,best,best is not None
