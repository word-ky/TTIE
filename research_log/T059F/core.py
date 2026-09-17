"""Frozen closed-form scale diagnostic. CPU float32 residuals as in E."""
import torch
from research_log.T059D.core import spearman,distribution,LIMIT

def decompose(v):
    p,t=v['p'],v['t'];dp=torch.empty_like(p);dt=torch.empty_like(t);scaled=torch.empty_like(p);banks=[];failures=[]
    for bi in dict.fromkeys(v['bank_indices'].tolist()):
        ids=(v['bank_indices']==bi).nonzero().flatten()
        az=ids[v['state_indices'][ids]==0];assert len(az)==1
        aidx=int(az[0]);rp=p[ids]-p[aidx];rt=t[ids]-t[aidx]
        dp[ids]=rp;dt[ids]=rt
        den=torch.dot(rp,rp);num=torch.dot(rp,rt)
        a=max(0.,float(num/den)) if den!=0 else 0.
        sp=a*rp;scaled[ids]=sp
        rho,status=spearman(p[ids],t[ids]);rho2,status2=spearman(sp,rt)
        choice=int(p[ids].argmin());choice2=int(sp.argmin())
        regret=float(t[ids][choice]-t[ids].min());regret2=float(t[ids][choice2]-t[ids].min())
        invariant=(rho==rho2 and status==status2 and regret==regret2)
        if not invariant:failures.append(bi)
        banks.append(dict(bank_index=bi,rows=len(ids),anchor_state_index=0,anchor_global_index=int(v['global_indices'][aidx]),scale=a,numerator=float(num),denominator=float(den),relative_huber=float(torch.nn.functional.huber_loss(rp,rt)),corrected_huber=float(torch.nn.functional.huber_loss(sp,rt)),spearman=rho,spearman_status=status,scaled_spearman=rho2,scaled_spearman_status=status2,argmin_regret=regret,scaled_argmin_regret=regret2,predicted_argmin_state_index=int(v['state_indices'][ids[choice]]),scaled_argmin_state_index=int(v['state_indices'][ids[choice2]]),invariant=invariant))
    assert torch.equal(dp,v['delta_p']) and torch.equal(dt,v['delta_t'])
    h=float(torch.nn.functional.huber_loss(scaled,dt));scales=[b['scale'] for b in banks]
    classification=('T059-E value failure is consistent with bankwise positive-scale miscalibration after offset removal' if h<=LIMIT else 'T059-E value failure is not explained by bankwise positive-scale miscalibration after offset removal')
    return dict(rows=len(p),bank_count=len(banks),relative_huber=float(torch.nn.functional.huber_loss(dp,dt)),corrected_huber=h,threshold=LIMIT,threshold_margin=LIMIT-h,classification=classification if not failures else None,status='DONE' if not failures else 'PARTIAL',stop_reason=None if not failures else 'Prescribed nonnegative scaling changed ordering/regret; task stopped without scientific classification',invariance_failures=failures,scales=distribution(scales),zero_scale_count=sum(a==0 for a in scales),spearman_distribution=distribution([b['spearman'] for b in banks]),regret_distribution=distribution([b['argmin_regret'] for b in banks]),banks=banks,worst10=sorted(banks,key=lambda b:(-b['corrected_huber'],b['bank_index']))[:10])
