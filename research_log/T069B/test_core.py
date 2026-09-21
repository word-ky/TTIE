import numpy as np
import torch
from research_log.T069B.core import score,weighted_gradients,threshold,diagnose

def test_aligned_opposed_zero():
    g=np.zeros((3,12));g[:,0]=[1,2,3];assert score(g)['R_cancel']==0
    g[:,0]=[1,-1,0];assert score(g)['R_cancel']==1
    assert score(np.zeros((3,12)))['reason']=='zero_component_gradient'

def test_weighted_gradients_and_direct_total():
    x=torch.ones(12,requires_grad=True);parts=torch.stack([x.sum(),(x*x).sum(),3*x.sum()]);g=weighted_gradients(parts,x)
    assert torch.equal(g[0],torch.ones(12)) and torch.equal(g[1],20*torch.ones(12)) and torch.equal(g[2],15*torch.ones(12))
    direct,=torch.autograd.grad(parts@parts.new_tensor([1,10,5]),x);assert torch.equal(sum(g),direct)

def test_nearest_rank_strict_flags_and_verdict():
    t=threshold([dict(R_cancel=k/100) for k in range(100)]);assert t==.98 and not(t>t)
    rows=[dict(unsafe=True,above_T99_cancel=True)]+[dict(unsafe=False,above_T99_cancel=True)]*5
    assert diagnose(rows)['classification']=='GRADIENT_CANCELLATION_SIGNAL_PRESENT'
    assert diagnose(rows+[dict(unsafe=False,above_T99_cancel=True)])['classification']=='GRADIENT_CANCELLATION_SIGNAL_ABSENT'
