import torch
from research_log.T058V.numerics import boundaries,ulp,agreement

def test_genuine_forward_mode_and_diagnostics():
    x=torch.tensor([.2,.3],requires_grad=True);d=torch.tensor([.6,.8])
    def fn(t):return (t.sin()*t).sum()
    rev=torch.autograd.grad(fn(x),x)[0]@d
    value,fwd=torch.func.jvp(fn,(x,),(d,))
    assert agreement(float(rev),float(fwd)) and torch.equal(value,fn(x))
    y=torch.tensor([0.,1.,torch.nextafter(torch.tensor(1.),torch.tensor(0.)).item(),.5]).reshape(1,1,2,2)
    b=boundaries(y,torch.ones(2,2,dtype=torch.bool))
    assert b['exact_one']==1 and b['within_one_ulp_one']==2 and b['exact_zero']==1
    assert ulp(-4.7963)==2**-21
