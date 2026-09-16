import torch
from research_log.T058V.numerics import agreement

def test_native_dual_forward_derivative():
    x=torch.tensor([.2,.3],requires_grad=True);d=torch.tensor([.6,.8])
    def fn(v):return (v.sin()*v).sum()
    value=fn(x);rev=torch.autograd.grad(value,x)[0]@d
    with torch.autograd.forward_ad.dual_level():
        dual=torch.autograd.forward_ad.make_dual(x.detach(),d)
        primal,fwd=torch.autograd.forward_ad.unpack_dual(fn(dual))
    assert torch.equal(primal,value) and agreement(float(rev),float(fwd))
