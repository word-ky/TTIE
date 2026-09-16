import torch
from research_log.T058X.criteria import compare

def test_criteria_and_flag_restoration():
    assert compare(-4.,-4.+2**-21,'primal')['passed']
    assert not compare(.01,.011,'derivative')['passed']
    flag=torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        assert torch.backends.mha.get_fastpath_enabled() is False
        x=torch.tensor([.2,.3],requires_grad=True);direction=torch.tensor([.6,.8])
        def fn(v):return (v.sin()*v).sum()
        rev=torch.autograd.grad(fn(x),x)[0]@direction
        with torch.autograd.forward_ad.dual_level():
            primal,tangent=torch.autograd.forward_ad.unpack_dual(fn(torch.autograd.forward_ad.make_dual(x.detach(),direction)))
        assert compare(float(rev),float(tangent),'derivative')['passed']
    finally:torch.backends.mha.set_fastpath_enabled(flag)
    assert torch.backends.mha.get_fastpath_enabled()==flag
