import torch
from research_log.T059S.core import Bridge,step,namespace

def model():
    torch.manual_seed(12)
    return Bridge(torch.rand(1,3,17,21),torch.tensor([True,False,True,False]),torch.zeros(1,2,2,2))

def test_exact_t054_first_step():
    a=model();b=model();target=torch.zeros_like(a.y0)
    loss=(a().double()-target.double()).square().mean();g,=torch.autograd.grad(loss,a.v)
    original=namespace['optimize'](b,target,steps=1)
    optimizer=step(a,g)
    assert torch.equal(a.v,original['v'][1])
    assert optimizer.state[a.v]['step']==1

def test_renderer_and_zero_gradient():
    a=model();assert torch.equal(a(),a.y0)
    step(a,torch.zeros_like(a.v));assert torch.count_nonzero(a.v)==0 and torch.equal(a(),a.y0)
    assert torch.equal(a.detail,a.y0-namespace['blur'](a.y0))

def test_inactive_identity():
    a=model();step(a,torch.arange(64).float().reshape_as(a.v)-32)
    mask=a.mask.expand_as(a.y0);assert torch.equal(a()[~mask],a.y0[~mask])
    assert torch.isfinite(a()).all() and a.v.abs().max()<=.05
