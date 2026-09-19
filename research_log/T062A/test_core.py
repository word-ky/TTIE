import torch
from types import SimpleNamespace
from research_log.T062A.core import losses,trajectory
from research_log.T062A.evaluate import independent_loss

def test_constant_neutral():
    x=torch.full((1,3,32,32),.6,dtype=torch.float64)
    assert torch.allclose(losses(x,x),torch.zeros(3,dtype=torch.float64),atol=1e-28)

def test_independent_equations_and_gradients():
    torch.manual_seed(7);x=torch.rand(1,3,32,48,dtype=torch.float64);y=torch.rand_like(x,requires_grad=True)
    value=losses(x,y)@y.new_tensor([1,10,5]);other=independent_loss(x.numpy(),y.detach().numpy())
    assert abs(float(value)-other)<1e-12
    g,=torch.autograd.grad(value,y);assert torch.isfinite(g).all() and g.abs().sum()>0

def test_unchanged_action_and_selection():
    x=torch.full((1,3,32,32),.2)
    gate=SimpleNamespace(active=torch.ones(4,dtype=torch.bool),winner=torch.zeros(4,dtype=torch.long))
    t=trajectory(x,gate)
    assert t['states'].shape==(41,1,3,2,2) and torch.count_nonzero(t['states'][0])==0
    assert len(t['values'])==41 and len(t['gradients'])==40
    assert t['selected_step']==min(range(41),key=lambda i:(t['values'][i],i))
