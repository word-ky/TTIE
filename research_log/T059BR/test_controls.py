import torch
from ttie.energy_model import EnergyHead
from research_log.T059BR.controls import scalar_gate,detail_control,chain_control,difference

def test_functional_control_matches_direct_chain_and_keeps_state():
    torch.manual_seed(7);head=EnergyHead().requires_grad_(False);x=torch.randn(4,28);j=torch.randn(4,28,64);truth=torch.randn(4,64);mask=torch.tensor([True,True,False,True]);before={k:v.clone() for k,v in head.state_dict().items()}
    stats,q,g=detail_control(head,x,j,truth,mask)
    v=torch.zeros(4,64,requires_grad=True);phi=x+torch.einsum('bfi,bi->bf',j,v);direct,=torch.autograd.grad(head(phi).sum(),v)
    assert chain_control(g,direct)['passed'] and stats['direction_rows']==3
    assert all(torch.equal(t,head.state_dict()[k]) for k,t in before.items()) and all(p.grad is None for p in head.parameters())

def test_differences_are_diagnostic_and_gates_are_fixed():
    x=torch.zeros(2,28);phi=x.clone();phi[0,12]=4e-5
    d=difference(x,phi);assert d['diagnostic_only'] and d['per_feature_nonidentical_counts'][12]==1 and d['max_abs']>1e-6
    assert scalar_gate(.051005665212869644,.051005665212869644)['passed']
    assert not scalar_gate(.8,.7)['passed']
    zero=torch.zeros(2,64);assert chain_control(zero,zero)['passed'];assert not chain_control(zero+3e-6,zero)['passed']
