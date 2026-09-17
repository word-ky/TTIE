import torch
from types import SimpleNamespace
from research_log.T058AB.chain import vector_checks,full_chain
from research_log.T058A_tangent.core import Detail
from research_log.T058Z.shadow import shadow_energy

def test_vector_acceptance_and_zero_case():
    g=torch.arange(64,dtype=torch.float64).reshape(1,1,8,8)/100
    assert all(c['passed'] for c in vector_checks(g.float(),g,g)['criteria'].values())
    bad=g.clone();bad[0,0,0,0]+=.01
    assert not vector_checks(g.float(),g,bad)['criteria']['chain']['passed']
    assert not vector_checks(-g.float(),g,g)['criteria']['orientation']['passed']
    zero=torch.zeros_like(g)
    assert vector_checks(zero.float(),zero,zero)['criteria']['orientation']['passed']
    assert not vector_checks(zero.float(),g,g)['criteria']['orientation']['passed']

def test_full_chain_with_frozen_inactive_pixels():
    y=torch.linspace(0,1,3*8*8,dtype=torch.float64).reshape(1,3,8,8);y[:,:,0,:]=0
    model=Detail(y,torch.tensor([True,False,True,False]),torch.zeros(1,2,2,2,dtype=torch.float64))
    obj=SimpleNamespace(active=torch.tensor([True,False,True,False]),winner=torch.zeros(4,dtype=torch.long),evidence=torch.zeros(4,dtype=torch.float64),calibration={'tau':[0.,0.],'scale':[1.,1.]},scorer=lambda x:x.mean().expand(4,2))
    head=lambda phi:phi.sum()
    E,out,_,_=shadow_energy(model,obj,head);g,=torch.autograd.grad(E,model.v)
    d=(torch.arange(64).reshape_as(model.v)%2*2-1).double()/8
    result=full_chain(model,obj,head,d,out,float(E.detach()),float(E.detach()),g.float(),g)
    assert all(c['passed'] for c in result['criteria'].values())
    assert result['counts']['inactive']>0 and result['counts']['lower']>0
