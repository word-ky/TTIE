import torch
from research_log.T058A_tangent.core import Detail,blur,fd,aggregate,classify,alignment

def test_detail_derivative_and_identity():
    torch.manual_seed(7); y=torch.rand(1,3,17,23)*.6+.2
    active=torch.tensor([True,False,True,False]);m=Detail(y,active,torch.zeros(1,2,2,2))
    assert torch.equal(m(),y) and list(dict(m.named_parameters()))==['v']
    assert torch.allclose(blur(torch.ones_like(y)),torch.ones_like(y))
    loss=m().double().square().mean();g,=torch.autograd.grad(loss,m.v)
    receipt=fd(m,lambda v:m.render(v).double().square().mean(),g)
    assert receipt['passed'] and torch.count_nonzero(m.v)==0 and m.v.grad is None
    with torch.no_grad():out=m.render(torch.ones_like(m.v))
    assert torch.equal(out[...,~m.mask],y[...,~m.mask])
    no=Detail(y,torch.zeros(4,dtype=torch.bool),m.grid)
    gn,=torch.autograd.grad(no().sum(),no.v);assert torch.count_nonzero(gn)==0

def test_degeneracy_and_gate():
    v=torch.ones(1,1,8,8);mask=torch.ones_like(v,dtype=torch.bool)
    good=alignment(v,v,mask);zero=alignment(v,v*0,mask)
    s=aggregate([good,zero]);assert s['nondegenerate']==1 and s['degenerate']==1
    assert classify(s)=='frozen-energy detail tangent supported on source'
    s['cosine_median']=.499;assert classify(s)=='frozen-energy detail tangent not ready'
