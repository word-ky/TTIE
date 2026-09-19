import torch
from ttie.common_gain import CommonRegion2
from research_log.T060B.core import Gain,select

def test_exact_renderer_and_gain_only_derivative():
    torch.manual_seed(7);low=torch.rand(1,3,11,13);raw=torch.randn(1,3,2,2)*.2;raw[:,2]=0;active=torch.tensor([True,False,True,False])
    old=CommonRegion2(active);old.raw.data.copy_(raw);new=Gain(low,raw,active)
    assert torch.equal(old(low),new())
    g0,=torch.autograd.grad(old(low).sum(),old.raw);g1,=torch.autograd.grad(new().sum(),new.gain)
    assert torch.equal(g0[:,2:3],g1) and torch.count_nonzero(g1.flatten()[~active])==0
    assert [n for n,p in new.named_parameters() if p.requires_grad]==['gain']
    assert torch.equal(new.fixed,raw) and torch.equal(new.legacy.raw,raw)

def test_selection_uses_only_gate():
    rows=[dict(index=i,bank_index=i,image_id=7,gate={'active':[bool(i%2),False,False,False]},reference_norm=999-i) for i in range(80)]
    table,chosen=select(rows)
    assert len(table)==80 and [r['index'] for r in chosen]==list(range(1,80,2))
    for r in rows:r['reference_norm']=-100
    assert select(rows)[0]==table
