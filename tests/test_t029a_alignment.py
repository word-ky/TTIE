import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research_log/T029A_alignment'))
from common import alignment,summarize,classify
from reference_gradient import reference_gradient
from ttie.semantic_ttt import Region2

def test_active_alignment_and_fixed_thresholds():
    mask=torch.tensor([True,True,False])
    a=alignment(torch.tensor([1.,0.,999.]),torch.tensor([-1.,0.,999.]),mask)
    assert a['active_count']==2 and a['cosine']==-1 and not a['positive_dot']
    zero=alignment(torch.zeros(3),torch.ones(3),mask)
    s=summarize([a,zero]);assert s['nondegenerate']==1 and s['energy_zero_fraction']==.5
    assert classify(s)=='strong field-direction mismatch'
    assert classify(dict(nondegenerate=1,cosine_median=.249,positive_dot_fraction=.9))=='weak/mixed field alignment'
    assert classify(dict(nondegenerate=1,cosine_median=.25,positive_dot_fraction=.75))=='broad field-direction alignment'

def test_reference_gradient_only_differentiates_and_preserves_raw():
    torch.manual_seed(7)
    model=Region2(torch.tensor([True,False,True,False]))
    low=torch.rand(1,3,12,16)*.7+.1;reference=low*.8
    before=model.raw.detach().clone();version=model.raw._version
    g=reference_gradient(model,low,reference)
    assert torch.isfinite(g).all() and model.raw.grad is None
    assert torch.equal(before,model.raw) and model.raw._version==version
    assert torch.equal(g[:,:,:,1],torch.zeros_like(g[:,:,:,1]))
    direction=torch.zeros_like(before);direction[0,0,0,0]=1
    eps=1e-3
    losses=[]
    for sign in [-1,1]:
        y=torch.func.functional_call(model,{'raw':before+sign*eps*direction},(low,))
        losses.append((y.double()-reference.double()).square().mean())
    finite_difference=(losses[1]-losses[0])/(2*eps)
    torch.testing.assert_close(finite_difference,g[0,0,0,0].double(),rtol=.002,atol=1e-6)
