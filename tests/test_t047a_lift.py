import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('lift_oracle',Path(__file__).parents[1]/'research_log/T047A_oracle/core.py')
o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)
def setup(low):
    active=torch.tensor([True,False,True,True]);winner=torch.tensor([0,0,1,0]);wb=o.t046.t035.wb
    box=wb.Gamma05Box(wb.SimpleNamespace(active=active,winner=winner),2)
    gate=dict(active=active.tolist(),winner=winner.tolist());bounds=dict(lower=box.lower.tolist(),upper=box.upper.tolist())
    old,project=o.t046.frozen_model(dict(gate=gate,diagnostics=dict(action_box=bounds)),low)
    with torch.no_grad():old.raw.uniform_(-.4,.4);project(old)
    return old,o.Lift(low,old.raw.detach(),gate,bounds)
def test_zero_identity_and_preclamp_order():
    torch.manual_seed(7);low=torch.rand(1,3,13,17);old,new=setup(low)
    assert torch.equal(old(low),new())
    with torch.no_grad():new.common.fill_(1.1);new.b.fill_(-.2);new.project()
    expected=torch.where(new.active[new.yy[:,None],new.xx[None,:]],torch.full_like(low,.9),low)
    assert torch.allclose(new(),expected,atol=1e-7,rtol=0)
    assert [n for n,_ in new.named_parameters()]==['b']
def test_projection_frozen_legacy_and_earliest_selection():
    torch.manual_seed(7);low=torch.rand(1,3,12,16)*.1;old,new=setup(low);raw=new.legacy_raw.clone()
    result=o.optimize(new,torch.ones_like(low),steps=30)
    assert result['lift'].shape==(31,2,2) and result['updates']==30
    assert (result['lift'].abs()<=.2).all() and torch.count_nonzero(result['lift'][:,~new.active])==0
    assert torch.equal(raw,new.legacy_raw) and torch.equal(new.b.cpu(),result['lift'][int(result['mse'].argmin())])
    _,same=setup(low);r=o.optimize(same,same().detach(),steps=3);assert r['best_step']==0
    assert o.verdict(.5,.25)=='additive-lift marginal capacity supported'
    assert 'not supported' in o.verdict(.499,.25) and 'not supported' in o.verdict(.5,.249)
