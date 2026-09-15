import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('tone_oracle',Path(__file__).parents[1]/'research_log/T049A_oracle/core.py');o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)
def setup(low):
    active=torch.tensor([True,False,True,True]);winner=torch.tensor([0,0,1,0]);wb=o.t046.t035.wb
    box=wb.Gamma05Box(wb.SimpleNamespace(active=active,winner=winner),2);gate=dict(active=active.tolist(),winner=winner.tolist());bounds=dict(lower=box.lower.tolist(),upper=box.upper.tolist())
    old,project=o.t046.frozen_model(dict(gate=gate,diagnostics=dict(action_box=bounds)),low)
    with torch.no_grad():old.raw.uniform_(-.4,.4);project(old)
    lift=low.new_tensor([[.1,0],[-.05,.02]]);affine=o.t048.Affine(low,old.raw.detach(),gate,bounds,lift)
    return affine,o.Tone(low,old.raw.detach(),gate,bounds,lift)
def test_identity_endpoints_monotonicity_and_gate():
    torch.manual_seed(7);low=torch.rand(1,3,13,17);old,new=setup(low)
    assert (old()-new()).abs().max()<=1e-6 and torch.equal(o.knots(new.q),torch.arange(9).expand(4,9)/8)
    assert list(dict(new.named_parameters()))==['q'] and new.q.shape==(4,8)
    with torch.no_grad():new.q.normal_(0,4)
    y=o.knots(new.q);assert (y.diff(dim=-1)>=0).all() and (y[:,0]==0).all() and (y[:,-1]==1).all()
    assert torch.equal(new()[:,:,:6,8:],low[:,:,:6,8:])
    assert new().min()>=0 and new().max()<=1
def test_only_tone_updates_frozen_affine_and_earliest():
    torch.manual_seed(7);low=torch.rand(1,3,12,16)*.3;old,new=setup(low);raw=new.raw.clone();lift=new.lift.clone()
    r=o.optimize(new,torch.ones_like(low)*.6,steps=6)
    assert r['q'].shape==(7,4,8) and r['updates']==6 and torch.count_nonzero(r['q'][0])==0
    assert torch.equal(new.raw,raw) and torch.equal(new.lift,lift) and torch.count_nonzero(r['q'][:,~new.active.flatten()])==0
    assert torch.equal(new.q,r['q'][int(r['mse'].argmin())])
    old,same=setup(low);r=o.optimize(same,same().detach(),steps=3);assert r['best_step']==0
    assert o.verdict(2,1)=='SOTA-scale monotonic-tone capacity supported'
    assert 'not supported' in o.verdict(1.99,1) and 'not supported' in o.verdict(2,.99)
