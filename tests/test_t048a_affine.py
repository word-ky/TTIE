import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('affine_oracle',Path(__file__).parents[1]/'research_log/T048A_oracle/core.py');o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)
spec=importlib.util.spec_from_file_location('prior_lift',Path(__file__).parents[1]/'research_log/T047A_oracle/core.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
def setup(low):
    active=torch.tensor([True,False,True,True]);winner=torch.tensor([0,0,1,0]);wb=o.t046.t035.wb
    box=wb.Gamma05Box(wb.SimpleNamespace(active=active,winner=winner),2)
    gate=dict(active=active.tolist(),winner=winner.tolist());bounds=dict(lower=box.lower.tolist(),upper=box.upper.tolist())
    old,project=o.t046.frozen_model(dict(gate=gate,diagnostics=dict(action_box=bounds)),low)
    with torch.no_grad():old.raw.uniform_(-.4,.4);project(old)
    lift=prior.Lift(low,old.raw.detach(),gate,bounds)
    with torch.no_grad():lift.b.uniform_(-.1,.1);lift.project()
    return lift,o.Affine(low,old.raw.detach(),gate,bounds,lift.b.detach())
def test_exact_initialization_group_lrs_and_gate():
    torch.manual_seed(7);low=torch.rand(1,3,13,17);old,new=setup(low)
    assert torch.equal(old(),new()) and torch.equal(old.legacy_raw,new.raw()) and torch.equal(old.b,new.b)
    assert set(dict(new.named_parameters()))=={'b','gain_raw'}
    opt=o.optimizer_for(new);assert [g['lr'] for g in opt.param_groups]==[.05,.01]
    assert torch.equal(new()[:,:,:6,8:],low[:,:,:6,8:])
def test_joint_updates_ev_gamma_frozen_projection_selection():
    torch.manual_seed(7);low=torch.rand(1,3,12,16)*.15;old,new=setup(low);ev=new.ev_gamma.clone();g=new.gain_raw.detach().clone();b=new.b.detach().clone()
    r=o.optimize(new,torch.ones_like(low)*.8,steps=25)
    assert r['gain_raw'].shape==(26,1,1,2,2) and r['lift'].shape==(26,2,2) and r['updates']==25
    assert torch.equal(new.ev_gamma,ev) and torch.equal(r['gain_raw'][0],g) and torch.equal(r['lift'][0],b)
    assert not torch.equal(r['gain_raw'][-1],g) and not torch.equal(r['lift'][-1],b)
    assert (r['lift'].abs()<=.2).all() and torch.count_nonzero(r['lift'][:,~new.active])==0
    step=int(r['mse'].argmin());assert torch.equal(new.b,r['lift'][step]) and torch.equal(new.gain_raw,r['gain_raw'][step])
    old,same=setup(low);same_r=o.optimize(same,same().detach(),steps=3);assert same_r['best_step']==0
    assert o.verdict(.5,.25)=='post-gamma affine coupling materially supported'
    assert 'not supported' in o.verdict(.499,.25) and 'not supported' in o.verdict(.5,.249)
