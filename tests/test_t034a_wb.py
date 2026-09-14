import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('wb_oracle',Path(__file__).parents[1]/'research_log/T034A_oracle/core.py')
o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)

def setup():
    active=torch.tensor([True,False,True,True]);winner=torch.tensor([0,0,1,0])
    box=o.Gamma05Box(o.SimpleNamespace(active=active,winner=winner),2)
    return dict(gate=dict(active=active.tolist(),winner=winner.tolist()),diagnostics=dict(action_box=dict(lower=box.lower.tolist(),upper=box.upper.tolist())))

def test_identity_wb_exact_renderer_and_inactive_identity():
    torch.manual_seed(7);image=torch.rand(1,3,13,17)
    old,oldbox=o.prior.frozen_model(setup(),image);new,box=o.frozen_model(setup(),image)
    with torch.no_grad():old.raw.uniform_(-.4,.4);oldbox(old);new.raw.copy_(o.extend_start(old.raw))
    assert (old(image)-new(image)).abs().max()<=1e-6
    with torch.no_grad():new.raw[:,2:]=torch.tensor([.8,-.5,.3]).reshape(1,3,1,1);box(new)
    assert torch.equal(new(image)[:,:,:6,8:],image[:,:,:6,8:])
    assert torch.equal(new.physical_grid()[:,2:5,0,1],torch.ones(1,3))
    assert torch.equal(new.physical_grid()[:,5],torch.ones(1,2,2))

def test_reuse_optimizer_and_wb_bounds():
    assert o.optimize_start is o.prior.optimize_start
    torch.manual_seed(7);low=torch.rand(1,3,12,16)*.2;model,box=o.frozen_model(setup(),low)
    target=(low*torch.tensor([1.3,.8,1.1]).reshape(1,3,1,1)).clamp(0,1)
    r=o.optimize_start(model,box,low,target,torch.zeros_like(model.raw),steps=3)
    assert r['updates']==3 and r['raw_history'].shape==(4,1,5,2,2)
    assert r['best_mse']<=r['initial_mse'] and torch.isfinite(r['raw_history']).all()
    wb=model.physical_grid()[:,2:5];assert (wb>=.5).all() and (wb<=2).all()
    assert model.raw[:,2:].abs().sum()>0
