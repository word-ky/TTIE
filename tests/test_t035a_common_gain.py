import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('common_oracle',Path(__file__).parents[1]/'research_log/T035A_oracle/core.py')
o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)

def setup():
    active=torch.tensor([True,False,True,True]);winner=torch.tensor([0,0,1,0])
    box=o.wb.Gamma05Box(o.SimpleNamespace(active=active,winner=winner),2)
    return dict(gate=dict(active=active.tolist(),winner=winner.tolist()),diagnostics=dict(action_box=dict(lower=box.lower.tolist(),upper=box.upper.tolist())))

def test_identity_and_tied_wb_renderers():
    torch.manual_seed(7);image=torch.rand(1,3,13,17)
    old,oldbox=o.prior.frozen_model(setup(),image);new,box=o.frozen_model(setup(),image);full,_=o.wb.frozen_model(setup(),image)
    with torch.no_grad():old.raw.uniform_(-.4,.4);oldbox(old);new.raw.copy_(o.extend_start(old.raw))
    assert (old(image)-new(image)).abs().max()<=1e-6
    with torch.no_grad():
        new.raw[:,2:]=torch.tensor([.8,-.5,.3,-.2]).reshape(1,1,2,2);box(new)
        full.raw.copy_(torch.cat([new.raw[:,:2],new.raw[:,2:3].expand(-1,3,-1,-1)],1))
    assert (full(image)-new(image)).abs().max()<=1e-6
    assert torch.equal(new(image)[:,:,:6,8:],image[:,:,:6,8:])
    assert torch.equal(new.physical_grid()[:,2:5,0,1],torch.ones(1,3))

def test_only_one_gain_and_exact_optimizer_reuse():
    assert o.optimize_start is o.prior.optimize_start
    torch.manual_seed(7);low=torch.rand(1,3,12,16)*.2;model,box=o.frozen_model(setup(),low)
    assert model.raw.numel()==12
    r=o.optimize_start(model,box,low,low*1.3,torch.zeros_like(model.raw),steps=3)
    assert r['raw_history'].shape==(4,1,3,2,2) and r['updates']==3
    assert r['best_mse']<=r['initial_mse'] and model.raw[:,2:].abs().sum()>0
    grid=model.physical_grid();assert torch.equal(grid[:,2],grid[:,3]) and torch.equal(grid[:,3],grid[:,4])
    assert (grid[:,2:5]>=.5).all() and (grid[:,2:5]<=2).all()
