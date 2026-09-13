"""REFERENCE_ORACLE_ONLY focused checks for the isolated diagnostic."""
import importlib.util
from pathlib import Path
import torch

spec=importlib.util.spec_from_file_location('oracle',Path(__file__).parents[1]/'research_log/T025A_oracle/run_oracle.py')
o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)

def test_two_start_step_zero_retention_and_projection():
    torch.manual_seed(7);torch.set_num_threads(1)
    low=torch.rand(1,3,12,16)*.2
    active=torch.tensor([True,False,True,True]);winner=torch.tensor([0,0,1,0])
    obj=o.SimpleNamespace(active=active,winner=winner);box=o.DarkEV2Box(obj,2)
    decision=dict(gate=dict(active=active.tolist(),winner=winner.tolist()),diagnostics=dict(action_box=dict(lower=box.lower.tolist(),upper=box.upper.tolist())))
    model,box=o.frozen_model(decision,low)
    target=(low*2).clamp(0,1)
    result=o.optimize_start(model,box,low,target,torch.zeros_like(model.raw),steps=3)
    assert result['updates']==3 and len(result['history'])==4
    assert result['best_mse']<=result['history'][0]
    grid=model.physical_grid()[:,:2]
    assert (grid>=box.lower-1e-6).all() and (grid<=box.upper+1e-6).all()
    assert torch.equal(model(low)[:,:,:6,8:],low[:,:,:6,8:])
    # Exact zero-loss selected start must survive all later Adam steps.
    selected=model.raw.detach().clone();reference=model(low).detach()
    result=o.optimize_start(model,box,low,reference,selected,steps=3)
    assert result['best_step']==0 and result['best_mse']==0
    assert torch.equal(result['best_raw'],selected)
