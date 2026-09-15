import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'research_log/T046A_oracle'))
import core

def test_exact_optimizer_reuse_and_single_start_fresh_moments():
    assert core.optimize_start is core.t035.optimize_start is core.o.optimize_start
    active=torch.ones(4,dtype=torch.bool);model=core.t035.CommonRegion2(active)
    obj=core.t035.SimpleNamespace(active=active,winner=torch.zeros(4,dtype=torch.long));box=core.t035.wb.WBBox(obj,2)
    low=torch.full((1,3,12,16),.2);initial=torch.full_like(model.raw,.1)
    a=core.optimize_start(model,box,low,low*1.5,initial,steps=3)
    b=core.optimize_start(model,box,low,low*1.5,initial,steps=3)
    assert a['updates']==3 and a['raw_history'].shape==(4,1,3,2,2)
    assert torch.equal(a['raw_history'][0],initial) and torch.equal(a['raw_history'],b['raw_history'])
    assert a['best_step']==min(range(4),key=lambda j:a['history'][j])

def test_only_mean_and_median_gate():
    assert core.verdict(1,.75)=='T035 common-gain oracle materially underconverged'
    assert 'not supported' in core.verdict(.999,.9) and 'not supported' in core.verdict(1.2,.749)
