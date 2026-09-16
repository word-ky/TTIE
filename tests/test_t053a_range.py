import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('range_closure',Path(__file__).parents[1]/'research_log/T053A_oracle/core.py');o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)
spec=importlib.util.spec_from_file_location('prior_spatial',Path(__file__).with_name('test_t052a_spatial_affine.py'));prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
def setup():
    torch.manual_seed(7);low=torch.rand(1,3,16,20)*.3;_,m=prior.setup(low)
    with torch.no_grad():m.u.normal_(0,.2);m.b.uniform_(-.2,.2)
    gate=dict(active=[bool(x) for x in m.mask[::8,::10].flatten()],winner=[0,0,1,0])
    tone=prior.prior.o.t046.t035.wb
    box=tone.Gamma05Box(tone.SimpleNamespace(**{k:torch.tensor(v) for k,v in gate.items()}),2)
    n=o.Spatial(low,m.raw,gate,dict(lower=box.lower.tolist(),upper=box.upper.tolist()),m.lift,m.q,m.u,m.b)
    return m,n
def test_nonzero_selected_start_identity_frozen_u_and_gate():
    m,n=setup();assert torch.equal(m(),n()) and list(dict(n.named_parameters()))==['b'] and not n.u.requires_grad
    assert torch.equal(n()[:,:,~n.mask],n.low[:,:,~n.mask])
def test_projection_frozen_buffers_and_earliest_selection():
    m,n=setup();frozen={k:v.clone() for k,v in n.named_buffers()};initial=m.b.detach().clone();result=o.optimize(n,torch.ones_like(n.low),steps=70)
    assert torch.equal(result['b'][0],initial) and result['b'].shape==(71,1,1,8,8)
    assert result['b'].min()>=-.4 and result['b'].max()<=.4 and torch.any(result['b']==.4)
    assert torch.equal(n.b,result['b'][int(result['mse'].argmin())]) and all(torch.equal(v,dict(n.named_buffers())[k]) for k,v in frozen.items())
    m,n=setup();target=n().detach();h=o.optimize(n,target,steps=2);assert h['best_step']==0
    assert o.verdict(.5,.25,0)=='additive-range bottleneck supported'
    assert all(o.verdict(*v)=='not supported under fixed range-closure probe' for v in [(.49,.25,0),(.5,.24,0),(.5,.25,-.001)])
