import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('extension',Path(__file__).parents[1]/'research_log/T055A_oracle/core.py');o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)
spec=importlib.util.spec_from_file_location('prior_detail_test',Path(__file__).with_name('test_t054a_detail.py'));prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
def test_nonzero_continuation_fresh_adam_and_frozen_buffers():
    _,m=prior.setup();assert o.Detail is o.t054.Detail
    with torch.no_grad():m.v.fill_(-.4)
    initial=m.v.detach().clone();frozen={k:v.clone() for k,v in m.named_buffers()};target=o.blur(m.y0)
    h=o.optimize(m,target,steps=4)
    assert torch.equal(h['v'][0],initial) and h['v'].shape==(5,1,1,8,8) and h['updates']==4
    assert not torch.equal(h['v'][1],initial)
    assert h['best_step']==int(h['mse'].argmin()) and torch.equal(m.v,h['v'][h['best_step']])
    for name,value in m.named_buffers():assert torch.equal(value,frozen[name])
    # Independently reproduce the first fresh-Adam update from accepted state.
    with torch.no_grad():m.v.copy_(initial)
    optimizer=torch.optim.Adam([m.v],lr=.05);loss=(m().double()-target.double()).square().mean();optimizer.zero_grad();loss.backward();optimizer.step()
    assert torch.equal(m.v,h['v'][1])
def test_continuation_earliest_zero_loss_and_gate():
    _,m=prior.setup()
    with torch.no_grad():m.v.fill_(-.3)
    initial=m.v.detach().clone();h=o.optimize(m,m().detach(),steps=3)
    assert h['best_step']==0 and torch.equal(h['v'][0],initial) and torch.equal(m.v,initial)
    assert o.verdict(.25,.10,.010)=='material local-detail underconvergence supported'
    for args in [(.249,.10,.010),(.25,.099,.010),(.25,.10,.009)]:assert o.verdict(*args)=='material local-detail underconvergence not supported under fixed extension'
