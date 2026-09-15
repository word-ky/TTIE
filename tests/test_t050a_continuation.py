import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('continuation',Path(__file__).parents[1]/'research_log/T050A_oracle/core.py');o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)
spec=importlib.util.spec_from_file_location('prior_test',Path(__file__).with_name('test_t049a_tone.py'));prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
def test_renderer_reuse_and_zero_start_optimizer_equivalence():
    assert o.Tone is o.t049.Tone and o.knots is o.t049.knots
    torch.manual_seed(7);low=torch.rand(1,3,12,16)*.2;_,one=prior.setup(low);_,two=prior.setup(low)
    with torch.no_grad():two.raw.copy_(one.raw);two.lift.copy_(one.lift)
    # Use identical model buffers, including cached immutable pixels.
    two.load_state_dict(one.state_dict());reference=low*.7
    a=o.t049.optimize(one,reference,steps=4);b=o.optimize(two,reference,steps=4)
    assert torch.equal(a['q'],b['q']) and torch.equal(a['mse'],b['mse'])
def test_nonzero_start_fresh_moments_and_frozen_coordinates():
    torch.manual_seed(8);low=torch.rand(1,3,12,16)*.4;_,one=prior.setup(low);_,two=prior.setup(low)
    with torch.no_grad():one.q.normal_(0,.2);one.q[~one.active.flatten()]=0
    two.load_state_dict(one.state_dict());initial=one.q.detach().clone();raw=one.raw.clone();lift=one.lift.clone();reference=low*.8
    a=o.optimize(one,reference,steps=5);b=o.optimize(two,reference,steps=5)
    assert a['q'].shape==(6,4,8) and torch.equal(a['q'][0],initial) and torch.equal(a['q'],b['q']) and torch.equal(a['mse'],b['mse'])
    assert torch.equal(one.raw,raw) and torch.equal(one.lift,lift) and torch.equal(one.q,a['q'][int(a['mse'].argmin())])
    assert o.verdict(1,.5)=='material monotonic-tone underconvergence supported'
    assert 'not supported' in o.verdict(.99,.5) and 'not supported' in o.verdict(1,.49)
