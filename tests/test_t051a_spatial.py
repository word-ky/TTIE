import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('spatial',Path(__file__).parents[1]/'research_log/T051A_oracle/core.py');o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)
spec=importlib.util.spec_from_file_location('prior',Path(__file__).with_name('test_t049a_tone.py'));prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
def setup(low):
    _,tone=prior.setup(low)
    with torch.no_grad():tone.q.normal_(0,.3);tone.q[~tone.active.flatten()]=0
    gate=dict(active=tone.active.flatten().tolist(),winner=[0,0,1,0])
    box=prior.o.t046.t035.wb.Gamma05Box(prior.o.t046.t035.wb.SimpleNamespace(**{k:torch.tensor(v) for k,v in gate.items()}),2)
    return tone,o.Spatial(low,tone.raw,gate,dict(lower=box.lower.tolist(),upper=box.upper.tolist()),tone.lift,tone.q)
def test_zero_identity_gate_and_interpolation():
    torch.manual_seed(7);low=torch.rand(1,3,13,17)*.4;tone,m=setup(low)
    assert torch.equal(tone(),m()) and list(dict(m.named_parameters()))==['u']
    with torch.no_grad():m.u.copy_(torch.linspace(-1,1,64).reshape(1,1,8,8))
    expected=torch.nn.functional.interpolate(2*m.u.tanh(),size=(13,17),mode='bilinear',align_corners=False)
    assert torch.equal(m.ev(),expected) and m.ev().min()>=-2 and m.ev().max()<=2
    assert torch.equal(m()[:,:,~m.mask],low[:,:,~m.mask])
def test_frozen_coordinates_fresh_adam_earliest_and_gate():
    torch.manual_seed(8);low=torch.rand(1,3,12,16)*.3;_,m=setup(low);_,n=setup(low);n.load_state_dict(m.state_dict())
    frozen={k:v.clone() for k,v in m.named_buffers()};a=o.optimize(m,low*.8,steps=4);b=o.optimize(n,low*.8,steps=4)
    assert a['u'].shape==(5,1,1,8,8) and torch.count_nonzero(a['u'][0])==0 and torch.equal(a['u'],b['u'])
    assert torch.equal(m.u,a['u'][int(a['mse'].argmin())]) and all(torch.equal(v,dict(m.named_buffers())[k]) for k,v in frozen.items())
    assert o.verdict(1.5,.75,0)=='smooth spatial illumination capacity supported'
    assert all(o.verdict(*v)=='not supported under fixed probe' for v in [(1.49,.75,0),(1.5,.74,0),(1.5,.75,-.001)])
