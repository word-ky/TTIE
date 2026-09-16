import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('detail',Path(__file__).parents[1]/'research_log/T054A_oracle/core.py');o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)
spec=importlib.util.spec_from_file_location('prior',Path(__file__).with_name('test_t052a_spatial_affine.py'));prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
def setup():
    torch.manual_seed(7);low=torch.rand(1,3,16,20)*.3;_,m=prior.setup(low)
    with torch.no_grad():m.u.normal_(0,.2);m.b.uniform_(-.2,.2)
    gate=dict(active=[bool(x) for x in m.mask[::8,::10].flatten()],winner=[0,0,1,0]);wb=prior.prior.o.t046.t035.wb;box=wb.Gamma05Box(wb.SimpleNamespace(**{k:torch.tensor(v) for k,v in gate.items()}),2)
    return m,o.Detail(low,m.raw,gate,dict(lower=box.lower.tolist(),upper=box.upper.tolist()),m.lift,m.q,m.u,m.b)
def test_analytic_binomial_impulse_constant_reflect():
    w=torch.tensor([1,4,6,4,1],dtype=torch.float32)/16;y=torch.zeros(1,3,9,11);y[0,0,4,5]=1;y[0,1,4,5]=.5
    expected=torch.zeros_like(y);expected[0,0,2:7,3:8]=w[:,None]*w[None,:];expected[0,1]=expected[0,0]*.5
    assert torch.equal(o.blur(y),expected) and torch.equal(y-o.blur(y),y-expected)
    constant=torch.full_like(y,.25);assert torch.equal(o.blur(constant),constant)
    ramp=torch.arange(7,dtype=torch.float32).reshape(1,1,1,7).expand(1,3,7,7)/16;blur=o.blur(ramp)
    assert torch.all(blur[...,0]==3/64) and torch.all(blur[...,3]==3/16)
def test_zero_identity_raw_interpolation_then_tanh_and_inactive():
    base,m=setup();assert torch.equal(base(),m()) and list(dict(m.named_parameters()))==['v']
    with torch.no_grad():m.v.copy_(torch.linspace(-2,2,64).reshape(1,1,8,8))
    expected=torch.nn.functional.interpolate(m.v,size=(16,20),mode='bilinear',align_corners=False).tanh()
    assert torch.equal(m.coefficient(),expected) and expected.min()>=-1 and expected.max()<=1
    assert not torch.equal(expected,torch.nn.functional.interpolate(m.v.tanh(),size=(16,20),mode='bilinear',align_corners=False))
    assert torch.equal(m()[:,:,~m.mask],m.y0[:,:,~m.mask])
def test_only_v_updates_and_earliest_minimum():
    _,m=setup();frozen={k:v.clone() for k,v in m.named_buffers()};target=o.blur(m.y0);h=o.optimize(m,target,steps=5)
    assert h['v'].shape==(6,1,1,8,8) and torch.count_nonzero(h['v'][0])==0 and torch.equal(m.v,h['v'][int(h['mse'].argmin())])
    assert all(torch.equal(v,dict(m.named_buffers())[k]) for k,v in frozen.items())
    _,m=setup();h=o.optimize(m,m.y0,steps=2);assert h['best_step']==0
    assert o.verdict(.5,.25,.020)=='local-detail capacity supported'
    assert all(o.verdict(*v)=='not supported under fixed local-detail probe' for v in [(.49,.25,.02),(.5,.24,.02),(.5,.25,.019)])
