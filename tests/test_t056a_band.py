import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('band',Path(__file__).parents[1]/'research_log/T056A_oracle/core.py');o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)
spec=importlib.util.spec_from_file_location('prior_test',Path(__file__).with_name('test_t054a_detail.py'));prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
def setup():
 _,m=prior.setup()
 with torch.no_grad():m.v.normal_(-1,.3)
 gate=dict(active=[bool(x) for x in m.mask[::8,::10].flatten()],winner=[0,0,1,0]);wb=prior.prior.prior.o.t046.t035.wb;box=wb.Gamma05Box(wb.SimpleNamespace(**{k:torch.tensor(v) for k,v in gate.items()}),2)
 return m,o.Band(m.low,m.raw,gate,dict(lower=box.lower.tolist(),upper=box.upper.tolist()),m.lift,m.q,m.u,m.b,m.v)
def test_analytic_nine_tap_and_band():
 k=torch.tensor([1,8,28,56,70,56,28,8,1],dtype=torch.float32)/256;y=torch.zeros(1,3,13,13);y[0,0,6,6]=1;expected=torch.zeros_like(y);expected[0,0,2:11,2:11]=k[:,None]*k[None,:]
 assert torch.equal(o.blur9(y),expected)
 k5=torch.tensor([1,4,6,4,1],dtype=torch.float32)/16;five=torch.zeros_like(y);five[0,0,4:9,4:9]=k5[:,None]*k5[None,:]
 assert torch.equal(o.prior.blur(y)-o.blur9(y),five-expected)
 constant=torch.full_like(y,.25);assert torch.equal(o.blur9(constant),constant)
 ramp=torch.arange(11,dtype=torch.float32).reshape(1,1,1,11).expand(1,3,11,11)/16;assert torch.all(o.blur9(ramp)[...,0]==35/512)
def test_only_w_and_zero_identity():
 old,m=setup();assert torch.equal(m(),old()) and torch.equal(m.one,old()) and list(dict(m.named_parameters()))==['w']
 frozen={k:v.detach().clone() for k,v in m.named_buffers()};hist=o.optimize(m,m.one-o.prior.blur(m.d2),steps=4)
 assert hist['w'].shape==(5,1,1,8,8) and torch.count_nonzero(hist['w'][0])==0 and hist['best_step']==int(hist['mse'].argmin())
 for name,value in m.named_buffers():assert torch.equal(value,frozen[name])
 assert torch.equal(m()[:,:,~m.mask],m.one[:,:,~m.mask])
 assert o.verdict(.5,.25,.020)=='coarser detail band materially supported'
 assert o.verdict(.5,.25,.019)=='not supported under fixed probe'
