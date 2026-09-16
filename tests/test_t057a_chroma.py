import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('band',Path(__file__).parents[1]/'research_log/T057A_oracle/core.py');o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)
spec=importlib.util.spec_from_file_location('prior_test',Path(__file__).with_name('test_t054a_detail.py'));prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
def setup():
 _,m=prior.setup()
 with torch.no_grad():m.v.normal_(-1,.3)
 gate=dict(active=[bool(x) for x in m.mask[::8,::10].flatten()],winner=[0,0,1,0]);wb=prior.prior.prior.o.t046.t035.wb;box=wb.Gamma05Box(wb.SimpleNamespace(**{k:torch.tensor(v) for k,v in gate.items()}),2)
 return m,o.Chroma(m.low,m.raw,gate,dict(lower=box.lower.tolist(),upper=box.upper.tolist()),m.lift,m.q,m.u,m.b,m.v)
def test_analytic_chroma_projection():
 detail=torch.tensor([1.,0.,0.]).reshape(1,3,1,1);expected=torch.tensor([2/3,-1/3,-1/3]).reshape(1,3,1,1)
 assert torch.allclose(o.chroma_basis(detail),expected,atol=1e-7,rtol=0)
 mono=torch.full((1,3,9,11),.25);assert torch.count_nonzero(o.chroma_basis(mono))==0
 torch.manual_seed(11);d=torch.rand(1,3,13,17)*2-1;c=o.chroma_basis(d);assert c.sum(dim=1).abs().max()<=1e-6
 assert torch.allclose(o.chroma_basis(c),c,atol=1e-7,rtol=0)
def test_only_w_and_zero_identity():
 old,m=setup();assert torch.equal(m(),old()) and torch.equal(m.one,old()) and list(dict(m.named_parameters()))==['w_c']
 frozen={k:v.detach().clone() for k,v in m.named_buffers()};hist=o.optimize(m,m.one-o.prior.blur(m.d_chroma),steps=4)
 assert hist['w_c'].shape==(5,1,1,8,8) and torch.count_nonzero(hist['w_c'][0])==0 and hist['best_step']==int(hist['mse'].argmin())
 for name,value in m.named_buffers():assert torch.equal(value,frozen[name])
 assert torch.equal(m()[:,:,~m.mask],m.one[:,:,~m.mask])
 assert o.verdict(.5,.25,.020)=='chroma-detail capacity materially supported'
 assert o.verdict(.5,.25,.019)=='not supported under fixed probe'
