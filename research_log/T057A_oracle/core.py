"""REFERENCE_ORACLE_ONLY fixed zero-RGB-mean chroma detail, frozen accepted T055."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('prior_detail',Path(__file__).resolve().parents[1]/'T054A_oracle/core.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
torch=prior.torch;o=prior.o;json=o.json;sha=o.sha;write=o.write;utc=o.utc;native=o.native;LABEL=o.LABEL;SPLIT_SHA=o.SPLIT_SHA;thash=prior.thash
PRIOR_FREEZE='64ca0dd47ece431012fac01c9b0a319e24e42e62cff4683c587effe765abd6ae'
PRIOR_PAIRS='c1c1fc5830c50f9ef9ca331747ea0df0a9a3af2d8e7617fff5f7d33f03ad104c'
PRIOR_PREFLIGHT='fbd55327ab55b89b71c7ee8d7871543bc8eb028d5191e67ba9350449d254495e'
OLD=['raw','lift','q','u','b','v','knots','ev','y0','detail','mask','c','c_controls']
def verdict(mean,median,ssim):return 'chroma-detail capacity materially supported' if mean>=.5 and median>=.25 and ssim>=.020 else 'not supported under fixed probe'
def chroma_basis(detail):return detail-detail.mean(dim=1,keepdim=True)
class Chroma(torch.nn.Module):
 def __init__(self,low,raw,gate,box,lift,q,u,b,v):
  super().__init__();base=prior.Detail(low,raw,gate,box,lift,q,u,b)
  with torch.no_grad():base.v.copy_(v);one=base();c=base.coefficient();controls=base.v.tanh()
  for name,value in base.named_buffers():self.register_buffer(name,value.detach().clone())
  self.register_buffer('v',v.detach().clone());self.register_buffer('c',c.detach().clone());self.register_buffer('c_controls',controls.detach().clone());self.register_buffer('one',one.detach().clone());self.register_buffer('d_chroma',chroma_basis(base.detail).detach().clone());self.w_c=torch.nn.Parameter(torch.zeros_like(v))
 def coefficient(self):return torch.nn.functional.interpolate(self.w_c,size=self.y0.shape[-2:],mode='bilinear',align_corners=False).tanh()
 def forward(self):return torch.where(self.mask,(self.one+self.coefficient()*self.d_chroma).clamp(0,1),self.one)
def optimize(model,reference,steps=500):
    assert list(dict(model.named_parameters()))==['w_c'] and torch.count_nonzero(model.w_c)==0
    optimizer=torch.optim.Adam([model.w_c],lr=.05);losses=[];states=[];best=float('inf');best_step=0
    for step in range(steps+1):
        output=model();loss=(output.double()-reference.double()).square().mean();value=float(loss.detach())
        assert torch.isfinite(loss) and torch.isfinite(model.w_c).all()
        losses.append(value);states.append(model.w_c.detach().cpu().clone())
        if value<best:best=value;best_step=step
        if step<steps:optimizer.zero_grad();loss.backward();optimizer.step()
    with torch.no_grad():model.w_c.copy_(states[best_step])
    return dict(w_c=torch.stack(states),mse=torch.tensor(losses,dtype=torch.float64),best_step=best_step,updates=steps)
