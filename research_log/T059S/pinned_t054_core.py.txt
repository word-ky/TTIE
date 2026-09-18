"""REFERENCE_ORACLE_ONLY: frozen T052 output plus one local-detail basis."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('accepted_t052',Path(__file__).resolve().parents[1]/'T052A_oracle/core.py')
t052=importlib.util.module_from_spec(spec);spec.loader.exec_module(t052)
torch=t052.torch;o=t052.o;json=o.json;sha=o.sha;write=o.write;utc=o.utc;native=o.native;LABEL=o.LABEL;SPLIT_SHA=o.SPLIT_SHA;thash=t052.thash
PRIOR_FREEZE='83ebd03b57e4c5e43191194bc66d507f56090ed593d9681d140ffa29b4e3baa8'
PRIOR_PAIRS='b0c571e305f28e30b4acbe9b1dc0e7d3b06b5fc577f266fcf5d42641368555ff'
PRIOR_PREFLIGHT='ceefa9bad231075bffdd910a9c5e7724cc89f885b2b1f6f807ea01e87988e120'
def verdict(mean,median,ssim):
    return 'local-detail capacity supported' if mean>=.5 and median>=.25 and ssim>=.020 else 'not supported under fixed local-detail probe'
def blur(y):
    # Ordered separable float32 five-tap sums; no convolution algorithm selection.
    h,w=y.shape[-2:];k=(1/16,4/16,6/16,4/16,1/16)
    padded=torch.nn.functional.pad(y,(2,2,0,0),mode='reflect');horizontal=torch.zeros_like(y)
    for i,weight in enumerate(k):horizontal=horizontal+padded[...,i:i+w]*weight
    padded=torch.nn.functional.pad(horizontal,(0,0,2,2),mode='reflect');result=torch.zeros_like(y)
    for i,weight in enumerate(k):result=result+padded[...,i:i+h,:]*weight
    return result
class Detail(torch.nn.Module):
    def __init__(self,low,raw,gate,box,lift,q,u,b):
        super().__init__();base=t052.Spatial(low,raw,gate,box,lift,q,u)
        with torch.no_grad():base.b.copy_(b);y0=base();ev=base.ev()
        for name in ['low','raw','lift','q','u','b','mask']:self.register_buffer(name,getattr(base,name).detach().clone())
        self.register_buffer('knots',base.y.detach().clone());self.register_buffer('ev',ev.detach().clone());self.register_buffer('y0',y0.detach().clone());self.register_buffer('detail',(y0-blur(y0)).detach().clone())
        self.v=torch.nn.Parameter(low.new_zeros(1,1,8,8))
    def coefficient(self):return torch.nn.functional.interpolate(self.v,size=self.y0.shape[-2:],mode='bilinear',align_corners=False).tanh()
    def forward(self):return torch.where(self.mask,(self.y0+self.coefficient()*self.detail).clamp(0,1),self.y0)
def optimize(model,reference,steps=500):
    assert list(dict(model.named_parameters()))==['v'] and torch.count_nonzero(model.v)==0
    optimizer=torch.optim.Adam([model.v],lr=.05);losses=[];states=[];best=float('inf');best_step=0
    for step in range(steps+1):
        output=model();loss=(output.double()-reference.double()).square().mean();value=float(loss.detach())
        assert torch.isfinite(loss) and torch.isfinite(model.v).all()
        losses.append(value);states.append(model.v.detach().cpu().clone())
        if value<best:best=value;best_step=step
        if step<steps:optimizer.zero_grad();loss.backward();optimizer.step()
    with torch.no_grad():model.v.copy_(states[best_step])
    return dict(v=torch.stack(states),mse=torch.tensor(losses,dtype=torch.float64),best_step=best_step,updates=steps)
