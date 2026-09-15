"""REFERENCE_ORACLE_ONLY: frozen T048 plus regional monotonic tone LUT."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('accepted_t048',Path(__file__).resolve().parents[1]/'T048A_oracle/core.py')
t048=importlib.util.module_from_spec(spec);spec.loader.exec_module(t048)
t046=t048.t046
o=t048.o
torch=t048.torch
json=o.json
sha=o.sha
write=o.write
utc=o.utc
native=o.native
LABEL=o.LABEL
SPLIT_SHA=o.SPLIT_SHA
thash=t048.thash
PRIOR_FREEZE='6ff1ad745422d8842a31bb62f17dada5d170bab9b0d9a61d2a21ce23d7f2da2b'
PRIOR_PAIRS='29a33623bbcfc8d9500254344173b68556a6187ce15b1115a435b53fb2a18b48'
PRIOR_PREFLIGHT='f9d72c6f6fa8542bd64186546f1318dfe9a854e7fff686aa5e1e31388687e1a1'
def verdict(mean,median):
    return 'SOTA-scale monotonic-tone capacity supported' if mean>=2 and median>=1 else 'SOTA-scale monotonic-tone capacity not supported under fixed probe'
def knots(q):
    d=torch.nn.functional.softplus(q);d=d/d.sum(dim=-1,keepdim=True)
    return torch.cat([torch.zeros_like(d[...,:1]),d.cumsum(-1)[...,:7],torch.ones_like(d[...,:1])],-1)
class Tone(torch.nn.Module):
    def __init__(self,low,raw,gate,box,lift):
        super().__init__();affine=t048.Affine(low,raw,gate,box,lift)
        with torch.no_grad():
            gain=t046.physical(raw)[:,2:5][:,:,affine.yy[:,None],affine.xx[None,:]]
            base=(.5+(affine.pre_gain*gain+lift[affine.yy[:,None],affine.xx[None,:]]-.5)).clamp(0,1)
            scaled=base*8;seg=scaled.floor().long().clamp(max=7)
            region=(affine.yy[:,None]*2+affine.xx[None,:]);index=region*9+seg
        self.register_buffer('raw',raw.detach().clone());self.register_buffer('lift',lift.detach().clone());self.register_buffer('low',low.detach().clone());self.register_buffer('base',base)
        self.register_buffer('active',affine.active.detach().clone());self.register_buffer('mask',affine.active[affine.yy[:,None],affine.xx[None,:]])
        self.register_buffer('index',index);self.register_buffer('fraction',scaled-seg)
        self.q=torch.nn.Parameter(low.new_zeros(4,8))
    def forward(self):
        y=knots(self.q).flatten();lo=y[self.index];hi=y[self.index+1]
        corrected=lo+(hi-lo)*self.fraction
        return torch.where(self.mask,corrected,self.low)
def optimize(model,reference,steps=500):
    assert torch.count_nonzero(model.q)==0
    optimizer=torch.optim.Adam([model.q],lr=.03);losses=[];states=[];best=float('inf');best_step=0
    for step in range(steps+1):
        output=model();loss=(output.double()-reference.double()).square().mean();v=float(loss.detach());y=knots(model.q)
        assert torch.isfinite(loss) and torch.isfinite(model.q).all() and (y.diff(dim=-1)>=0).all() and torch.equal(y[:,0],torch.zeros_like(y[:,0])) and torch.equal(y[:,-1],torch.ones_like(y[:,-1]))
        losses.append(v);states.append(model.q.detach().cpu().clone())
        if v<best:best=v;best_step=step
        if step<steps:optimizer.zero_grad();loss.backward();optimizer.step()
    with torch.no_grad():model.q.copy_(states[best_step])
    return dict(q=torch.stack(states),mse=torch.tensor(losses,dtype=torch.float64),best_step=best_step,updates=steps)
