"""Isolated REFERENCE_ORACLE_ONLY affine closure; EV/gamma fixed."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('accepted_t046',Path(__file__).resolve().parents[1]/'T046A_oracle/core.py')
t046=importlib.util.module_from_spec(spec);spec.loader.exec_module(t046)
o=t046.o
torch=t046.torch
json=o.json
sha=o.sha
write=o.write
utc=o.utc
native=o.native
LABEL=o.LABEL
SPLIT_SHA=o.SPLIT_SHA
thash=t046.thash
PRIOR_FREEZE='c82e6b609c14a0464e2cab1fdf6a5e11036aeac6e8d800c708e8640646e6a75e'
PRIOR_PAIRS='d4b12f46734c05369efea777b5e699e659d82a16d1e969d4bd93a9cdecd37790'
PRIOR_PREFLIGHT='90edc063e6dc92722c3db46e6afe7b292d41e2f40e767857f09efa4d58820813'
def verdict(mean,median):
    return 'post-gamma affine coupling materially supported' if mean>=.5 and median>=.25 else 'post-gamma affine coupling material increment not supported under fixed probe'

class Affine(torch.nn.Module):
    def __init__(self,low,raw,gate,box,lift):
        super().__init__()
        model,_=t046.frozen_model(dict(gate=gate,diagnostics=dict(action_box=box)),low)
        with torch.no_grad():
            model.raw.copy_(raw);field=model.parameter_field(low.shape[-2:])
            v=low*torch.exp2(field[:,:1]);v=(v+1e-6).pow(field[:,1:2])-torch.pow(1e-6,field[:,1:2])
        self.register_buffer('pre_gain',v.detach());self.register_buffer('low',low.detach().clone())
        self.register_buffer('ev_gamma',raw[:,:2].detach().clone());self.register_buffer('active',torch.tensor(gate['active'],device=low.device,dtype=torch.bool).reshape(2,2))
        h,w=low.shape[-2:];self.register_buffer('yy',(torch.arange(h,device=low.device)>=h//2).long());self.register_buffer('xx',(torch.arange(w,device=low.device)>=w//2).long())
        self.gain_raw=torch.nn.Parameter(raw[:,2:3].detach().clone());self.b=torch.nn.Parameter(lift.detach().clone().to(low))
    def raw(self):return torch.cat([self.ev_gamma,self.gain_raw],1)
    def forward(self):
        gain=t046.physical(self.raw())[:,2:5,:, :][:,:,self.yy[:,None],self.xx[None,:]]
        lift=self.b[self.yy[:,None],self.xx[None,:]]
        corrected=.5+(self.pre_gain*gain+lift-.5)
        return torch.where(self.active[self.yy[:,None],self.xx[None,:]],corrected.clamp(0,1),self.low)
    def project(self):
        # Accepted common-gain tanh map already bounds physical gain [.5,2].
        # The accepted gain projection only forces inactive raw gain to zero.
        with torch.no_grad():
            self.gain_raw.masked_fill_(~self.active.reshape(1,1,2,2),0.)
            self.b.clamp_(-.2,.2);self.b.masked_fill_(~self.active,0.)

def optimizer_for(model):
    return torch.optim.Adam([dict(params=[model.gain_raw],lr=.05),dict(params=[model.b],lr=.01)])
def optimize(model,reference,steps=500):
    optimizer=optimizer_for(model);history=[];lifts=[];gains=[];best=float('inf');best_step=0
    for step in range(steps+1):
        output=model();loss=(output.double()-reference.double()).square().mean();v=float(loss.detach())
        assert torch.isfinite(loss) and torch.isfinite(model.b).all() and torch.isfinite(model.gain_raw).all() and (model.b.abs()<=.2).all()
        history.append(v);lifts.append(model.b.detach().cpu().clone());gains.append(model.gain_raw.detach().cpu().clone())
        if v<best:best=v;best_step=step
        if step<steps:optimizer.zero_grad();loss.backward();optimizer.step();model.project()
    with torch.no_grad():model.b.copy_(lifts[best_step]);model.gain_raw.copy_(gains[best_step])
    return dict(lift=torch.stack(lifts),gain_raw=torch.stack(gains),mse=torch.tensor(history,dtype=torch.float64),best_step=best_step,updates=steps)
