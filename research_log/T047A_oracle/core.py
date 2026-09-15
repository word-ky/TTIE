"""Isolated REFERENCE_ORACLE_ONLY marginal additive lift; legacy state fixed."""
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
PRIOR_FREEZE='2cdf5b8f104dc11445a44e6a68ebc8209192288e08e10bb60859cb95d5693057'
PRIOR_PAIRS='93e6b6dc1d463d1b088cb30fed5b995797c9f1d7dcb72d2544ac2751b7ef2652'
PRIOR_PREFLIGHT='c7a75601cc2949fb1ef3d73d19a456cc34ae448a6965f299ffaa63fe5451c69a'
def verdict(mean,median):
    return 'additive-lift marginal capacity supported' if mean>=.5 and median>=.25 else 'additive-lift material marginal capacity not supported under fixed probe'

class Lift(torch.nn.Module):
    def __init__(self,low,raw,gate,box):
        super().__init__()
        model,_=t046.frozen_model(dict(gate=gate,diagnostics=dict(action_box=box)),low)
        with torch.no_grad():
            model.raw.copy_(raw);field=model.parameter_field(low.shape[-2:])
            v=low*torch.exp2(field[:,:1]);v=(v+1e-6).pow(field[:,1:2])-torch.pow(1e-6,field[:,1:2]);v=v*field[:,2:5]
        # Cache only immutable common-gain preclamp values. Keep the accepted
        # identity-contrast arithmetic, so lift=0 reproduces bit-exactly.
        self.register_buffer('common',v.detach());self.register_buffer('low',low.detach().clone())
        self.register_buffer('legacy_raw',raw.detach().clone());self.register_buffer('active',torch.tensor(gate['active'],device=low.device,dtype=torch.bool).reshape(2,2))
        h,w=low.shape[-2:];self.register_buffer('yy',(torch.arange(h,device=low.device)>=h//2).long());self.register_buffer('xx',(torch.arange(w,device=low.device)>=w//2).long())
        self.b=torch.nn.Parameter(low.new_zeros(2,2))
    def forward(self):
        lift=self.b[self.yy[:,None],self.xx[None,:]]
        corrected=.5+(self.common+lift-.5)
        return torch.where(self.active[self.yy[:,None],self.xx[None,:]],corrected.clamp(0,1),self.low)
    def project(self):
        with torch.no_grad():self.b.clamp_(-.2,.2);self.b.masked_fill_(~self.active,0.)

def optimize(model,reference,steps=500):
    assert torch.count_nonzero(model.b)==0
    optimizer=torch.optim.Adam([model.b],lr=.01);history=[];states=[];best=float('inf');best_step=0
    for step in range(steps+1):
        output=model();loss=(output.double()-reference.double()).square().mean();v=float(loss.detach())
        assert torch.isfinite(loss) and torch.isfinite(model.b).all() and (model.b.abs()<=.2).all()
        history.append(v);states.append(model.b.detach().cpu().clone())
        if v<best:best=v;best_step=step
        if step<steps:optimizer.zero_grad();loss.backward();optimizer.step();model.project()
    with torch.no_grad():model.b.copy_(states[best_step])
    return dict(lift=torch.stack(states),mse=torch.tensor(history,dtype=torch.float64),best_step=best_step,updates=steps)
