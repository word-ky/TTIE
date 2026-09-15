"""REFERENCE_ORACLE_ONLY: frozen affine/tone plus smooth exposure field."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('accepted_t050',Path(__file__).resolve().parents[1]/'T050A_oracle/core.py')
t050=importlib.util.module_from_spec(spec);spec.loader.exec_module(t050)
torch=t050.torch;o=t050.o;json=o.json;sha=o.sha;write=o.write;utc=o.utc;native=o.native;LABEL=o.LABEL;SPLIT_SHA=o.SPLIT_SHA;thash=t050.thash
PRIOR_FREEZE='b58696cd5874d92a72cebd03132fa0dd8e93fd75bfba638caf36642532b1dd5c'
PRIOR_PAIRS='2f3cf008677427fec0a1be0202e63272b1cd6d275957f295972161ec237716d1'
PRIOR_PREFLIGHT='ebe4679c8e40ff8b6a2496ba0c7c97a1f77a376a9b5bbb7416700ce49f574f17'
def verdict(mean,median,ssim):
    return 'smooth spatial illumination capacity supported' if mean>=1.5 and median>=.75 and ssim>=0 else 'not supported under fixed probe'
class Spatial(torch.nn.Module):
    def __init__(self,low,raw,gate,box,lift,q):
        super().__init__();tone=t050.Tone(low,raw,gate,box,lift)
        for name in ['raw','lift','low','base','mask']:
            self.register_buffer(name,getattr(tone,name).detach().clone())
        self.register_buffer('q',q.detach().clone());self.register_buffer('y',t050.knots(q).detach().clone())
        self.register_buffer('region',tone.index//9)
        self.u=torch.nn.Parameter(low.new_zeros(1,1,8,8))
    def ev(self):return torch.nn.functional.interpolate(2*self.u.tanh(),size=self.low.shape[-2:],mode='bilinear',align_corners=False)
    def forward(self):
        z=(self.base*torch.exp2(self.ev())).clamp(0,1);scaled=z*8;segment=scaled.floor().long().clamp(max=7)
        index=self.region*9+segment;y=self.y.flatten();lo=y[index];hi=y[index+1]
        return torch.where(self.mask,lo+(hi-lo)*(scaled-segment),self.low)
def optimize(model,reference,steps=500):
    assert torch.count_nonzero(model.u)==0 and list(dict(model.named_parameters()))==['u']
    optimizer=torch.optim.Adam([model.u],lr=.05);losses=[];states=[];best=float('inf');best_step=0
    for step in range(steps+1):
        output=model();loss=(output.double()-reference.double()).square().mean();v=float(loss.detach())
        assert torch.isfinite(loss) and torch.isfinite(model.u).all()
        losses.append(v);states.append(model.u.detach().cpu().clone())
        if v<best:best=v;best_step=step
        if step<steps:optimizer.zero_grad();loss.backward();optimizer.step()
    with torch.no_grad():model.u.copy_(states[best_step])
    return dict(u=torch.stack(states),mse=torch.tensor(losses,dtype=torch.float64),best_step=best_step,updates=steps)
