"""REFERENCE_ORACLE_ONLY: frozen regional affine/tone plus joint smooth exposure and additive fields."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('accepted_t051',Path(__file__).resolve().parents[1]/'T051A_oracle/core.py')
t051=importlib.util.module_from_spec(spec);spec.loader.exec_module(t051)
t050=t051.t050
torch=t051.torch;o=t051.o;json=o.json;sha=o.sha;write=o.write;utc=o.utc;native=o.native;LABEL=o.LABEL;SPLIT_SHA=o.SPLIT_SHA;thash=t050.thash
PRIOR_FREEZE='d66edca57c6cac1ffe15a428552c73e2e343363165a2808f16a1b01cce460667'
PRIOR_PAIRS='ecc607ac2c04a21d5ad53c06264f22869e0ee3f65810c22d952b5bbec665be2f'
PRIOR_PREFLIGHT='9a9c4e9964a5eaf99924d454fc6206964a8586f89c84a6da99a4163d39f00fca'
def verdict(mean,median,ssim):
    return 'spatial affine coupling capacity supported' if mean>=1.5 and median>=.75 and ssim>=0 else 'not supported under fixed probe'
class Spatial(t051.Spatial):
    def __init__(self,low,raw,gate,box,lift,q,u):
        super().__init__(low,raw,gate,box,lift,q)
        with torch.no_grad():self.u.copy_(u)
        self.b=torch.nn.Parameter(low.new_zeros(1,1,8,8))
    def offset(self):return torch.nn.functional.interpolate(self.b,size=self.low.shape[-2:],mode='bilinear',align_corners=False)
    def forward(self):
        z=(self.base*torch.exp2(self.ev())+self.offset()).clamp(0,1);scaled=z*8;segment=scaled.floor().long().clamp(max=7)
        index=self.region*9+segment;y=self.y.flatten();lo=y[index];hi=y[index+1]
        return torch.where(self.mask,lo+(hi-lo)*(scaled-segment),self.low)
def optimize(model,reference,steps=500):
    assert torch.count_nonzero(model.b)==0 and list(dict(model.named_parameters()))==['u','b']
    optimizer=torch.optim.Adam([dict(params=[model.u],lr=.05),dict(params=[model.b],lr=.01)])
    losses=[];us=[];bs=[];best=float('inf');best_step=0
    for step in range(steps+1):
        output=model();loss=(output.double()-reference.double()).square().mean();v=float(loss.detach())
        assert torch.isfinite(loss) and torch.isfinite(model.u).all() and torch.isfinite(model.b).all()
        losses.append(v);us.append(model.u.detach().cpu().clone());bs.append(model.b.detach().cpu().clone())
        if v<best:best=v;best_step=step
        if step<steps:
            optimizer.zero_grad();loss.backward();optimizer.step()
            with torch.no_grad():model.b.clamp_(-.2,.2)
    with torch.no_grad():model.u.copy_(us[best_step]);model.b.copy_(bs[best_step])
    return dict(u=torch.stack(us),b=torch.stack(bs),mse=torch.tensor(losses,dtype=torch.float64),best_step=best_step,updates=steps)
