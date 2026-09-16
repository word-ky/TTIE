"""REFERENCE_ORACLE_ONLY: fixed additive-range closure; only b changes."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('accepted_t052',Path(__file__).resolve().parents[1]/'T052A_oracle/core.py')
t052=importlib.util.module_from_spec(spec);spec.loader.exec_module(t052)
torch=t052.torch;o=t052.o;json=o.json;sha=o.sha;write=o.write;utc=o.utc;native=o.native;LABEL=o.LABEL;SPLIT_SHA=o.SPLIT_SHA;thash=t052.thash
PRIOR_FREEZE='83ebd03b57e4c5e43191194bc66d507f56090ed593d9681d140ffa29b4e3baa8'
PRIOR_PAIRS='b0c571e305f28e30b4acbe9b1dc0e7d3b06b5fc577f266fcf5d42641368555ff'
PRIOR_PREFLIGHT='ceefa9bad231075bffdd910a9c5e7724cc89f885b2b1f6f807ea01e87988e120'
def verdict(mean,median,ssim):
    return 'additive-range bottleneck supported' if mean>=.5 and median>=.25 and ssim>=0 else 'not supported under fixed range-closure probe'
class Spatial(t052.Spatial):
    def __init__(self,low,raw,gate,box,lift,q,u,b):
        super().__init__(low,raw,gate,box,lift,q,u)
        frozen=self.u.detach().clone();del self.u;self.register_buffer('u',frozen)
        with torch.no_grad():self.b.copy_(b)
def optimize(model,reference,steps=500):
    assert list(dict(model.named_parameters()))==['b'] and model.b.min()>=-.2 and model.b.max()<=.2
    optimizer=torch.optim.Adam([model.b],lr=.01);losses=[];states=[];best=float('inf');best_step=0
    for step in range(steps+1):
        output=model();loss=(output.double()-reference.double()).square().mean();v=float(loss.detach())
        assert torch.isfinite(loss) and torch.isfinite(model.b).all()
        losses.append(v);states.append(model.b.detach().cpu().clone())
        if v<best:best=v;best_step=step
        if step<steps:
            optimizer.zero_grad();loss.backward();optimizer.step()
            with torch.no_grad():model.b.clamp_(-.4,.4)
    with torch.no_grad():model.b.copy_(states[best_step])
    return dict(b=torch.stack(states),mse=torch.tensor(losses,dtype=torch.float64),best_step=best_step,updates=steps)
