"""REFERENCE_ORACLE_ONLY: fixed continuation at accepted T049 q."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('accepted_t049',Path(__file__).resolve().parents[1]/'T049A_oracle/core.py')
t049=importlib.util.module_from_spec(spec);spec.loader.exec_module(t049)
t048=t049.t048
t046=t049.t046
o=t049.o
torch=t049.torch
json=o.json
sha=o.sha
write=o.write
utc=o.utc
native=o.native
LABEL=o.LABEL
SPLIT_SHA=o.SPLIT_SHA
thash=t049.thash
Tone=t049.Tone
knots=t049.knots
PRIOR_FREEZE='f7ff8fcb0f405a00b0b07475dcb7f4c87301d2e9c44fade07f6192a4b81a1880'
PRIOR_PAIRS='3c92f22701a69591f78737b3d90a6cfb84fe3a62efe6911ad5db29189752ffd1'
PRIOR_PREFLIGHT='391ec16d991f156f52fdb92de796b8737aaaad6a07ac0c6239a66327e272bba1'
def verdict(mean,median):
    return 'material monotonic-tone underconvergence supported' if mean>=1 and median>=.5 else 'material monotonic-tone underconvergence not supported under fixed extension'
def optimize(model,reference,steps=1000):
    optimizer=torch.optim.Adam([model.q],lr=.03);losses=[];states=[];best=float('inf');best_step=0
    for step in range(steps+1):
        output=model();loss=(output.double()-reference.double()).square().mean();v=float(loss.detach());y=knots(model.q)
        assert torch.isfinite(loss) and torch.isfinite(model.q).all() and (y.diff(dim=-1)>=0).all() and torch.equal(y[:,0],torch.zeros_like(y[:,0])) and torch.equal(y[:,-1],torch.ones_like(y[:,-1]))
        losses.append(v);states.append(model.q.detach().cpu().clone())
        if v<best:best=v;best_step=step
        if step<steps:optimizer.zero_grad();loss.backward();optimizer.step()
    with torch.no_grad():model.q.copy_(states[best_step])
    return dict(q=torch.stack(states),mse=torch.tensor(losses,dtype=torch.float64),best_step=best_step,updates=steps)
