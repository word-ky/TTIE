"""REFERENCE_ORACLE_ONLY: unchanged accepted T054 renderer, continuation only."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('accepted_t054',Path(__file__).resolve().parents[1]/'T054A_oracle/core.py')
t054=importlib.util.module_from_spec(spec);spec.loader.exec_module(t054)
torch=t054.torch;o=t054.o;json=o.json;sha=o.sha;write=o.write;utc=o.utc;native=o.native;LABEL=o.LABEL;SPLIT_SHA=o.SPLIT_SHA;thash=t054.thash;Detail=t054.Detail;blur=t054.blur
PRIOR_FREEZE='d9130a48a8959b0ef77715bfb1f1fd22d61b19000b9994f62dab96fa9ec83c55'
PRIOR_PAIRS='34c7752f8d415c982728ec58a85906c5186b3367a516f1d76957a8c239a96d5e'
PRIOR_PREFLIGHT='8a41fc421a313cba7363d4267fd878122321e8abac17045e8b81fd077dd356a4'
def verdict(mean,median,ssim):
    return 'material local-detail underconvergence supported' if mean>=.25 and median>=.10 and ssim>=.010 else 'material local-detail underconvergence not supported under fixed extension'
def optimize(model,reference,steps=1000):
    assert list(dict(model.named_parameters()))==['v']
    optimizer=torch.optim.Adam([model.v],lr=.05);losses=[];states=[];best=float('inf');best_step=0
    for step in range(steps+1):
        output=model();loss=(output.double()-reference.double()).square().mean();value=float(loss.detach())
        assert torch.isfinite(loss) and torch.isfinite(model.v).all()
        losses.append(value);states.append(model.v.detach().cpu().clone())
        if value<best:best=value;best_step=step
        if step<steps:optimizer.zero_grad();loss.backward();optimizer.step()
    with torch.no_grad():model.v.copy_(states[best_step])
    return dict(v=torch.stack(states),mse=torch.tensor(losses,dtype=torch.float64),best_step=best_step,updates=steps)
