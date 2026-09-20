"""Exact accepted T062-A loop prefix; output fixed step27."""
import torch
from ttie.common_gain import CommonRegion2,CommonBox
from research_log.T062A.core import losses

def trajectory(low,gate):
    model=CommonRegion2(gate.active).to(low);box=CommonBox(gate,2)
    assert model.raw.numel()==12 and gate.active.any()
    opt=torch.optim.Adam([model.raw],lr=.03)
    images=[];states=[];values=[];components=[];grads=[];pre=[]
    for step in range(28):
        y=model(low);parts=losses(low,y);value=parts @ parts.new_tensor([1.,10.,5.])
        assert torch.isfinite(value) and torch.isfinite(y).all()
        images.append(y.detach().cpu().clone());states.append(model.raw.detach().cpu().clone())
        values.append(float(value.detach()));components.append(parts.detach().cpu())
        if step==27:break
        opt.zero_grad(set_to_none=True);g,=torch.autograd.grad(value,model.raw)
        assert torch.isfinite(g).all();grads.append(g.detach().cpu());model.raw.grad=g
        opt.step();pre.append(model.raw.detach().cpu().clone());box(model)
    selected=27
    return dict(images=torch.stack(images),states=torch.stack(states),values=values,
        components=torch.stack(components),gradients=torch.stack(grads),pre_box=torch.stack(pre),
        selected_step=selected,active=gate.active.cpu(),winner=gate.winner.cpu(),
        lower=box.lower.cpu(),upper=box.upper.cpu())
