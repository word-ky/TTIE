"""Fixed zero-reference objective on unchanged T036 renderer and CommonBox."""
import torch
import torch.nn.functional as F
from ttie.common_gain import CommonRegion2, CommonBox

def losses(x,y):
    xp=F.avg_pool2d(x,4,4);yp=F.avg_pool2d(y,4,4)
    horizontal=(torch.diff(yp,dim=-1)-torch.diff(xp,dim=-1)).abs().flatten()
    vertical=(torch.diff(yp,dim=-2)-torch.diff(xp,dim=-2)).abs().flatten()
    spa=torch.cat((horizontal,vertical)).mean()
    exp=(F.avg_pool2d(y,16,16).mean(dim=1)-.60).square().mean()
    mu=y.mean(dim=(-2,-1))
    col=((mu[:,0]-mu[:,1]).square()+(mu[:,0]-mu[:,2]).square()+(mu[:,1]-mu[:,2]).square()).mean()
    return torch.stack((spa,exp,col))

def trajectory(low,gate):
    model=CommonRegion2(gate.active).to(low);box=CommonBox(gate,2)
    assert model.raw.numel()==12 and gate.active.any()
    opt=torch.optim.Adam([model.raw],lr=.03)
    images=[];states=[];values=[];components=[];grads=[];pre=[]
    for step in range(41):
        y=model(low);parts=losses(low,y);value=parts @ parts.new_tensor([1.,10.,5.])
        assert torch.isfinite(value) and torch.isfinite(y).all()
        images.append(y.detach().cpu().clone());states.append(model.raw.detach().cpu().clone())
        values.append(float(value.detach()));components.append(parts.detach().cpu())
        if step==40:break
        opt.zero_grad(set_to_none=True);g,=torch.autograd.grad(value,model.raw)
        assert torch.isfinite(g).all();grads.append(g.detach().cpu());model.raw.grad=g
        opt.step();pre.append(model.raw.detach().cpu().clone());box(model)
    selected=min(range(41),key=lambda i:(values[i],i))
    return dict(images=torch.stack(images),states=torch.stack(states),values=values,
        components=torch.stack(components),gradients=torch.stack(grads),pre_box=torch.stack(pre),
        selected_step=selected,active=gate.active.cpu(),winner=gate.winner.cpu(),
        lower=box.lower.cpu(),upper=box.upper.cpu())
