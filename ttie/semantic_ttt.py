"""T008 target-free EV/gamma controls and episodic semantic optimization."""
import math
import torch
from torch import nn

from .isp import ISP, physical_parameters
from .joint_gate import both_gates

METHODS=('identity','global_direct','spatial2_direct','global_discrete','spatial2_discrete','global_ttt','spatial2_ttt')
EV_CANDIDATES=(-1.,-.5,0.,.5,1.)
GAMMA_CANDIDATES=(.8,1.,1.25)


class EVGamma(ISP):
    def __init__(self, size=1):
        nn.Module.__init__(self)
        self.raw=nn.Parameter(torch.zeros(1,2,size,size))

    def physical_grid(self):
        other=self.raw.new_zeros(1,4,*self.raw.shape[-2:])
        return physical_parameters(torch.cat((self.raw,other),dim=1))

    def set_grid(self, values):
        raw=torch.cat((torch.atanh(values[:,:1]/2),torch.atanh(values[:,1:].log()/math.log(2))),dim=1)
        with torch.no_grad():self.raw.copy_(raw)


class SemanticScorer(nn.Module):
    def __init__(self, encoder, prototypes):
        super().__init__()
        self.encoder=encoder.eval().requires_grad_(False)
        self.prototypes=prototypes.eval().requires_grad_(False)

    def forward(self, image):
        # Preserve differentiability to the image; no learned_scores/no_grad here.
        return self.prototypes.scores(self.encoder.image_embeddings(image))[1:]


class FixedObjective:
    def __init__(self, scorer, image, receipt):
        self.scorer=scorer
        self.calibration=receipt['calibration']
        with torch.no_grad():
            self.original_scores=scorer(image).detach()
            self.winner,_,self.active,self.evidence=both_gates(self.original_scores,receipt)
        self.active=self.active.clone()

    def __call__(self, image):
        if not self.active.any():return image.sum()*0
        return self.from_scores(self.scorer(image))

    def from_scores(self, scores):
        if not self.active.any():return scores.sum()*0
        scores=scores.to(torch.float64)
        z=(scores-scores.new_tensor(self.calibration['tau']))/scores.new_tensor(self.calibration['scale'])
        return z[self.active].relu().square().sum(dim=-1).mean()


def grid_ranges(model):
    grid=model.physical_grid().detach()[:,:2]
    return dict(min=grid.amin(dim=(0,2,3)).tolist(),max=grid.amax(dim=(0,2,3)).tolist())


def direct_grid(objective, size):
    action=torch.where(objective.active,torch.where(objective.winner==0,.5,-.5),0.)
    ev=action.mean().reshape(1,1,1,1) if size==1 else action.reshape(1,1,2,2)
    return torch.cat((ev,torch.ones_like(ev)),dim=1)


def choose_candidate(candidates, losses, identity):
    return min(zip(candidates,losses),key=lambda pair:(pair[1],abs(pair[0]-identity),pair[0]))[0]


def run_method(image, objective, method, *, max_steps=40):
    source=image.detach()
    size=2 if method.startswith('spatial2') else 1
    model=EVGamma(size).to(source)
    losses=[];gradients=[];ranges=[];search=[];updates=0
    with torch.no_grad():initial=float(objective.from_scores(objective.original_scores))
    reason='identity' if method=='identity' else 'no_active' if not objective.active.any() else None
    if reason is not None:
        output=source.clone();losses=[initial];ranges=[grid_ranges(model)]
    elif method.endswith('_direct'):
        model.set_grid(direct_grid(objective,size))
        with torch.no_grad():output=model(source);final=float(objective(output))
        losses=[initial,final];ranges=[grid_ranges(model)];reason='fixed_direct'
    elif method.endswith('_discrete'):
        values=model.physical_grid().detach()[:,:2].clone()
        losses=[initial]
        with torch.no_grad():
            for y in range(size):
                for x in range(size):
                    for channel,candidates,identity in ((0,EV_CANDIDATES,0.),(1,GAMMA_CANDIDATES,1.)):
                        scores=[]
                        for candidate in candidates:
                            values[0,channel,y,x]=candidate;model.set_grid(values)
                            scores.append(float(objective(model(source))))
                        chosen=choose_candidate(candidates,scores,identity)
                        values[0,channel,y,x]=chosen;model.set_grid(values)
                        losses.append(scores[candidates.index(chosen)])
                        search.append(dict(node=[y,x],channel=channel,candidates=candidates,losses=scores,chosen=chosen))
                        ranges.append(grid_ranges(model))
            output=model(source)
        reason='one_coordinate_pass'
    else:
        optimizer=torch.optim.Adam([model.raw],lr=.03)
        for step in range(max_steps+1):
            output=model(source)
            loss=objective(output)
            losses.append(float(loss.detach()));ranges.append(grid_ranges(model))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite T008 semantic loss')
            if float(loss.detach())<=1e-8:reason='clean_envelope';break
            if step==max_steps:reason='max_updates';break
            optimizer.zero_grad(set_to_none=True)
            gradient,=torch.autograd.grad(loss,model.raw)
            if not torch.isfinite(gradient).all():raise RuntimeError('Nonfinite T008 ISP gradient')
            gradients.append(float(gradient.norm()));model.raw.grad=gradient
            optimizer.step();updates+=1
    output=output.detach();grid=model.physical_grid().detach()[:,:2]
    assert torch.isfinite(output).all() and torch.isfinite(grid).all()
    return dict(image=output,raw=model.raw.detach().clone(),grid=grid,
                diagnostics=dict(loss_before=initial,loss_after=losses[-1],loss_trajectory=losses,gradient_norms=gradients,
                 parameter_ranges=ranges,steps=updates,stop_reason=reason,search=search,active_count=int(objective.active.sum()),
                 all_finite=True,parameter_count=model.raw.numel(),final_grid=grid.cpu().tolist(),
                 final_ranges=grid_ranges(model),reset='identity and fresh optimizer per episode'))


def run_all(image, scorer, receipt, *, max_steps=40):
    objective=FixedObjective(scorer,image,receipt)
    results={name:run_method(image,objective,name,max_steps=max_steps) for name in METHODS}
    gate=dict(scores=objective.original_scores.cpu().tolist(),winner=objective.winner.cpu().tolist(),
              active=objective.active.cpu().tolist(),evidence=objective.evidence.cpu().tolist())
    return results,gate
