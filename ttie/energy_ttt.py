"""Projected state-aware learned-energy trajectory; no reference inputs."""
import torch
from .semantic_ttt import FixedObjective,Region2,EVGamma,grid_ranges
from .projected_ttt import ActionBox
from .stop_trajectory import checkpoint
from .energy_model import features

METHODS=('global_ttt_energy','bilinear2_ttt_energy','region2_ttt_energy')


def make_model(objective,image,basis):
    if basis=='region2':return Region2(objective.active).to(image)
    return EVGamma(1 if basis=='global' else 2).to(image)


def evaluate_energy(model,image,objective,head):
    output=model(image);scores=objective.scorer(output);grid=model.physical_grid()[:,:2]
    f=features(objective,scores,grid)
    return head(f).squeeze(),output,scores,grid,f


def trajectory(image,scorer,receipt,head,*,basis='region2',max_steps=40):
    source=image.detach();objective=FixedObjective(scorer,source,receipt)
    model=make_model(objective,source,basis);box=ActionBox(objective,model.raw.shape[-1])
    head=head.to(source).eval().requires_grad_(False)
    active=bool(objective.active.any());budget=max_steps if active else 0
    optimizer=torch.optim.Adam([model.raw],lr=.03)
    arrays={k:[] for k in ('images','scores','grids','states','features')}
    energies=[];gradients=[];projections=[];ranges=[]
    for step in range(budget+1):
        if active:
            value,output,scores,grid,f=evaluate_energy(model,source,objective,head)
        else:
            output=source.clone();scores=objective.original_scores;grid=model.physical_grid()[:,:2]
            f=features(objective,scores,grid)
            with torch.no_grad():value=head(f).squeeze()
        assert torch.isfinite(value)
        for k,v in dict(images=output,scores=scores,grids=grid,states=model.raw,features=f).items():arrays[k].append(v.detach().cpu().clone())
        energies.append(float(value.detach()));ranges.append(grid_ranges(model))
        if step==budget:break
        optimizer.zero_grad(set_to_none=True);gradient,=torch.autograd.grad(value,model.raw)
        assert torch.isfinite(gradient).all()
        gradients.append(gradient.detach().cpu().tolist());model.raw.grad=gradient
        optimizer.step()
        p=dict(pre_raw=model.raw.detach().cpu().tolist(),pre_grid=model.physical_grid().detach()[:,:2].cpu().tolist())
        box(model);p.update(post_raw=model.raw.detach().cpu().tolist(),post_grid=model.physical_grid().detach()[:,:2].cpu().tolist());projections.append(p)
    gate=dict(scores=objective.original_scores.cpu().tolist(),active=objective.active.cpu().tolist(),winner=objective.winner.cpu().tolist(),evidence=objective.evidence.cpu().tolist())
    diagnostics=dict(loss_before=energies[0],loss_after=energies[-1],loss_trajectory=energies,
        gradient_norms=[float(torch.tensor(g).norm()) for g in gradients],gradient_vectors=gradients,
        parameter_ranges=ranges,steps=budget,stop_reason='exact_update_budget' if active else 'no_active',
        search=[],active_count=int(objective.active.sum()),projections=projections,all_finite=True,
        parameter_count=model.raw.numel(),final_grid=arrays['grids'][-1].tolist(),final_ranges=ranges[-1],
        reset='identity and fresh optimizer per episode',objective='frozen T013 predicted log MSE',
        action_box=dict(lower=box.lower.cpu().tolist(),upper=box.upper.cpu().tolist()))
    t=dict(**{k:torch.stack(v) for k,v in arrays.items()},gate=gate,diagnostics=diagnostics)
    index=min(range(len(energies)),key=lambda i:(energies[i],i)) if active else 0
    decision=dict(selected_step=index,scores=energies,bypass=None if active else 'no_active')
    return checkpoint(t,index),t,decision
