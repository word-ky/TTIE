"""Fixed source state bank; no outcome-dependent sampling."""
import torch
from .semantic_ttt import FixedObjective,Region2
from .projected_ttt import ActionBox,run_candidate
from .stop_trajectory import capture,checkpoint
from .energy_model import features

BANK=dict(version='T013-bank-v1',order=['identity','region2_direct','region2_discrete_projected',
    'semantic_1','semantic_4','semantic_8','semantic_16','semantic_40']+[f'sobol_{i:02d}' for i in range(16)],
    sobol=dict(dimension=8,scramble=False,skip=0,count=16,coordinates='EV TL TR BL BR; gamma TL TR BL BR',mapping='lower+u*(upper-lower)'),
    duplicate_states='retain fixed entries',no_active='identity only')


def state_bank(image,scorer,receipt,*,semantic_steps=40):
    objective=FixedObjective(scorer,image,receipt);model=Region2(objective.active).to(image);box=ActionBox(objective,2)
    records=[('identity',image.detach(),model.raw.detach().clone(),model.physical_grid().detach()[:,:2])]
    semantic=None
    if objective.active.any():
        for method in ('region2_direct','region2_discrete_projected'):
            r=run_candidate(image,objective,method);records.append((method,r['image'],r['raw'],r['grid']))
        semantic=capture(image,scorer,receipt,max_steps=semantic_steps)
        for step in (1,4,8,16,40):
            r=checkpoint(semantic,step);records.append((f'semantic_{step}',r['image'].to(image),r['raw'].to(image),r['grid'].to(image)))
        points=torch.quasirandom.SobolEngine(8,scramble=False).draw(16).to(image)
        for i,p in enumerate(points):
            grid=box.lower+p.reshape(1,2,2,2)*(box.upper-box.lower);model.set_grid(grid)
            with torch.no_grad():output=model(image)
            records.append((f'sobol_{i:02d}',output,model.raw.detach().clone(),model.physical_grid().detach()[:,:2]))
    arrays={k:[] for k in ('images','states','grids','scores','features')}
    with torch.no_grad():
        for _,pixels,raw,grid in records:
            scores=scorer(pixels);f=features(objective,scores,grid)
            for k,v in dict(images=pixels,states=raw,grids=grid,scores=scores,features=f).items():arrays[k].append(v.detach().cpu().clone())
    gate=dict(scores=objective.original_scores.cpu().tolist(),active=objective.active.cpu().tolist(),winner=objective.winner.cpu().tolist(),evidence=objective.evidence.cpu().tolist())
    return dict(**{k:torch.stack(v) for k,v in arrays.items()},names=[r[0] for r in records],gate=gate,
                semantic_diagnostics=None if semantic is None else semantic['diagnostics'])
