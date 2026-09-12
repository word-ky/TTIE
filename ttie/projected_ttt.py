"""T011: label-free gate-consistent action geometry, original semantic objective."""
import math
import torch
from .semantic_ttt import FixedObjective,run_method

METHODS=('identity','region2_direct','region2_discrete_projected','global_ttt_envelope',
         'region2_ttt_envelope','global_ttt_projected','bilinear2_ttt_projected',
         'region2_ttt_projected_1step','region2_ttt_projected')


class ActionBox:
    def __init__(self,objective,size):
        self.active=objective.active.detach().clone()
        self.winner=objective.winner.detach().clone()
        lo=torch.zeros(1,2,size,size,device=self.active.device)
        hi=torch.zeros_like(lo);lo[:,1]=1.;hi[:,1]=1.
        if size==1:
            if self.active.any():
                winners=self.winner[self.active]
                lo[:,0]=0. if (winners==0).all() else -.5
                hi[:,0]=0. if (winners==1).all() else .5
                lo[:,1]=.8;hi[:,1]=1.25
        else:
            active=self.active.reshape(2,2);dark=self.winner.reshape(2,2)==0
            lo[0,0]=torch.where(active & ~dark,-.5,0.)
            hi[0,0]=torch.where(active & dark,.5,0.)
            lo[0,1]=torch.where(active,.8,1.);hi[0,1]=torch.where(active,1.25,1.)
        self.lower=lo;self.upper=hi

    @staticmethod
    def raw(grid):
        return torch.cat((torch.atanh(grid[:,:1]/2),torch.atanh(grid[:,1:].log()/math.log(2))),dim=1)

    def __call__(self,model):
        # Monotone coordinate map: raw clipping is physical box projection.
        # Keep Adam moments, and use exact raw zero for inactive identity nodes.
        with torch.no_grad():
            model.raw.clamp_(min=self.raw(self.lower.to(model.raw)),max=self.raw(self.upper.to(model.raw)))

    def candidates(self,y,x,channel):
        i=2*y+x
        if not self.active[i]:return (0.,) if channel==0 else (1.,)
        if channel==1:return (.8,1.,1.25)
        return (0.,.25,.5) if self.winner[i]==0 else (-.5,-.25,0.)


def run_candidate(image,objective,method,*,max_steps=40,record_states=True):
    region=method.startswith('region2');spatial=region or method.startswith('bilinear2')
    projected='projected' in method
    box=ActionBox(objective,2 if spatial else 1) if projected else None
    policy='discrete' if 'discrete' in method else 'direct' if 'direct' in method else 'ttt'
    core='identity' if method=='identity' else ('spatial2_' if spatial else 'global_')+policy
    one_step=method.endswith('_1step')
    result=run_method(image,objective,core,max_steps=1 if one_step else max_steps,record_states=record_states,
        renderer='region2' if region else 'bilinear2',project=box,
        search_candidates=box.candidates if policy=='discrete' and box else None,exact_updates=one_step)
    if box is not None:
        result['diagnostics']['action_box']=dict(lower=box.lower.cpu().tolist(),upper=box.upper.cpu().tolist())
    if one_step and result['diagnostics']['steps']==1:result['diagnostics']['stop_reason']='exact_one_update'
    return result


def run_all(image,scorer,receipt,*,max_steps=40,record_states=True):
    objective=FixedObjective(scorer,image,receipt)
    results={name:run_candidate(image,objective,name,max_steps=max_steps,record_states=record_states) for name in METHODS}
    gate=dict(scores=objective.original_scores.cpu().tolist(),winner=objective.winner.cpu().tolist(),
              active=objective.active.cpu().tolist(),evidence=objective.evidence.cpu().tolist())
    return results,gate
