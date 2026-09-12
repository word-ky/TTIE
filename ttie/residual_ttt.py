"""T010 label-free residual-evidence correction; no evaluation metadata API."""
from dataclasses import dataclass
import torch
from .semantic_ttt import FixedObjective,run_method

RHO_GRID=(.25,.50,.75,.90)
METHODS=('identity','global_direct','region2_direct','region2_discrete_rho',
         'global_ttt_rho','bilinear2_ttt_rho','region2_ttt_rho','region2_ttt_envelope')


@dataclass(frozen=True)
class Rhos:
    dark: float
    bright: float


class ResidualObjective(FixedObjective):
    def __init__(self,scorer,image,receipt,rhos):
        super().__init__(scorer,image,receipt)
        self.rhos=rhos
        z=self.standardize(self.original_scores)
        self.e0=z.gather(1,self.winner[:,None]).squeeze(1).relu().square().detach()
        self.targets=self.e0*self.e0.new_tensor((rhos.dark,rhos.bright))[self.winner]

    def standardize(self,scores):
        scores=scores.to(torch.float64)
        return (scores-scores.new_tensor(self.calibration['tau']))/scores.new_tensor(self.calibration['scale'])

    def from_scores(self,scores):
        if not self.active.any():return scores.sum()*0
        energy=self.standardize(scores).relu().square()
        winner=energy.gather(1,self.winner[:,None]).squeeze(1)
        opposite=energy.gather(1,(1-self.winner)[:,None]).squeeze(1)
        return ((winner-self.targets).relu()+opposite)[self.active].mean()


def run_candidate(image,objective,method,*,max_steps=40,record_states=False):
    region=method.startswith('region2')
    spatial=region or method.startswith('bilinear2')
    policy='discrete' if 'discrete' in method else 'direct' if 'direct' in method else 'ttt'
    core='identity' if method=='identity' else ('spatial2_' if spatial else 'global_')+policy
    result=run_method(image,objective,core,max_steps=max_steps,record_states=record_states,
                      renderer='region2' if region else 'bilinear2')
    if isinstance(objective,ResidualObjective) and objective.rhos!=Rhos(0.,0.):
        if result['diagnostics']['stop_reason']=='clean_envelope':result['diagnostics']['stop_reason']='residual_target'
    return result


def run_all(image,scorer,receipt,rhos,*,max_steps=40,record_states=False):
    objective=ResidualObjective(scorer,image,receipt,rhos)
    envelope=FixedObjective(scorer,image,receipt)
    results={name:run_candidate(image,envelope if name=='region2_ttt_envelope' else objective,name,
               max_steps=max_steps,record_states=record_states) for name in METHODS}
    gate=dict(scores=objective.original_scores.cpu().tolist(),winner=objective.winner.cpu().tolist(),
              active=objective.active.cpu().tolist(),e0=objective.e0.cpu().tolist(),targets=objective.targets.cpu().tolist())
    return results,gate
