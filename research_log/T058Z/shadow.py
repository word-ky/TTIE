"""Verifier only: exact accepted features expression without terminal .float()."""
import torch
from ttie.energy_model import features as accepted_features

def features64(objective,scores,grid):
    active=objective.active.to(scores);signed=active*torch.where(objective.winner==0,1.,-1.)
    z=(scores.double()-scores.new_tensor(objective.calibration['tau'],dtype=torch.float64))/scores.new_tensor(objective.calibration['scale'],dtype=torch.float64)
    state=grid.expand(1,2,2,2)
    parts=(active,signed,objective.evidence,z[:,0],z[:,1],state[0,0].flatten(),state[0,1].flatten())
    assert all(t.dtype==torch.float64 and t.device.type=='cpu' for t in parts)
    phi=torch.cat(parts);assert phi.dtype==torch.float64 and phi.shape==(28,)
    return phi,{n:str(t.dtype) for n,t in zip(['active','signed','evidence','z_dark','z_bright','grid_ev','grid_gamma'],parts)}

def shadow_energy(model,obj,head,v=None,verify=False):
    y=model() if v is None else model.render(v);scores=obj.scorer(y)
    phi,dtypes=features64(obj,scores,model.grid);proof=None
    if verify:
        accepted=accepted_features(obj,scores,model.grid)
        assert torch.equal(accepted,phi.float()),'CAST_ISOLATION_IDENTITY_FAILURE'
        proof=dict(accepted_shadow_features=accepted.detach().tolist(),phi64=phi.detach().tolist(),bitwise_castback_identity=True,intermediate_dtypes=dtypes,scores_dtype=str(scores.dtype),phi_dtype=str(phi.dtype),device=str(phi.device))
    return head(phi).squeeze(),y,phi,proof
