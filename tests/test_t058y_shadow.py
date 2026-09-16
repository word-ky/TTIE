import torch
from research_log.T058Y.numerics import base_checks,fd_check,boundary
from research_log.T058A_tangent.core import Detail

def test_fixed_criteria_and_shadow_cast():
    y=torch.full((1,3,8,8),.5);y[:,:,2,2]=1
    original=Detail(y,torch.ones(4,dtype=torch.bool),torch.zeros(1,2,2,2))
    import copy
    shadow=copy.deepcopy(original).cpu().double();d=torch.ones_like(shadow.v)/8
    assert original.y0.dtype==torch.float32 and shadow.y0.dtype==torch.float64
    assert boundary(shadow,d)['boundary_directional_count']==3
    assert all(c['passed'] for c in base_checks(1.,.1,1.,.1).values())
    ladder=[dict(h=h,central=.1,plus_secant=.11,minus_secant=.09) for h in [.004,.002,.001,.0005]]
    assert fd_check(.1,0,ladder)['classification']=='smooth-FD confirmed'
    assert fd_check(.1,3,ladder)['classification']=='clamp-convention explained'
    assert fd_check(.2,0,ladder)['classification']=='unresolved'
