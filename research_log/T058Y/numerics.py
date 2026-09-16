import torch

STEPS=(.004,.002,.001,.0005)

def boundary(model,direction):
    tangent=torch.nn.functional.interpolate(direction,size=model.y0.shape[-2:],mode='bilinear',align_corners=False)*model.detail
    active=model.mask.expand_as(model.y0)
    count=int((((model.y0==0)|(model.y0==1))&(tangent!=0)&active).sum())
    distances=torch.minimum(model.y0.abs(),(model.y0-1).abs())[active]
    nonzero=distances[distances>0]
    return dict(boundary_directional_count=count,min_nonzero_boundary_distance=float(nonzero.min()) if len(nonzero) else None)

def margin(error,tolerance):return dict(error=error,tolerance=tolerance,margin=tolerance-error,passed=error<=tolerance)

def base_checks(e32,d32,e64,d64):
    return dict(primal=margin(abs(e64-e32),2e-4*max(1,abs(e64))),reverse=margin(abs(d32-d64),2e-5+5e-3*abs(d64)))

def fd_check(derivative,count,ladder):
    tol=2e-6+2e-3*abs(derivative)
    if count==0:
        checks=[margin(abs(r['central']-derivative),tol) for r in ladder if r['h'] in [.001,.0005]]
        return dict(classification='smooth-FD confirmed' if len(checks)==2 and all(c['passed'] for c in checks) else 'unresolved',checks=checks)
    r=next(r for r in ladder if r['h']==.0005);lo=min(r['plus_secant'],r['minus_secant']);hi=max(r['plus_secant'],r['minus_secant'])
    return dict(classification='clamp-convention explained' if lo-tol<=derivative<=hi+tol else 'unresolved',interval=[lo,hi],expanded_interval=[lo-tol,hi+tol])
