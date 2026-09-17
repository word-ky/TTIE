"""Verifier-only exact first-order final-clamp decomposition."""
import torch
from research_log.T058Z.shadow import features64

EXPLAINED='T058 row3 reverse gradient explained by mixed-boundary clamp backward convention; prior aggregate interval criterion invalid'
UNRESOLVED='T058 derivative verifier unresolved'

def microprobe():
    results=[]
    for _ in range(2):
        x=torch.tensor([0.,1.],dtype=torch.float64,requires_grad=True)
        g,=torch.autograd.grad(x.clamp(0,1).sum(),x)
        results.append(g.tolist())
    return dict(multipliers=results,deterministic=results[0]==results[1],inclusive=all(v==1. for row in results for v in row))

def parts(z0,z_dot,q,active):
    active=active.expand_as(z0)
    interior=active & (z0>0) & (z0<1)
    lower=active & (z0==0);upper=active & (z0==1)
    plus=(lower & (z_dot>0)) | (upper & (z_dot<0))
    minus=(lower & (z_dot<0)) | (upper & (z_dot>0))
    zero=(lower|upper) & (z_dot==0)
    r=q*z_dot
    values={k:float(r[m].sum()) for k,m in dict(I=interior,B_plus=plus,B_minus=minus,zero_tangent=zero).items()}
    counts={k:int(m.sum()) for k,m in dict(interior=interior,lower=lower,upper=upper,B_plus=plus,B_minus=minus,zero_tangent_boundary=zero,outside=active & ((z0<0)|(z0>1)),inactive=~active).items()}
    for name,mask in [('lower',lower),('upper',upper)]:
        for sign,m in [('positive',z_dot>0),('negative',z_dot<0),('zero',z_dot==0)]:counts[name+'_'+sign]=int((mask&m).sum())
    I,Bp,Bm=values['I'],values['B_plus'],values['B_minus']
    values.update(d_rev_pred=I+Bp+Bm,d_plus_pred=I+Bp,d_minus_pred=I+Bm)
    t=torch.zeros((),dtype=torch.float64,requires_grad=True)
    surrogate=(q*(z0+t*z_dot).clamp(0,1)).sum()
    ds,=torch.autograd.grad(surrogate,t)
    return dict(counts=counts,**values,surrogate_derivative=float(ds))

def close_check(actual,expected,tol):
    error=abs(actual-expected)
    return dict(actual=actual,expected=expected,abs_error=error,tolerance=tol,margin=tol-error,passed=error<=tol)

def decompose(model,obj,head,direction,unchanged_y,d64,historical_d64):
    interp=torch.nn.functional.interpolate(model.v,size=model.y0.shape[-2:],mode='bilinear',align_corners=False)
    z0=(model.y0+interp.tanh()*model.detail).detach()
    reconstructed=torch.where(model.mask,z0.clamp(0,1),model.y0)
    identity=torch.equal(reconstructed,unchanged_y.detach())
    probe=microprobe()
    criteria={'image_identity':dict(passed=identity), 'clamp_microprobe':dict(passed=probe['inclusive'] and probe['deterministic']), 'reverse_reproduction':close_check(d64,historical_d64,1e-10+1e-8*abs(d64))}
    result=dict(image_identity=identity,clamp_microprobe=probe,criteria=criteria)
    if not all(c['passed'] for c in criteria.values()):return result
    z_dot=torch.where(model.mask,model.detail*torch.nn.functional.interpolate(direction,size=model.y0.shape[-2:],mode='bilinear',align_corners=False),torch.zeros_like(model.y0)).detach()
    y=unchanged_y.detach().clone().requires_grad_(True)
    phi,dtypes=features64(obj,obj.scorer(y),model.grid)
    value=head(phi).squeeze();q,=torch.autograd.grad(value,y)
    assert q.dtype==torch.float64 and q.device.type=='cpu' and torch.isfinite(q).all()
    assert torch.isfinite(z_dot).all() and torch.count_nonzero(z_dot[~model.mask.expand_as(z_dot)])==0
    decomposition=parts(z0,z_dot,q.detach(),model.mask)
    result.update(decomposition,leaf_energy=float(value.detach()),q_dtype=str(q.dtype),feature_dtypes=dtypes,z_dtype=str(z0.dtype),tangent_dtype=str(z_dot.dtype),inactive_tangent_nonzero=0)
    tol=2e-8+2e-5*abs(d64)
    criteria['reverse_decomposition']=close_check(d64,result['d_rev_pred'],tol)
    criteria['clamp_surrogate']=close_check(result['surrogate_derivative'],result['d_rev_pred'],tol)
    criteria['mixed_boundaries']=dict(B_plus_nonzero=result['B_plus']!=0.,B_minus_nonzero=result['B_minus']!=0.,zero_tangent_contribution=result['zero_tangent'],passed=result['B_plus']!=0. and result['B_minus']!=0.)
    return result
