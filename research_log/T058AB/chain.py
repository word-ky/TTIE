"""Verifier-only complete detail-gradient chain rule with frozen downstream q."""
import torch
from research_log.T058Z.shadow import features64
from research_log.T058AA.decomposition import microprobe
from research_log.T058Y.numerics import boundary
from research_log.T058A_tangent.core import thash

CREDIBLE='T058 first16 reverse detail gradients numerically credible under exact clamp-aware chain rule'
UNRESOLVED='T058 derivative verifier unresolved'

def vector_checks(g32,g64,g_chain):
    a=g32.double();b=g64;delta=a-b
    n32=float(torch.linalg.vector_norm(a));n64=float(torch.linalg.vector_norm(b))
    error=float(torch.linalg.vector_norm(delta));limit=1e-5+1e-3*n64
    chain_error=float((b-g_chain).abs().max());chain_limit=2e-8+2e-5*float(b.abs().max())
    if max(n32,n64)>1e-8:
        cosine=float((a*b).sum()/(n32*n64)) if n32*n64>0 else None
        orientation=dict(mode='cosine',cosine=cosine,threshold=.9999,margin=None if cosine is None else cosine-.9999,passed=cosine is not None and cosine>=.9999)
    else:orientation=dict(mode='both_negligible',cosine=None,l2_error=error,threshold=1e-8,margin=1e-8-error,passed=error<=1e-8)
    return dict(norm32=n32,norm64=n64,norm_chain=float(torch.linalg.vector_norm(g_chain)),criteria=dict(chain=dict(max_abs_error=chain_error,tolerance=chain_limit,margin=chain_limit-chain_error,passed=chain_error<=chain_limit),float32_float64=dict(l2_error=error,tolerance=limit,margin=limit-error,passed=error<=limit),orientation=orientation))

def full_chain(model,obj,head,direction,y64,E32,E64,g32,g64):
    v=torch.zeros_like(model.v,requires_grad=True)
    c=torch.nn.functional.interpolate(v,size=model.y0.shape[-2:],mode='bilinear',align_corners=False).tanh()
    z=model.y0+c*model.detail
    ys=torch.where(model.mask,z.clamp(0,1),model.y0)
    identity=torch.equal(ys.detach(),y64.detach());probe=microprobe()
    y=y64.detach().clone().requires_grad_(True)
    phi,dtypes=features64(obj,obj.scorer(y),model.grid)
    leaf_energy=head(phi).squeeze();q,=torch.autograd.grad(leaf_energy,y)
    q=q.detach()
    inactive_gradient,=torch.autograd.grad((q*ys*(~model.mask)).sum(),v,retain_graph=True)
    g_chain,=torch.autograd.grad((q*ys).sum(),v)
    z0=z.detach();active=model.mask.expand_as(z0)
    tangent=torch.where(active,model.detail*torch.nn.functional.interpolate(direction,size=model.y0.shape[-2:],mode='bilinear',align_corners=False),torch.zeros_like(z0))
    counts=dict(interior=int((active&(z0>0)&(z0<1)).sum()),lower=int((active&(z0==0)).sum()),upper=int((active&(z0==1)).sum()),zero_tangent_boundary=int((active&((z0==0)|(z0==1))&(tangent==0)).sum()),inactive=int((~active).sum()))
    for name,mask in [('lower',active&(z0==0)),('upper',active&(z0==1))]:
        for sign,m in [('positive',tangent>0),('negative',tangent<0),('zero',tangent==0)]:counts[name+'_'+sign]=int((mask&m).sum())
    result=vector_checks(g32.detach(),g64.detach(),g_chain.detach());criteria=result['criteria']
    finite=all(bool(torch.isfinite(t).all()) for t in [g32,g64,g_chain,q,leaf_energy]) and all(torch.isfinite(torch.tensor(e)).item() for e in [E32,E64])
    inactive_nonzero=int(torch.count_nonzero(tangent[~active]));inactive_grad_nonzero=int(torch.count_nonzero(inactive_gradient))
    criteria.update(image_identity=dict(passed=identity),clamp_microprobe=dict(passed=probe['inclusive'] and probe['deterministic']),finite=dict(passed=finite),inactive_tangent=dict(nonzero=inactive_nonzero,inactive_gradient_nonzero=inactive_grad_nonzero,passed=inactive_nonzero==0 and inactive_grad_nonzero==0))
    result.update(image_identity=identity,clamp_microprobe=probe,counts=counts,boundary=boundary(model,direction),leaf_energy=float(leaf_energy.detach()),q_dtype=str(q.dtype),feature_dtypes=dtypes,gradients={n:dict(dtype=str(g.dtype),shape=list(g.shape),sha256=thash(g),values=g.detach().cpu().flatten().tolist()) for n,g in [('g32',g32),('g64',g64),('g_chain',g_chain)]})
    return result
