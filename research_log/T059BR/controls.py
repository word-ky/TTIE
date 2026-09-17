"""Fixed CPU frozen-head controls; feature differences are descriptive only."""
import math,torch
from ttie.sobolev_train import cosine
from research_log.T058A_tangent.core import thash

def scalar_gate(actual,expected):
    error=abs(actual-expected);limit=max(2e-6,2e-5*max(abs(actual),abs(expected)))
    return dict(actual=actual,expected=expected,abs_error=error,tolerance=limit,margin=limit-error,atol=2e-6,rtol=2e-5,passed=math.isclose(actual,expected,abs_tol=2e-6,rel_tol=2e-5))

def detail_control(head,features,jac,reference,mask):
    leaf=features.detach().cpu().float().requires_grad_(True)
    value=head(leaf);q,=torch.autograd.grad(value.sum(),leaf)
    grad=torch.einsum('bfi,bf->bi',jac,q)
    values=cosine(grad[mask],reference[mask])
    assert all(torch.isfinite(t).all() for t in [value,q,grad,values])
    stats=dict(direction_rows=int(mask.sum()),ineligible_rows=len(mask)-int(mask.sum()),positive_fraction=float((values>0).float().mean()),median_cosine=float(torch.quantile(values,.5)),direction_loss=float(((1-values)/2).mean()),q_sha256=thash(q),gradient_sha256=thash(grad),zero_rows=int((grad.norm(dim=1)==0).sum()))
    return stats,q.detach(),grad.detach()

def chain_control(grad,accepted):
    a=grad.double();b=accepted.double();delta=a-b;an=a.norm(dim=1);bn=b.norm(dim=1);valid=(an>0)&(bn>0)
    element_ok=torch.isclose(grad,accepted,atol=2e-6,rtol=2e-5);rows_ok=element_ok.all(1);zero_match=torch.equal(an==0,bn==0)
    return dict(passed=bool(rows_ok.all()) and zero_match,rows=len(a),passing_rows=int(rows_ok.sum()),atol=2e-6,rtol=2e-5,max_abs=float(delta.abs().max()),max_l2=float(delta.norm(dim=1).max()),min_nonzero_cosine=float(((a[valid]*b[valid]).sum(1)/(an[valid]*bn[valid])).min()) if valid.any() else None,accepted_zero_rows=int((bn==0).sum()),reconstructed_zero_rows=int((an==0).sum()),zero_status_matches=zero_match,minimum_element_margin=float((2e-6+2e-5*accepted.abs()-(grad-accepted).abs()).min()),accepted_sha256=thash(accepted),reconstructed_sha256=thash(grad))

def difference(x,phi):
    delta=(x.double()-phi.double()).abs();flat=delta.flatten();idx=int(flat.argmax());row,feature=divmod(idx,28)
    return dict(diagnostic_only=True,shape=list(x.shape),max_abs=float(flat.max()),mean_abs=float(flat.mean()),quantiles={str(q):float(torch.quantile(flat,q)) for q in [0.,.5,.9,.95,.99,1.]},per_feature_max_abs=delta.amax(0).tolist(),per_feature_mean_abs=delta.mean(0).tolist(),per_feature_nonidentical_counts=(x!=phi).sum(0).tolist(),nonidentical_rows=int((x!=phi).any(1).sum()),maximum_location=dict(row=row,feature=feature,x=float(x[row,feature]),phi=float(phi[row,feature])),x_sha256=thash(x),phi_sha256=thash(phi))
