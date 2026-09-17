"""One fixed T014 recipe with one additional matched detail loss."""
import torch
from ttie.energy_model import EnergyHead
from ttie.stop_quality import train_head
from ttie.sobolev_train import cosine,training_statistics
from research_log.T058A_tangent.core import thash

def detail_statistics(head,x,jac,truth,mask):
    leaf=x.detach().cpu().float().requires_grad_(True)
    q,=torch.autograd.grad(head(leaf).sum(),leaf)
    grad=torch.einsum('bfi,bf->bi',jac,q)
    values=cosine(grad[mask],truth[mask]);assert torch.isfinite(grad).all() and torch.isfinite(values).all()
    return dict(direction_rows=int(mask.sum()),positive_fraction=float((values>0).float().mean()),median_cosine=float(torch.quantile(values,.5)),direction_loss=float(((1-values)/2).mean()),gradient_sha256=thash(grad),zero_rows=int((grad.norm(dim=1)==0).sum()))

def dual_terms(head,prediction,x,indices,legacy,detail,create_graph=True):
    q,=torch.autograd.grad((prediction*head.y_scale+head.y_mean).sum(),x,create_graph=create_graph)
    losses=[]
    for records in [legacy,detail]:
        grad=torch.einsum('bfi,bf->bi',records['jacobian'][indices],q);valid=records['direction_mask'][indices]
        assert torch.isfinite(grad).all()
        loss=((1-cosine(grad[valid],records['reference_gradient'][indices][valid]))/2).mean() if valid.any() else grad.sum()*0
        assert torch.isfinite(loss);losses.append(loss)
    return losses

def train_fixed(x,mse,legacy,detail):
    batch_losses=[];initial={}
    def factory():
        head=EnergyHead();initial.update({k:thash(v) for k,v in head.state_dict().items()});return head
    def extra(head,prediction,features,indices):
        assert torch.isfinite(prediction).all()
        losses=dual_terms(head,prediction,features,indices,legacy,detail)
        batch_losses.append(dict(n=len(indices),legacy=float(losses[0].detach()),detail=float(losses[1].detach())))
        return losses[0]+losses[1]
    head,history=train_head(x,mse,head_factory=factory,extra_loss=extra)
    batches=(len(x)+255)//256;assert len(history)==100 and len(batch_losses)==100*batches
    for epoch,h in enumerate(history):
        group=batch_losses[epoch*batches:(epoch+1)*batches]
        h['legacy_sobolev']=sum(b['n']*b['legacy'] for b in group)/len(x);h['detail_sobolev']=sum(b['n']*b['detail'] for b in group)/len(x)
        h['total']=h['train_huber']+h['legacy_sobolev']+h['detail_sobolev']
    assert all(torch.isfinite(v).all() for v in head.state_dict().values())
    return head,history,initial

def verdict(legacy,detail):
    gates=dict(detail_positive=detail['positive_fraction']>=.75,detail_cosine=detail['median_cosine']>=.5,legacy_positive=legacy['positive_fraction']>=.95,legacy_cosine=legacy['median_cosine']>=.9,value_huber=legacy['value_huber']<=1.5*.051005665212869644)
    return dict(classification='dual-tangent source fit feasible' if all(gates.values()) else 'dual-tangent source fit not feasible under fixed recipe',gates=gates,value_huber_limit=1.5*.051005665212869644)
