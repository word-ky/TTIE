"""Two fixed T014 source heads; inference uses the unchanged T013 EnergyHead."""
import torch
from .energy_model import EnergyHead,train_energy,RECIPE
from .stop_quality import train_head

DERIVATIVE_LOSS=dict(equation='L_value + mean((1-cos(J^T grad_f E,g_ref))/2)',
    weights=[1.,1.],rows='active and nonzero reference gradient',
    zero_predicted_gradient='cosine 0; zero denominator replaced with 1',
    chain_rule='unstandardized energy with raw-feature Jacobian; normalization differentiated',
    empty_direction_batch='zero directional loss')


def raw_gradient(head,prediction,x,jacobian,*,create_graph=False):
    df=torch.autograd.grad((prediction*head.y_scale+head.y_mean).sum(),x,create_graph=create_graph)[0]
    return torch.einsum('bfi,bf->bi',jacobian,df)


def cosine(predicted,reference):
    denominator=predicted.norm(dim=-1)*reference.norm(dim=-1)
    return (predicted*reference).sum(-1)/torch.where(denominator>0,denominator,torch.ones_like(denominator))


def train_pair(x,mse,records):
    control,control_history=train_energy(x,mse)
    jac=records['jacobian'].cpu().float();truth=records['reference_gradient'].cpu().float();mask=records['direction_mask'].cpu()
    def extra(head,prediction,features,indices):
        grad=raw_gradient(head,prediction,features,jac[indices],create_graph=True)
        valid=mask[indices]
        return ((1-cosine(grad[valid],truth[indices][valid]))/2).mean() if valid.any() else grad.sum()*0
    primary,primary_history=train_head(x,mse,head_factory=EnergyHead,extra_loss=extra)
    return dict(value_only_control=control,sobolev_primary=primary),dict(value_only_control=control_history,sobolev_primary=primary_history)


def training_statistics(head,x,mse,records):
    x=x.detach().cpu().float().requires_grad_();head=head.cpu()
    prediction=head.standardized(x)
    grad=raw_gradient(head,prediction,x,records['jacobian'].float())
    mask=records['direction_mask'];values=cosine(grad[mask],records['reference_gradient'][mask])
    target=(((mse.double()+1e-6).log().float()-head.y_mean)/head.y_scale)
    return dict(value_huber=float(torch.nn.functional.huber_loss(prediction.detach(),target,delta=1.)),
        direction_rows=int(mask.sum()),cosines=values.detach().tolist(),
        positive_fraction=float((values>0).float().mean()) if len(values) else 0.,
        median_cosine=float(torch.quantile(values,.5)) if len(values) else 0.,
        direction_loss=float(((1-values)/2).mean()) if len(values) else 0.)
