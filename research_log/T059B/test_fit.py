import torch
from ttie.energy_model import EnergyHead
from ttie.stop_quality import train_head
from ttie.sobolev_train import raw_gradient,cosine
from research_log.T059B.fit import train_fixed,dual_terms,verdict

def test_zero_detail_reproduces_original_training_exactly():
    torch.manual_seed(21);x=torch.randn(8,28);mse=torch.rand(8,dtype=torch.float64)+.01
    legacy=dict(jacobian=torch.randn(8,28,8),reference_gradient=torch.randn(8,8),direction_mask=torch.ones(8,dtype=torch.bool))
    detail=dict(jacobian=torch.randn(8,28,64),reference_gradient=torch.randn(8,64),direction_mask=torch.zeros(8,dtype=torch.bool))
    def extra(head,prediction,features,indices):
        g=raw_gradient(head,prediction,features,legacy['jacobian'][indices],create_graph=True)
        return ((1-cosine(g,legacy['reference_gradient'][indices]))/2).mean()
    old,old_history=train_head(x,mse,head_factory=EnergyHead,extra_loss=extra)
    new,history,initial=train_fixed(x,mse,legacy,detail)
    assert all(torch.equal(t,new.state_dict()[n]) for n,t in old.state_dict().items())
    assert len(history)==100 and all(h['detail_sobolev']==0 for h in history)
    assert [h['train_huber'] for h in old_history]==[h['train_huber'] for h in history]

def test_both_terms_backpropagate_and_fixed_gates():
    torch.manual_seed(7);head=EnergyHead();x=torch.randn(3,28,requires_grad=True);idx=torch.arange(3)
    records=[dict(jacobian=torch.randn(3,28,n),reference_gradient=torch.randn(3,n),direction_mask=torch.tensor([True,False,True])) for n in [8,64]]
    losses=dual_terms(head,head.standardized(x),x,idx,*records);(losses[0]+losses[1]).backward()
    assert all(torch.isfinite(l) for l in losses) and any(p.grad is not None and p.grad.abs().sum()>0 for p in head.parameters())
    old=dict(positive_fraction=.95,median_cosine=.90,value_huber=1.5*.051005665212869644);detail=dict(positive_fraction=.75,median_cosine=.50)
    assert verdict(old,detail)['classification']=='dual-tangent source fit feasible'
    detail['median_cosine']-=1e-8;assert verdict(old,detail)['classification']=='dual-tangent source fit not feasible under fixed recipe'
