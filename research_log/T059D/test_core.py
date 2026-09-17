import torch
from research_log.T059D.core import average_ranks,spearman,bank_metrics

def test_average_ties_and_degenerate_ranks():
    assert torch.equal(average_ranks(torch.tensor([3.,1.,1.,2.])),torch.tensor([4.,1.5,1.5,3.],dtype=torch.float64))
    assert abs(spearman(torch.tensor([1.,1.,2.]),torch.tensor([2.,2.,1.]))[0]+1)<1e-12
    assert spearman(torch.tensor([1.]),torch.tensor([2.]))==(None,'singleton')
    assert spearman(torch.ones(3),torch.arange(3.))[0] is None

def test_anchor_offset_removal_and_earliest_tie():
    p=torch.tensor([4.,4.,5.,9.]);t=torch.tensor([1.,0.,2.,3.]);cos=torch.tensor([1.,-1.,.5,0.]);mask=torch.tensor([True,True,True,False]);bi=torch.tensor([7,7,7,9]);si=torch.tensor([0,1,2,0]);gi=torch.arange(4)
    banks,dp,dt,anchors=bank_metrics(p,t,cos,cos,mask,mask,bi,si,gi)
    assert banks[0]['predicted_argmin_state_index']==0 and banks[0]['argmin_regret']==1.
    assert banks[1]['relative_huber']==0 and banks[1]['spearman'] is None and banks[1]['detail']['positive_fraction'] is None
    shifted=bank_metrics(p+11,t+5,cos,cos,mask,mask,bi,si,gi)
    assert torch.equal(dp,shifted[1]) and torch.equal(dt,shifted[2])
    assert anchors.tolist()==[0,0,0,3]
