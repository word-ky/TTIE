import torch
from research_log.T059M.fit import relative_value
from research_log.T059M.run import classify,LABELS,LIMIT
def test_bank_relative_value_and_gradient():
 p=torch.tensor([2.,5.,8.],requires_grad=True);t=torch.tensor([7.,8.,9.]);a=torch.tensor([0,0,0]);loss=relative_value(p,p[a],t,t[a]);assert torch.equal(loss,torch.nn.functional.huber_loss(torch.tensor([0.,3.,6.]),torch.tensor([0.,1.,2.])));loss.backward();assert torch.allclose(p.grad,torch.tensor([-2/3,1/3,1/3]))
def test_classification():
 assert classify(LIMIT+1,0)==LABELS[0];assert classify(0,0)==LABELS[1];assert classify(0,LIMIT+1)==LABELS[2]
