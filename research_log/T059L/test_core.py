import torch,pytest
from research_log.T059L.core import displacement,nearest,classify,LABELS,LIMIT
def test_anchor_and_tie():
 x=torch.tensor([[1.,2.],[2.,4.],[9.,8.],[10.,10.]]);bank=torch.tensor([0,0,1,1]);state=torch.tensor([0,1,0,1]);a,dx=displacement(x,bank,state);assert a.tolist()==[0,0,2,2];assert torch.equal(dx[:2],dx[2:]);n=nearest(dx,dx,bank,bank,torch.arange(4),True,'cpu');assert n['index'].tolist()==[2,3,0,1]
 q=nearest(torch.zeros(1,2),torch.zeros(3,2),torch.tensor([4]),torch.arange(3),torch.tensor([2,5,9]),device='cpu');assert q['index'].item()==0 and q['tie_count'].item()==3
@pytest.mark.parametrize('state',[[1,1],[0,0]])
def test_missing_or_duplicate_anchor(state):
 with pytest.raises(AssertionError):displacement(torch.zeros(2,28),torch.zeros(2,dtype=torch.long),torch.tensor(state))
def test_classification():
 assert classify(LIMIT+1,0)==LABELS[0];assert classify(0,0)==LABELS[1];assert classify(0,LIMIT+1)==LABELS[2]
