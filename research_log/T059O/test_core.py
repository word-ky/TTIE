import torch,pytest
from research_log.T059O.core import context,nearest,classify,LABELS,LIMIT
def test_context_preserves_anchor():
 x=torch.tensor([[1.,2.],[2.,4.],[9.,8.],[10.,10.]]);b=torch.tensor([0,0,1,1]);s=torch.tensor([0,1,0,1]);q=context(x,torch.zeros(2),torch.ones(2),b,s);assert q['anchor'].tolist()==[0,0,2,2];assert torch.equal(q['z'][:2,:2],q['z'][2:,:2]);assert not torch.equal(q['z'][:2,2:],q['z'][2:,2:])
 n=nearest(torch.zeros(1,4),torch.zeros(3,4),torch.tensor([9]),torch.arange(3),torch.tensor([2,4,8]),device='cpu');assert n['global_index'].item()==2 and n['tie_count'].item()==3
@pytest.mark.parametrize('s',[[0,0],[1,1]])
def test_unique_anchor(s):
 with pytest.raises(AssertionError):context(torch.zeros(2,28),torch.zeros(28),torch.ones(28),torch.zeros(2,dtype=torch.long),torch.tensor(s))
def test_gates():
 assert classify(1,0)==LABELS[0];assert classify(0,1)==LABELS[1];assert classify(0,0)==LABELS[2]
