import torch
from research_log.T059H.core import boundaries,assign,reweight,NO,YES

def test_deciles_and_boundary_ties():
 b=boundaries(torch.arange(11,dtype=torch.float64));assert torch.equal(b,torch.arange(1,10,dtype=torch.float64))
 assert assign(torch.tensor([-1.,1.,9.,11.]),b).tolist()==[0,1,9,9]

def test_reweight_keeps_all_and_gate():
 d=torch.arange(10,dtype=torch.float64);b=boundaries(d);loss=torch.arange(10,dtype=torch.float32)/100
 r=reweight(d,d,loss,loss,b);assert r['classification']==NO and abs(r['distance_reweighted_train_huber']-.045)<1e-8
 r=reweight(d,d[-1:],loss,loss[-1:],b);assert r['classification']==YES and sum(x['held_rows'] for x in r['bins'])==1
