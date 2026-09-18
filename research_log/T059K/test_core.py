import torch
from research_log.T059K.core import oracle,classify,LABELS

def test_global_tie_and_loo_rank():
 p=dict(target=torch.tensor([0.,0.,2.]),x=torch.tensor([[0.],[1.],[2.]]),image=torch.tensor([1,2,3]),global_index=torch.tensor([4,8,12]));q=dict(x=torch.tensor([[0.]]),image=torch.tensor([1]));five=torch.tensor([[0,1,2]])
 s,r=oracle(p,q,torch.tensor([0.]),five,loo=True,device='cpu');assert r['global_index'].item()==8 and r['candidate_count'].item()==2 and r['distance_rank'].item()==1 and s['global_oracle_huber']==0
 s,r=oracle(p,q,torch.tensor([0.]),five,device='cpu');assert r['global_index'].item()==4 and r['scalar_minimum_tie_count'].item()==2

def test_gate_order():
 assert classify(1,0)==LABELS[0] and classify(0,0)==LABELS[1] and classify(0,1)==LABELS[2]
