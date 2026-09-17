import torch
from research_log.T059G.core import nearest,metrics,classify,LABELS

def test_tie_and_image_exclusion():
 x=torch.tensor([[0.],[0.],[2.]]);ids=torch.tensor([1,2,3]);g=torch.tensor([2,8,12]);q=torch.tensor([[0.]])
 assert nearest(q,x,torch.tensor([1]),ids,g,device='cpu')['global_index'].item()==2
 r=nearest(q,x,torch.tensor([1]),ids,g,loo=True,device='cpu');assert r['global_index'].item()==8 and r['tie_count'].item()==1

def test_zero_neighbor_not_dropped():
 t=torch.ones(2,64);p=t.clone();p[0]=0;r=metrics(torch.zeros(2),torch.zeros(2),p,t)
 assert r['detail_eligible']==2 and r['detail_positive']==.5 and r['detail_median']==.5 and r['neighbor_noneligible_on_eligible_query']==1

def test_gate_precedence():
 good=dict(relative_huber=.01,detail_positive=.9,detail_median=.7);bad={**good,'relative_huber':1}
 assert classify(bad,good)['classification']==LABELS[0]
 assert classify(good,bad)['classification']==LABELS[1]
 assert classify(good,good)['classification']==LABELS[2]
