import torch
from research_log.T059F.core import decompose

def payload(p,t):
 p=torch.tensor(p,dtype=torch.float32);t=torch.tensor(t,dtype=torch.float32);n=len(p)
 return dict(p=p,t=t,delta_p=p-p[0],delta_t=t-t[0],bank_indices=torch.zeros(n,dtype=torch.long),state_indices=torch.arange(n),global_indices=torch.arange(n))

def test_positive_scale_offset():
 r=decompose(payload([7,8,9],[4,6,8]));assert r['banks'][0]['scale']==2 and r['corrected_huber']==0 and r['status']=='DONE'

def test_singleton():
 r=decompose(payload([8],[3]));assert r['zero_scale_count']==1 and r['corrected_huber']==0 and r['status']=='DONE'

def test_negative_dot_stops():
 r=decompose(payload([0,1,2],[0,-1,-2]));assert r['banks'][0]['scale']==0 and r['status']=='PARTIAL' and r['classification'] is None
