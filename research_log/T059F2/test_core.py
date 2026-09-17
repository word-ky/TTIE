import torch
from research_log.T059F2.core import audit

def payload(p,t):
 p=torch.tensor(p,dtype=torch.float32);t=torch.tensor(t,dtype=torch.float32);n=len(p)
 return dict(p=p,t=t,delta_p=p-p[0],delta_t=t-t[0],bank_indices=torch.zeros(n,dtype=torch.long),state_indices=torch.arange(n),global_indices=torch.arange(n))

def test_positive():
 r=audit(payload([7,8,9],[4,6,8]));assert r['banks'][0]['admissible_scale']==2 and r['limit_huber']==0

def test_boundary_preserves_original_regret():
 r=audit(payload([0,1,2],[0,-1,-2]));b=r['banks'][0];assert b['category']=='positive_scale_boundary' and b['admissible_scale'] is None
 assert abs(b['spearman']+1)<1e-14 and b['argmin_regret']==2 and r['limit_huber']==float(torch.tensor([0.,.5,1.5]).mean())

def test_singleton_and_constant_degenerate():
 for p,t in [([8],[3]),([8,8],[3,4])]:
  b=audit(payload(p,t))['banks'][0];assert b['category']=='scale_unidentified_degenerate' and b['admissible_scale'] is None and b['spearman'] is None
