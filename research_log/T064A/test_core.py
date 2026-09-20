import torch
from ttie.common_gain import CommonRegion2
from research_log.T064A.reconstruct import render
from research_log.T064A.analyze import oracle

def test_saved_states_render_exactly():
 torch.set_num_threads(1);torch.manual_seed(7);x=torch.rand(1,3,32,48);active=torch.tensor([True,False,True,True]);states=torch.rand(28,1,3,2,2)*.2;states[0].zero_()
 images=render(x,dict(active=active,states=states));model=CommonRegion2(active)
 with torch.no_grad():model.raw.copy_(states[11]);expected=model(x)
 assert torch.equal(images[11],expected) and torch.equal(images[0],x)

def test_safe_oracle_earliest_tie():
 p=[0.]*28;p[3]=p[6]=5.;r=oracle(p,10.)
 assert r==dict(reachable_steps=[3,6],oracle_step=3,safety_unreachable=False)

def test_unreachable_reporting_only():
 p=[0.]*28;p[9]=1.;r=oracle(p,10.)
 assert r==dict(reachable_steps=[],oracle_step=9,safety_unreachable=True)
