import sys,math
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'research_log/T044A_audit'))
from core import score,physical
from association import summarize,verdict,SUPPORTED,NEGATIVE
from ttie.common_gain import physical_grid

def test_physical_displacement_active_mask_and_zero_gate():
    a=torch.zeros(1,3,2,2);b=a.clone()
    b[0,0,0,0]=math.atanh(.5);b[0,1,0,0]=math.atanh(.5)
    b[:,:,1,1]=3 # ignored inactive region
    active=[True,False,False,False]
    assert torch.equal(physical(b),physical_grid(b.double())[:,:2])
    expected=math.tanh(float(b[0,0,0,0]))/2
    assert abs(score(a,b,active)-expected)<1e-8
    assert score(a,b,[False]*4)==0
    b[:,2].fill_(-2)
    assert abs(score(a,b,active)-expected)<1e-8

def test_tied_auc_spearman_and_joint_gate():
    s=summarize([0,1,1,2],[2,-1,1,-2])
    assert s['roc_auc']==.875
    assert abs(s['spearman']+0.9486832980505138)<1e-12
    assert verdict(.75,-.35)==SUPPORTED
    assert verdict(.749999,-.9)==NEGATIVE and verdict(.9,-.349999)==NEGATIVE
