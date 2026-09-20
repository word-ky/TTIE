import torch
from types import SimpleNamespace
from research_log.T062CR2.core import trajectory
from research_log.T062A.core import trajectory as accepted

def test_exact_27_update_prefix():
    torch.set_num_threads(1);torch.manual_seed(7)
    low=torch.rand(1,3,32,48)*.3
    gate=SimpleNamespace(active=torch.tensor([True,False,True,True]),winner=torch.tensor([0,0,1,0]))
    before=low.clone();old=accepted(low,gate);new=trajectory(low,gate)
    for key in ['images','states','components']:assert torch.equal(new[key],old[key][:28])
    for key in ['gradients','pre_box']:assert torch.equal(new[key],old[key][:27])
    assert new['values']==old['values'][:28] and new['selected_step']==27
    assert torch.equal(low,before) and torch.count_nonzero(new['states'][0])==0
