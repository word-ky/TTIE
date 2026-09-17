import torch
from research_log.T058AA.decomposition import microprobe,parts,close_check

def test_mixed_boundary_reverse_outside_one_sided_interval():
    z=torch.tensor([.5,0.,1.,0.,1.,0.],dtype=torch.float64)
    tangent=torch.tensor([2.,1.,-1.,-1.,1.,0.],dtype=torch.float64)
    q=torch.tensor([3.,4.,-5.,-6.,7.,8.],dtype=torch.float64)
    result=parts(z,tangent,q,torch.ones_like(z,dtype=torch.bool))
    assert microprobe()['inclusive'] and microprobe()['deterministic']
    assert result['I']==6. and result['B_plus']==9. and result['B_minus']==13.
    assert result['d_rev_pred']==28. and result['surrogate_derivative']==28.
    assert result['d_plus_pred']==15. and result['d_minus_pred']==19.
    assert result['zero_tangent']==0. and result['counts']['zero_tangent_boundary']==1
    assert not close_check(28.,19.,1e-6)['passed']
