import numpy as np
from research_log.T068B.core import knee

def test_singleton_and_degenerate_endpoints():
    assert knee([2],0,0,[0])['k_knee']==0
    a=knee([1,1],0,1,[0,.1]);assert a['k_knee']==1 and a['reason']=='objective_degenerate' and a['u_k']==[None,None]
    b=knee([2,1],0,1,[0,0]);assert b['k_knee']==1 and b['reason']=='motion_degenerate' and b['v_k']==[None,None]
    assert knee([2,3],0,1,[0,.1])['k_knee']==1

def test_motion_curve_and_larger_exact_tie():
    a=knee([4,3,2,1],0,3,[0,1,1,1]);assert a['a_k']==[0,0,0,0] and a['k_knee']==3
    b=knee([4,2,1],0,2,[0,.25,.75]);assert b['s_k']==[0,.25,1] and b['k_knee']==1

def test_clipping_and_nonzero_interval():
    a=knee([9,4,5,0,2],1,4,[0,1,1,1]);assert a['u_k']==[0,0,1,1] and a['k_knee']==3

def test_rms_statistic_known_images():
    # The two RGB differences have RMS sqrt(12.5) and 2; this is not mean absolute motion.
    x=np.array([[0.,0.],[3.,4.],[5.,6.]],dtype=np.float64)
    motions=[0.]+[float(np.sqrt(np.mean((x[k]-x[k-1])**2))) for k in (1,2)]
    a=knee([3,2,1],0,2,motions)
    assert motions[1]==np.sqrt(12.5) and motions[2]==2
    assert a['s_k'][-1]==np.sqrt(12.5)+2


def test_gpu_rms_motion():
    import pytest,torch
    from research_log.T068B.core import gpu_motions
    if not torch.cuda.is_available():pytest.skip('GPU RMS exercised remotely')
    images=[torch.tensor([[[[0.,0.]]]]),torch.tensor([[[[.3,.4]]]]),torch.tensor([[[[.5,.6]]]])]
    got=gpu_motions(images,0,2)
    expected=[0.]+[float(np.sqrt(np.mean((images[k].numpy().astype(np.float64)-images[k-1].numpy().astype(np.float64))**2))) for k in (1,2)]
    np.testing.assert_allclose(got,expected,rtol=0,atol=1e-15)
