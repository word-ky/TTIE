import numpy as np
from research_log.T069BR.core import check,score,threshold,diagnose

def test_fixed_float32_path_identity():
    b=np.zeros(12);a=b.copy();a[0]=2e-7;assert check(a,b)['passes']
    a[0]=2.01e-7;assert not check(a,b)['passes']

def test_fixed_float64_linearity():
    b=np.zeros(12);a=b.copy();a[0]=1e-12;assert check(a,b,True)['passes']
    a[0]=1.01e-12;assert not check(a,b,True)['passes']
    assert check(np.ones(12),np.ones(12),True)['max_abs']==0

def test_symmetric_score_threshold_and_flags():
    g=np.zeros((3,12));g[:,0]=[1.,-1.,2.];assert score(g)['R_cancel']==.5
    assert score(np.zeros_like(g))['reason']=='zero_component_gradient'
    t=threshold([dict(R_cancel=k/100) for k in range(100)]);assert t==.98 and not(t>t)
    rows=[dict(unsafe=True,above_T99_cancel=True)]+[dict(unsafe=False,above_T99_cancel=True)]*5
    assert diagnose(rows)['classification']=='GRADIENT_CANCELLATION_SIGNAL_PRESENT'
    assert diagnose(rows+[dict(unsafe=False,above_T99_cancel=True)])['classification']=='GRADIENT_CANCELLATION_SIGNAL_ABSENT'


def test_cuda_pinned_endpoint_adapter():
    import pytest,torch
    if not torch.cuda.is_available():pytest.skip('CUDA test runs on A6000')
    from research_log.T069BN.run import compute
    from research_log.T069BN.verify import equal
    from research_log.T069BR.run import gradients
    from research_log.T069BR.verify import recompute
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    low=torch.linspace(.1,.8,3*32*32).reshape(1,3,32,32)
    trace=dict(active=torch.ones(2,2,dtype=torch.bool),states=torch.zeros(1,1,3,2,2))
    old=compute(low,trace,0,torch.float32)
    trace['gradients']=torch.tensor(old['direct']['values'],dtype=torch.float32).reshape(1,1,3,2,2)
    a=gradients(low,trace,0,old['output_hash']);b=recompute(low,trace,0,old['output_hash']);equal(a,b)
    assert a['path32']['check']['passes'] and a['double_check']['passes']
