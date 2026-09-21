import numpy as np
from research_log.T069BN.core import comparison,bookkeeping,audit_status

def test_original_tolerance_and_trace_comparison():
    b=np.zeros(12);a=b.copy();a[0]=2e-7
    assert comparison(a,b)['passes']
    a[0]=2.01e-7;c=comparison(a,b)
    assert not c['passes'] and c['failed_coordinates']==1 and c['max_abs']==2.01e-7
    assert comparison(np.ones(12),np.ones(12))['l2_relative']==0

def test_sum_residual_and_score_sensitivity():
    gs=np.zeros((3,12));gs[:,0]=[1.,-1.,2.];d=np.zeros(12);d[0]=1.5
    s=bookkeeping(gs,d,d)
    assert s['sum_vs_direct']['max_abs']==.5
    assert s['sum_vs_direct']['l2_relative']==1/3
    assert s['R_cancel']==.5 and s['R_directnorm']==.625 and s['score_sensitivity']==.125

def test_repeat_acceptance():
    z=np.zeros(12);ok=comparison(z,z);bad=comparison(np.ones(12),z)
    row=dict(float32=dict(direct_vs_trace=ok),repeat=dict(direct_comparison=ok))
    assert audit_status([row])=='GRADIENT_NUMERICS_CHARACTERIZED'
    row['repeat']['direct_comparison']=bad
    assert audit_status([row])=='GRADIENT_PATH_MISMATCH'


def test_cuda_shared_separate_and_repeat():
    import pytest,torch
    if not torch.cuda.is_available():pytest.skip('CUDA adapter integration runs on A6000')
    from research_log.T069BN.run import compute
    from research_log.T069BN.verify import compute as independent,equal
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    low=torch.linspace(.1,.8,3*32*32).reshape(1,3,32,32)
    trace=dict(active=torch.ones(2,2,dtype=torch.bool),states=torch.zeros(1,1,3,2,2))
    for dtype in [torch.float32,torch.float64]:
        a=compute(low,trace,0,dtype);b=compute(low,trace,0,dtype);c=independent(low,trace,0,dtype)
        equal(a,b);equal(a,c)
