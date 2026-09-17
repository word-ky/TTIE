import os,json
from pathlib import Path
import pytest,torch
from research_log.T058AB.chain import vector_checks as old_vector_checks
from research_log.T058AC.chain import vector_checks
from research_log.T058A_tangent.core import thash

def test_real_cuda32_cpu64_comparison_regression():
    assert torch.cuda.is_available(),'CUDA unavailable; CPU-only substitute is not allowed'
    records=[]
    for label in ['nonzero','zero']:
        g32=(torch.arange(64,device='cuda',dtype=torch.float32).reshape(1,1,8,8)/100 if label=='nonzero' else torch.zeros(1,1,8,8,device='cuda',dtype=torch.float32)).requires_grad_()
        g64=g32.detach().cpu().double().requires_grad_();chain=g64.detach().clone().requires_grad_()
        inputs=[g32,g64,chain];before=[dict(hash=thash(g),device=str(g.device),dtype=str(g.dtype),requires_grad=g.requires_grad) for g in inputs]
        with pytest.raises(RuntimeError,match='same device') as exc:old_vector_checks(*inputs)
        result=vector_checks(*inputs)
        after=[dict(hash=thash(g),device=str(g.device),dtype=str(g.dtype),requires_grad=g.requires_grad) for g in inputs]
        assert before==after and all(g.grad is None for g in inputs)
        assert all(c['passed'] for c in result['criteria'].values())
        assert all(c==dict(device='cpu',dtype='torch.float64',requires_grad=False) for c in result['comparison_copies'].values())
        records.append(dict(case=label,old_error=str(exc.value),old_failed=True,new_passed=True,original_before=before,original_after=after,comparison=result))
    receipt=dict(cuda_available=True,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),cases=records)
    print(json.dumps(receipt),flush=True)
    if os.environ.get('AUTODL_ARTIFACTS_DIR'):
        Path(os.environ['AUTODL_ARTIFACTS_DIR'],'T058AC_cuda_regression.json').write_text(json.dumps(receipt,indent=2)+'\n')
