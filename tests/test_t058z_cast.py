import torch
from types import SimpleNamespace
from ttie.energy_model import features
from research_log.T058Z.shadow import features64

def test_exact_same_input_cast_isolation():
    torch.manual_seed(7)
    obj=SimpleNamespace(active=torch.tensor([True,False,True,True]),winner=torch.tensor([0,1,0,1]),evidence=torch.randn(4,dtype=torch.float64),calibration={'tau':[.1,.2],'scale':[.3,.4]})
    scores=torch.randn(4,2,dtype=torch.float64,requires_grad=True);grid=torch.randn(1,2,2,2,dtype=torch.float64)
    phi,dtypes=features64(obj,scores,grid)
    assert torch.equal(features(obj,scores,grid),phi.float())
    assert all(x=='torch.float64' for x in dtypes.values())
    g,=torch.autograd.grad(phi.sum(),scores);assert torch.isfinite(g).all()
