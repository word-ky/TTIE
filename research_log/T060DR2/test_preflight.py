import torch
from test_semantic_ttt import scorer,RECEIPT
from ttie.semantic_ttt import FixedObjective
from ttie.common_gain import CommonRegion2
from ttie.energy_model import EnergyHead,features
from research_log.T060DR2.preflight import fresh_gain,comparison

def test_online_gain_matches_direct_model_gradient():
    torch.manual_seed(7);torch.set_num_threads(1)
    low=torch.full((1,3,16,18),.1);s=scorer();obj=FixedObjective(s,low,RECEIPT);head=EnergyHead().eval().requires_grad_(False)
    gain,trace=fresh_gain(low,obj,s,head)
    m=CommonRegion2(obj.active);x=features(obj,s(m(low)),m.physical_grid()[:,:2]);direct,=torch.autograd.grad(head(x).sum(),m.raw)
    torch.testing.assert_close(gain,direct[:,2:3].flatten(),rtol=3e-5,atol=2e-6)
    assert comparison(gain,direct[:,2:3])['passed']
    assert not comparison(gain,-gain)['passed']
    assert comparison(torch.zeros(4),torch.zeros(4))['passed']
    assert not comparison(gain,torch.zeros(4))['passed']
