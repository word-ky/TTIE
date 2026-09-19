import subprocess,sys
import numpy as np
import torch
from ttie.energy_model import EnergyHead
from research_log.T060B.core import Gain,summarize
from research_log.T060B.verify import reference_analytic,q_analytic

def test_analytic_gain_with_mask_clamp_and_odd_size():
    torch.manual_seed(7);low=torch.rand(1,3,11,13)*1.5;low[:,:,0,0]=0;low[:,:,0,1]=1;clean=torch.rand_like(low);raw=torch.randn(1,3,2,2)*.2;raw[:,2]=0;active=[True,False,True,True];m=Gain(low,raw,torch.tensor(active));loss=(m().double()-clean.double()).square().mean();g,=torch.autograd.grad(loss,m.gain)
    a=reference_analytic(low,raw,active,clean.double().numpy());np.testing.assert_allclose(a,g.numpy(),rtol=3e-5,atol=1e-9)

def test_heads_keep_their_own_normalization():
    torch.manual_seed(7);x=torch.randn(28,requires_grad=True);heads=[EnergyHead(),EnergyHead()]
    for i,h in enumerate(heads):
        h.x_mean.copy_(torch.randn(28)*i);h.x_scale.copy_(torch.rand(28)+.5+i);h.y_scale.fill_(i+1)
        q,=torch.autograd.grad(h(x).sum(),x);a=q_analytic(h.state_dict(),x.detach().numpy());np.testing.assert_allclose(a,q.detach().numpy(),rtol=3e-5,atol=2e-6)
    assert not torch.equal(heads[0].x_scale,heads[1].x_scale)

def test_reference_access_is_blocked(tmp_path):
    code='from pathlib import Path; import research_log.T060B.core as c; c.ROOT=Path('+repr(str(tmp_path))+'); c.firewall(set(),c.ROOT/"out"); (c.ROOT/"source_clean.png").read_bytes()'
    r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True);assert r.returncode!=0 and 'T060B target-free boundary' in r.stderr

def test_fixed_comparison_gates():
    def row(e=.7,o=.6):return dict(reference_norm=1.,nondegenerate=True,E=dict(norm=1.,dot=e,cosine=e,positive_dot=e>0),**{'014':dict(norm=1.,dot=o,cosine=o,positive_dot=o>0)})
    assert 'candidate' in summarize([row() for _ in range(40)])['classification']
    assert 'lacks nondegenerate coverage' in summarize([row() for _ in range(39)])['classification']
    assert 'does not improve' in summarize([row(.62,.60) for _ in range(40)])['classification']
