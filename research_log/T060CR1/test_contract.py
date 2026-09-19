import torch,numpy as np
from ttie.energy_model import EnergyHead,features
from ttie.semantic_ttt import FixedObjective
from ttie.common_gain import CommonRegion2,CommonBox
from test_semantic_ttt import scorer,RECEIPT
from research_log.T060CR1.core import trajectory

def test_joint_trajectory_and_coordinate_guidance():
    torch.manual_seed(7);torch.set_num_threads(1);image=torch.full((1,3,16,18),.1);s=scorer();h014=EnergyHead().eval().requires_grad_(False);he=EnergyHead().eval().requires_grad_(False)
    result,t,d,traces=trajectory(image,s,RECEIPT,h014,he)
    assert t['states'][0].shape==(1,3,2,2) and torch.count_nonzero(t['states'][0])==0
    assert len(traces)==t['diagnostics']['steps']==40 and t['diagnostics']['parameter_count']==12
    assert d['selected_step']==min(range(41),key=lambda i:(d['scores'][i],i))
    obj=FixedObjective(s,image,RECEIPT);m=CommonRegion2(obj.active);box=CommonBox(obj,2);optimizer=torch.optim.Adam([m.raw],lr=.03)
    for i,a in enumerate(traces):
        assert torch.equal(a['hybrid'][:,:2],a['g014'][:,:2]);np.testing.assert_allclose(a['J_gain'].double().numpy().T@a['q_E'].double().numpy(),a['hybrid'][:,2:3].flatten(),rtol=2e-5,atol=2e-6)
        if i in [0,20,39]:
            check=CommonRegion2(obj.active);check.raw.data.copy_(t['states'][i]);x=features(obj,s(check(image)),check.physical_grid()[:,:2]);g014,=torch.autograd.grad(h014(x).sum(),check.raw,retain_graph=True);ge,=torch.autograd.grad(he(x).sum(),check.raw)
            assert torch.equal(a['g014'],g014);torch.testing.assert_close(a['g_E_gain'],ge[:,2:3],rtol=2e-5,atol=2e-6)
        optimizer.zero_grad(set_to_none=True);m.raw.grad=a['hybrid'].clone();optimizer.step();box(m)
        assert torch.equal(m.raw,t['states'][i+1])
    for i,x in enumerate(t['features']):assert float(h014(x))==d['scores'][i]
    assert torch.equal(result['raw'],t['states'][d['selected_step']])
    from research_log.T060CR1.verify import replay
    assert replay(t['states'],traces,t['gate'],t['diagnostics']['action_box'])<3e-5

def test_inference_firewall():
    import subprocess,sys
    code='from pathlib import Path; from research_log.T060CR1.core import firewall,ROOT; firewall(set(),ROOT/"runs/test_output"); (ROOT/"shared/t036a/normal/Train/Normal/normal00353.png").read_bytes()'
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode!=0 and 'target-free boundary' in result.stderr
