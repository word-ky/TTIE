import torch
from test_semantic_ttt import scorer,RECEIPT
from ttie.energy_model import EnergyHead
from research_log.T060DR2.infer import episode
from ttie.common_gain_ttt import trajectory
from research_log.T060DR2.verify import adam_replay

def test_literal_A_and_only_gain_substitution_B():
    torch.manual_seed(7);torch.set_num_threads(1);low=torch.full((1,3,16,18),.1);s=scorer();a=EnergyHead().eval().requires_grad_(False);b=EnergyHead().eval().requires_grad_(False)
    _,literal,ld=trajectory(low,s,RECEIPT,a)
    outputs={}
    for name in ['A','B']:
        result,t,d,g=episode(name,low,s,RECEIPT,a,b);outputs[name]=(t,d,g)
        assert len(t['images'])==41 and len(g)==40 and torch.count_nonzero(t['states'][0])==0
        assert t['diagnostics']['parameter_count']==12
        scores=[float(a(x)) for x in t['features']];assert scores==d['scores'];assert d['selected_step']==min(range(41),key=lambda j:(scores[j],j))
        assert torch.equal(result['image'],t['images'][d['selected_step']])
        assert adam_replay(t['states'],[v['hybrid'] for v in g],t['gate'],t['diagnostics']['action_box'])<3e-5
    ta,da,ga=outputs['A'];tb,db,gb=outputs['B']
    assert torch.equal(ta['states'],literal['states']) and da==ld
    torch.testing.assert_close(ga[0]['g014'][:,:2],gb[0]['hybrid'][:,:2],rtol=0,atol=0)
    assert not torch.equal(ga[0]['hybrid'][:,2:3],gb[0]['hybrid'][:,2:3])
    for g in gb:
        assert torch.equal(g['hybrid'][:,:2],g['g014'][:,:2])
        assert torch.equal(g['hybrid'][:,2:3],g['g_E_gain'])
