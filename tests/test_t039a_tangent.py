import torch
from research_log.T039A_tangent.core import probe_raw,masks,classify
from ttie.common_gain import physical_grid

def test_fixed_probes_preserve_legacy_and_inactive_identity():
    legacy=torch.arange(8).reshape(1,2,2,2).float()/10;active=torch.tensor([True,False,True,False])
    for gain in [.75,1.,1.25]:
        raw=probe_raw(legacy,active,gain);assert torch.equal(raw[:,:2],legacy)
        value=physical_grid(raw)[0,2].flatten()
        torch.testing.assert_close(value[active],torch.full((2,),gain),rtol=0,atol=1e-7)
        assert torch.equal(value[~active],torch.ones(2))
    group=masks(active);assert torch.equal(group['legacy']|group['gain'],group['total']) and not (group['legacy']&group['gain']).any()

def test_joint_deficit_gate():
    def result(lc=.5,gc=.25,lp=.75,gp=.5):return classify(dict(legacy=dict(cosine_median=lc,positive_dot_fraction=lp),gain=dict(cosine_median=gc,positive_dot_fraction=gp)))
    assert result()=='source gain-tangent deficit supported'
    assert result(gc=.251)=='source gain-tangent deficit not supported / real-domain effect remains plausible'
    assert result(gp=.56)=='source gain-tangent deficit not supported / real-domain effect remains plausible'
