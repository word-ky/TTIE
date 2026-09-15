import torch
from research_log.T041A_audit.core import probe_raw,masks,classify,SOURCE_POSITIVE,SOURCE_COSINE
from ttie.common_gain import physical_grid

def test_fixed_gains_preserve_selected_legacy_and_inactive_identity():
    legacy=torch.arange(8).reshape(1,2,2,2).float()/10;active=torch.tensor([True,False,True,False])
    for gain in [1.25,1.75]:
        raw=probe_raw(legacy,active,gain);assert torch.equal(raw[:,:2],legacy)
        grid=physical_grid(raw)[0,2].flatten()
        torch.testing.assert_close(grid[active],torch.full((2,),gain),atol=1e-7,rtol=0)
        assert torch.equal(grid[~active],torch.ones(2))
    groups=masks(active);assert torch.equal(groups['legacy']|groups['gain'],groups['total']) and not (groups['legacy']&groups['gain']).any()

def test_total_group_joint_source_deficit_gate():
    def result(p=SOURCE_POSITIVE-.21,c=SOURCE_COSINE-.26):return classify(dict(positive_dot_fraction=p,cosine_median=c))
    assert result()=='real selected-state field deficit beyond source high-gain supported'
    for args in [dict(p=SOURCE_POSITIVE-.19),dict(c=SOURCE_COSINE-.24),dict(c=None)]:assert result(**args)=='real selected-state field deficit beyond source high-gain not supported / mixed'
