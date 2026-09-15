import torch
from research_log.T042A_audit.core import probe_raw,classify,LEGACY_STEP,GAINS
from ttie.common_gain import physical_grid

def test_fixed_step_and_gain_preserve_legacy():
    assert LEGACY_STEP==10 and GAINS==[1.75]
    legacy=torch.arange(8).reshape(1,2,2,2).float()/10;active=torch.tensor([True,False,True,False])
    raw=probe_raw(legacy,active,GAINS[0]);assert torch.equal(raw[:,:2],legacy)
    grid=physical_grid(raw)[0,2].flatten();torch.testing.assert_close(grid[active],torch.full((2,),1.75),rtol=0,atol=1e-7)
    assert torch.equal(grid[~active],torch.ones(2))

def test_predeclared_joint_improvement_thresholds():
    def result(p=.57,c=.06415200731653636):return classify(dict(positive_dot_fraction=p,cosine_median=c))
    assert result()=='late real legacy-state effect supported'
    for args in [dict(p=.569),dict(c=.064),dict(c=None)]:assert result(**args)=='late real legacy-state effect not supported / mixed'
