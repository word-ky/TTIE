import torch
from research_log.T038A_attribution.core import masks,alignment,classify

def test_active_disjoint_groups_and_t029_degeneracy():
    groups=masks(torch.tensor([True,False,True,False]));a=torch.arange(1,13).reshape(1,3,2,2).float();b=-a
    assert groups['legacy'].sum()==4 and groups['gain'].sum()==2
    assert not (groups['legacy']&groups['gain']).any()
    assert torch.equal(groups['legacy']|groups['gain'],groups['total'])
    assert abs(sum(alignment(a,b,groups[g])['dot'] for g in ['legacy','gain'])-alignment(a,b,groups['total'])['dot'])<1e-12
    assert abs(alignment(a,b,groups['gain'])['cosine']+1)<1e-12
    zero=alignment(a*0,b,groups['gain']);assert zero['cosine'] is None and zero['degenerate'] and not zero['positive_dot']

def test_classification_requires_all_three_conditions():
    def result(cos=-.25,gain=.25,legacy=.5):return classify(dict(gain=dict(cosine_median=cos,positive_dot_fraction=gain),legacy=dict(positive_dot_fraction=legacy)))
    assert result()=='gain-specific mismatch supported'
    for args in [dict(cos=-.249),dict(gain=.36,legacy=.7),dict(legacy=.44),dict(cos=None)]:
        assert result(**args)=='gain-specific mismatch not supported / shared-or-mixed field failure'
