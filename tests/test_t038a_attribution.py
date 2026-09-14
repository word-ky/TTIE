import torch
import pytest
from research_log.T038A_attribution.core import masks,alignment,classify,check_replay_errors

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

def test_research_lead_replay_limits():
    check_replay_errors(dict(max_feature_abs_error=0.,max_energy_abs_error=0.,max_historical_gradient_abs_error=2.6226043701171875e-6))
    for key,value in [('max_feature_abs_error',1.1e-6),('max_energy_abs_error',1.1e-6),('max_historical_gradient_abs_error',1.1e-5)]:
        receipt=dict(max_feature_abs_error=0.,max_energy_abs_error=0.,max_historical_gradient_abs_error=0.);receipt[key]=value
        with pytest.raises(AssertionError,match='BLOCKED_REPLAY_MISMATCH'):check_replay_errors(receipt)
