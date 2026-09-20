import pytest
from research_log.T067C.core import select
from research_log.T067C.verify import independent_select
from research_log.T067B.core import choices

def test_exact_development_rule_and_independent():
    values=[1.-k/27 for k in range(28)]
    for fs in [0,1,9,12,27]:
        p=[.1]*fs+[.5]*(28-fs);a=select(values,p,27);b=choices(values,p,27)[7]
        assert a=={k:v for k,v in b.items() if k!='lambda_value'} and a==independent_select(values,p,27)

def test_no_fallback():
    with pytest.raises(AssertionError):select([1.-k/27 for k in range(28)],[.1]*28,27)
