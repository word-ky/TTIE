import pytest
from research_log.T067A.core import first_safe
from research_log.T067A.verify import independent_first

def test_earliest_crossing_and_exact_threshold():
    p=[.1,.5,.4,.9];assert first_safe(p,3)==independent_first(p,3)==1

def test_identity_is_allowed():
    p=[.9,.1,.9];assert first_safe(p,2)==independent_first(p,2)==0

def test_no_fallback_or_outside_prefix():
    for fn in [first_safe,independent_first]:
        with pytest.raises(AssertionError):fn([.1,.1,.9],1)
