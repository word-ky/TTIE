from research_log.T067B.core import choices,select
from research_log.T067B.verify import independent_choices
import pytest

def test_endpoints_interval_and_independent():
    values=[1.-k/27 for k in range(28)];p=[.1]*4+[.5]*24;rows=choices(values,p,27)
    assert rows[0]['selected_step']==4 and rows[-1]['selected_step']==27
    assert [(r['lambda_value'],r['selected_step']) for r in rows]==independent_choices(values,p,27)
def test_no_fallback():
    with pytest.raises(AssertionError):choices([1.-k/27 for k in range(28)],[.1]*28,27)
def test_exact_robustness_ties_and_endpoint_negative():
    def row(l,w,m):return dict(lambda_value=l,worst_delta_t026=w,mean_delta_psnr=m,gates={'all':True})
    table=[row(1,-2,4),row(.5,-1,3),row(.25,-1,3),row(.125,-1,2)]
    assert select(table)[0]['lambda_value']==.25
    assert select([table[0]])[1]=='INTERIOR_PROGRESS_DEV_NEGATIVE'
    assert select([])[1]=='INTERIOR_PROGRESS_DEV_NEGATIVE'
