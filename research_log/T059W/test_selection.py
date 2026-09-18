import pytest
from research_log.T059W.select import select
def row(bank,norm,active=1):return dict(image_id=1,bank_index=bank,state_index=0,index=bank,position=bank,norm=norm,active_regions=active)
def test_maximum_tie():
    assert select([row(2,3),row(1,3),row(0,2)],[1])[0]['bank_index']==1
def test_inactive_coverage():
    with pytest.raises(AssertionError,match='coverage blocked'):select([row(1,0,0)],[1])
def test_missing_image():
    with pytest.raises(AssertionError,match='coverage blocked'):select([row(1,2)],[1,2])
