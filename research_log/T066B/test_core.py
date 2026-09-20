import numpy as np
from research_log.T066B.core import geometry,classification,scores,distribution
from research_log.T066B.verify import independent_geometry,independent_scores,independent_distribution


def test_geometry_excludes_whole_image_and_matches_independent():
    rng=np.random.default_rng(71);bank=rng.normal(size=(84,19));labels=np.arange(84)%3!=0
    got=geometry(bank,bank,labels,np.repeat(np.arange(3),28));want=independent_geometry(bank,bank,labels,True)
    for i,(a,b) in enumerate(zip(got,want)):
        assert a['safe_bank_row']//28!=i//28 and a['unsafe_bank_row']//28!=i//28
        for k in a:np.testing.assert_allclose(a[k],b[k],rtol=0,atol=1e-12)


def test_geometry_tie_and_margin_direction():
    bank=np.array([[0.],[2.],[2.],[5.]]);r=geometry([[2.]],bank,[False,True,True,False])[0]
    assert r['safe_bank_row']==1 and r['unsafe_bank_row']==0 and r['margin']==2.


def test_categories_exact_boundaries():
    assert classification(.49,.75,1)=='TRANSFER_SUPPORT_SHIFT'
    assert classification(.49,.7499,1)=='BOUNDARY_MISMATCH_WITH_UNSAFE_SUPPORT'
    assert classification(.5,.9,1)=='SELECTED_TAIL_SPECIFIC_FAILURE'
    assert classification(.5,.9,0)=='NO_DIAGNOSTIC_FAILURE'


def test_confusion_subset_and_distribution():
    y=[True,False,False,True];p=[.5,.2,.5,.1];mask=np.array([1,1,0,1],dtype=bool)
    assert scores(y,p,mask)==independent_scores(y,p,mask)
    rows=[dict(d_safe=float(i),d_unsafe=float(4-i),margin=float(4-2*i)) for i in range(4)]
    assert distribution(rows,mask)==independent_distribution(rows,mask)
