from research_log.T068D.core import score,threshold,diagnose

def test_weighted_components_extrema_and_exact_ratio():
    r=score([[99,99,99],[1,.2,.4],[2,.1,.8],[1.5,.3,.2]],1,3)
    assert r['Z']==[[1,2,2],[2,1,4],[1.5,3,1]]
    assert r['zmin']==[1,1,1] and r['zmax']==[2,3,4]
    assert r['a']==[.5,2,0] and r['e']==[1,2,3] and r['R_comp']==2.5/6

def test_singleton_and_zero_excursion():
    assert score([[2,3,4]],0,0)['reason']=='singleton'
    r=score([[2,3,4],[2,3,4]],0,1);assert r['R_comp']==0 and r['reason']=='zero_excursion'
    r=score([[0,0,0],[1e-13,0,0]],0,1);assert r['R_comp']==0 and r['sum_e']<=1e-12

def test_nearest_rank_strict_flags_and_acceptance():
    t=threshold([dict(R_comp=k/100) for k in range(100)]);assert t==.98
    assert not (t>t)
    rows=[dict(unsafe=True,above_T99_comp=True)]+[dict(unsafe=False,above_T99_comp=True)]*5
    assert diagnose(rows)['classification']=='COMPONENT_REGRET_SIGNAL_PRESENT'
    assert diagnose(rows+[dict(unsafe=False,above_T99_comp=True)])['classification']=='COMPONENT_REGRET_SIGNAL_ABSENT'
    assert diagnose([dict(unsafe=True,above_T99_comp=False)])['classification']=='COMPONENT_REGRET_SIGNAL_ABSENT'
