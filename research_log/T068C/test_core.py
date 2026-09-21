import math
from research_log.T068C.core import score,threshold,diagnose

def test_three_transition_window():
    r=score([10,9,8,7,6,5],1,5,[0,1,2,3,4]);assert r['k0']==2 and r['M_tail']==9 and r['M_total']==10 and r['p_tail']==.75 and r['q_tail']==.9
    assert score([3,2,1],0,2,[0,2,3])['k0']==0

def test_singleton_floors_and_clipping():
    r=score([2],0,0,[0]);assert r['p_tail']==r['q_tail']==r['R']==0 and r['reason']=='singleton'
    r=score([1,0,0,0,0],0,4,[0,0,0,0,0]);assert r['p_tail']==r['q_tail']==r['R']==0
    r=score([1,2,2,2,1],0,4,[0,1,1,1,1]);assert r['D_total']==0 and r['p_tail']==1 and math.isfinite(r['R'])
    r=score([3,0,0,0,1],0,4,[0,1,1,1,1]);assert r['p_tail']==0 and math.isfinite(r['R'])

def test_nearest_rank_strict_threshold_and_diagnosis():
    assert threshold([dict(R=v) for v in reversed(range(100))])==98
    rows=[dict(unsafe=True,above_T99=True)]+[dict(unsafe=False,above_T99=True)]*5
    assert diagnose(rows)['classification']=='TAIL_INEFFICIENCY_SIGNAL_PRESENT'
    assert diagnose(rows+[dict(unsafe=False,above_T99=True)])['classification']=='TAIL_INEFFICIENCY_SIGNAL_ABSENT'
    assert diagnose([dict(unsafe=True,above_T99=False)])['classification']=='TAIL_INEFFICIENCY_SIGNAL_ABSENT'
