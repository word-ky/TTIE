import numpy as np
from research_log.T069A.core import score,threshold,diagnose

def test_last_transition_ev_gamma_only():
    states=np.zeros((3,1,3,2,2));pre=np.zeros((2,1,3,2,2))
    states[1,0,:2]=1;states[2,0,:2]=2;pre[1,0,:2]=3;pre[0,0,:2]=100
    pre[:,0,2]=1e6;states[:,0,2]=-1e6
    r=score(states,pre,0,2);assert r['R_proj']==.5 and np.array(r['s_prev']).shape==(1,2,2,2)
    assert np.all(np.array(r['p'])==3) and np.all(np.array(r['s_end'])==2)

def test_no_transition_zero_proposal_and_no_clip():
    states=np.zeros((2,1,3,2,2));pre=np.zeros((1,1,3,2,2))
    assert score(states,pre,0,0)['reason']=='no_transition'
    assert score(states,pre,0,1)['reason']=='zero_proposal'
    pre[0,0,0,0,0]=states[1,0,0,0,0]=1
    assert score(states,pre,0,1)['R_proj']==0

def test_nearest_rank_strict_threshold_and_verdict():
    t=threshold([dict(R_proj=k/100) for k in range(100)]);assert t==.98 and not(t>t)
    rows=[dict(unsafe=True,above_T99_proj=True)]+[dict(unsafe=False,above_T99_proj=True)]*5
    assert diagnose(rows)['classification']=='PROJECTION_PRESSURE_SIGNAL_PRESENT'
    assert diagnose(rows+[dict(unsafe=False,above_T99_proj=True)])['classification']=='PROJECTION_PRESSURE_SIGNAL_ABSENT'
