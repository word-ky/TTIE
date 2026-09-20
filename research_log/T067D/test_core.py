from research_log.T067D.core import interval,boundaries
from research_log.T067D.verify import independent_boundary

def test_q_definition_and_selected_flag():
    r=interval([1.,.8,.3,0.],1,3,2);assert r[0]['q_k']==0 and r[-1]['q_k']>1 and r[1]['selected_flag']
def test_boundary_recovery_and_unsafe_start():
    for flags in [[True,True,False,True],[False,True,False,True],[True]*4]:
        rows=[dict(index=i,k=k,k_FS=0,k_rho=3,q_k=k/3,selected_flag=k==2,safe=s,margin=0. if s else -6.) for i in range(100) for k,s in enumerate(flags)]
        b,a=boundaries(rows);assert b[0]==independent_boundary(rows[:4]);assert a['intervals_with_recovery']==(100 if False in flags and flags[-1] else 0)
def test_bins_preserve_overshoot():
    rows=[dict(index=i,k=k,k_FS=0,k_rho=4,q_k=q,selected_flag=k==4,safe=False,margin=-6.) for i in range(100) for k,q in enumerate([0,.5,.75,.875,1.01])]
    b,a=boundaries(rows);assert all(x==100 for x in a['fixed_q_bins'].values())
