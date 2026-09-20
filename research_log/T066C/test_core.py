from research_log.T066C.core import events
from research_log.T066C.verify import independent_events

def table(p,base):return [dict(index=i,step=k,base_step=base,p_safe=p[k] if k<len(p) else 1.) for i in range(100) for k in range(28)]
def test_first_safe_and_runs():
    r=table([.1,.5,.1,.2,.7,.1,.8],6);a=events(r);assert a==independent_events(r);e=a[0]
    assert e['first_safe']==1 and e['post_safe_unsafe_steps']==[2,3,5] and e['unsafe_run_count']==2 and e['max_unsafe_run_length']==2 and e['reentry']
def test_initial_unsafe_not_reentry():
    r=table([.1,.1,.5,1.],3);assert events(r)==independent_events(r);assert not events(r)[0]['reentry']
def test_no_safe_or_unsafe_base():
    for p in [[.1,.1,.1],[.5,.1,.1]]:
        r=table(p,2);assert events(r)==independent_events(r);assert not events(r)[0]['reentry']
def test_outside_prefix_warning_ignored():
    r=table([.5,.5,.1,.8],1);assert not events(r)[0]['reentry'];assert events(r)==independent_events(r)
