from research_log.T063C.core import progress
RHO=.9857470621423519
def interval(values,fs,base,selected):
    _,_,r=progress(values);den=max(RHO-r[fs],1e-12)
    return [dict(k=k,r_k=float(r[k]),q_k=float((r[k]-r[fs])/den),k_FS=fs,k_rho=base,selected_flag=k==selected) for k in range(fs,base+1)]

def boundaries(rows):
    result=[];bins={'[0,.50)':0,'[.50,.75)':0,'[.75,.875)':0,'[.875,1.0001]':0,'outside_fixed_bins':0};outliers=[]
    for i in range(100):
        states=[r for r in rows if r['index']==i];unsafe=[r for r in states if not r['safe']];first=unsafe[0] if unsafe else None;prefix=[]
        for r in states:
            if not r['safe']:break
            prefix.append(r)
        end=prefix[-1] if prefix else None;selected=next(r for r in states if r['selected_flag']);after=[r for r in unsafe if r['k']>r['k_FS']]
        result.append(dict(index=i,k_FS=states[0]['k_FS'],k_rho=states[0]['k_rho'],selected_step=selected['k'],any_interval_unsafe=bool(unsafe),first_unsafe_step=first['k'] if first else None,first_unsafe_strictly_after_FS=after[0]['k'] if after else None,first_safe_state_is_reference_safe=bool(states[0]['safe']),contiguous_safe_prefix_end=end['k'] if end else None,safety_recovers=any(r['safe'] and r['k']>first['k'] for r in states) if first else False,q_first_unsafe=first['q_k'] if first else None,q_safe_prefix_end=end['q_k'] if end else None,selected_boundary_relation=('before' if selected['k']<first['k'] else 'on' if selected['k']==first['k'] else 'after') if first else 'no_unsafe_boundary',selected_margin=selected['margin'],selected_safe=selected['safe'],unsafe_states=len(unsafe)))
        for r in unsafe:
            q=r['q_k'];key='[0,.50)' if 0<=q<.5 else '[.50,.75)' if .5<=q<.75 else '[.75,.875)' if .75<=q<.875 else '[.875,1.0001]' if .875<=q<=1.0001 else 'outside_fixed_bins';bins[key]+=1
            if key=='outside_fixed_bins':outliers.append(dict(index=i,k=r['k'],q_k=q))
    aggregate=dict(images=100,interval_states=len(rows),intervals_with_unsafe=sum(r['any_interval_unsafe'] for r in result),intervals_with_recovery=sum(r['safety_recovers'] for r in result),selected_unsafe=sum(not r['selected_safe'] for r in result),unsafe_interval_states=sum(r['unsafe_states'] for r in result),fixed_q_bins=bins,outside_fixed_bin_rows=outliers)
    return result,aggregate
