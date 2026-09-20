"""Exact frozen-probability prefix events, with no quality inputs."""
def events(rows):
    out=[]
    for i in range(100):
        image=rows[i*28:(i+1)*28];base=image[0]['base_step'];prob=[r['p_safe'] for r in image[:base+1]]
        first=next((k for k,p in enumerate(prob) if p>=.5),None)
        unsafe=[k for k,p in enumerate(prob) if first is not None and k>first and p<.5]
        runs=[]
        for k in unsafe:
            if not runs or k!=runs[-1][-1]+1:runs.append([k])
            else:runs[-1].append(k)
        out.append(dict(index=i,base_step=base,prefix_probabilities=prob,first_safe=first,post_safe_unsafe_steps=unsafe,first_post_safe_unsafe=unsafe[0] if unsafe else None,last_post_safe_unsafe=unsafe[-1] if unsafe else None,post_safe_unsafe_count=len(unsafe),unsafe_run_count=len(runs),max_unsafe_run_length=max(map(len,runs),default=0),base_pred_safe=prob[-1]>=.5,reentry=prob[-1]>=.5 and bool(unsafe)))
    return out


def diagnose(event_table,labels):
    counts={'safe_without_reentry':0,'safe_with_reentry':0,'unsafe_without_reentry':0,'unsafe_with_reentry':0};tails=[];categories={k:0 for k in ['before_first_safe','during_post_safe_unsafe','after_safe_reentry','initial_safe_without_prior_warning','outside_prefix','no_first_safe']};state_rows=[]
    for e in event_table:
        q=labels[e['index']*28+e['base_step']];counts[('safe' if q['safe'] else 'unsafe')+('_with_reentry' if e['reentry'] else '_without_reentry')]+=1
        if not q['safe'] or e['index'] in [16,86]:tails.append(dict(e,base_safe=q['safe'],quality_margin=q['quality_margin'],steps_since_last_warning=e['base_step']-e['last_post_safe_unsafe'] if e['last_post_safe_unsafe'] is not None else None))
        for row in labels[e['index']*28:(e['index']+1)*28]:
            if row['safe']:continue
            k=row['step'];f=e['first_safe'];warnings=e['post_safe_unsafe_steps']
            if k>e['base_step']:category='outside_prefix'
            elif f is None:category='no_first_safe'
            elif k<f:category='before_first_safe'
            elif k in warnings:category='during_post_safe_unsafe'
            elif any(j<k for j in warnings):category='after_safe_reentry'
            else:category='initial_safe_without_prior_warning'
            categories[category]+=1;state_rows.append(dict(index=e['index'],step=k,category=category))
    unsafe_bases=[r for r in tails if not r['base_safe']];assert len(unsafe_bases)==2 and [r['index'] for r in unsafe_bases]==[16,86]
    n=sum(r['reentry'] for r in unsafe_bases)
    return dict(classification=['PREFIX_REENTRY_SIGNAL_ABSENT','PREFIX_REENTRY_SIGNAL_PARTIAL','PREFIX_REENTRY_SIGNAL_PRESENT'][n],base_contingency=counts,unsafe_state_categories=categories,unsafe_state_count=sum(categories.values()),unsafe_state_rows=state_rows,tail_rows=tails)
