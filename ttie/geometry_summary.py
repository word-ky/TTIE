"""Predeclared development-only diagnosis rules; no held-out qualification."""
import statistics
from .offline_geometry import correlation

CONDITIONS=('clean','homogeneous_dark','homogeneous_bright','left_right','quadrants')
COORDINATES=('ev_only','gamma_only','ev_gamma')


def mean(values):
    values=[v for v in values if v is not None]
    return statistics.mean(values) if values else None


def distribution(values):
    values=[v for v in values if v is not None]
    return dict(n=len(values),mean=mean(values),median=statistics.median(values) if values else None)


def trajectory_summary(records):
    active=[r for r in records if r['active_count']>0 and len(r['trace'])>1]
    first=[];final=[];last_update=[];early=[];ds=[];dm=[]
    for r in active:
        t=r['trace'];m=[p['mse'] for p in t]
        first.append(m[1]<m[0]);final.append(m[-1]<m[0]);last_update.append(m[-1]<m[-2])
        early.append(min(m[:-1])<m[-1])
        ds.append(t[0]['semantic_loss']-t[-1]['semantic_loss']);dm.append(m[-1]-m[0])
    result=dict(episodes=len(records),active_updated_episodes=len(active),step1_improves=mean(first),
      final_improves_vs_identity=mean(final),last_update_improves_vs_previous=mean(last_update),
      strictly_better_earlier_step_fraction=mean(early),loss_decrease_vs_mse_change_pearson=correlation(ds,dm))
    for name,index in (('identity',0),('step1',1),('final',-1)):
        result[name+'_mse']=distribution([r['trace'][min(index,len(r['trace'])-1)]['mse'] for r in records])
    result['offline_oracle_best_mse']=distribution([min(t['mse'] for t in r['trace']) for r in records])
    result['identity_mse_active']=distribution([r['trace'][0]['mse'] for r in active])
    result['step1_mse_active']=distribution([r['trace'][1]['mse'] for r in active])
    result['final_mse_active']=distribution([r['trace'][-1]['mse'] for r in active])
    result['oracle_best_mse_active']=distribution([min(t['mse'] for t in r['trace']) for r in active])
    return result


def alignment_summary(records):
    usable=[r['alignment'] for r in records if r.get('alignment') is not None]
    values=[r['cosine'] for r in usable if r['cosine'] is not None]
    result=dict(cosine=distribution(values),positive_fraction=sum(v>0 for v in values)/len(usable) if usable else None,
                undefined_cosine_count=len(usable)-len(values),active_alignment_count=len(usable),
                dark_cosine=distribution([r.get('dark_cosine') for r in usable]),
                bright_cosine=distribution([r.get('bright_cosine') for r in usable]))
    for coordinate in ('ev','gamma'):
        items=[r[coordinate] for r in usable if coordinate in r]
        result[coordinate]={k:distribution([r[k] for r in items]) for k in ('sign_agreement','semantic_abs_sum','reference_abs_sum')}
    return result


def diagnose(records,surfaces,renderers):
    groups={}
    for condition in (*CONDITIONS,'pooled_all','pooled_nonclean','heterogeneous'):
        select=lambda r:True if condition=='pooled_all' else r['condition']!='clean' if condition=='pooled_nonclean' else r['condition'] in ('left_right','quadrants') if condition=='heterogeneous' else r['condition']==condition
        selected=[r for r in records if select(r)];groups[condition]={}
        for size in (1,2):
            for coordinates in COORDINATES:
                rs=[r for r in selected if r['size']==size and r['coordinates']==coordinates]
                group=dict(alignment=alignment_summary(rs),trajectory=trajectory_summary(rs),
                    final_mse=distribution([r['trace'][-1]['mse'] for r in rs]),
                    semantic_reduction=distribution([r['trace'][0]['semantic_loss']-r['trace'][-1]['semantic_loss'] for r in rs]),
                    dark_mse=distribution([r['trace'][-1].get('dark_mse') for r in rs]),
                    bright_mse=distribution([r['trace'][-1].get('bright_mse') for r in rs]))
                groups[condition][f'{size}_{coordinates}']=group
    rsummary={}
    for condition in ('left_right','quadrants'):
        selected=[r for r in renderers if r['condition']==condition]
        rsummary[condition]={}
        for method in ('direct','oracle'):
            for renderer in ('bilinear2','piecewise2'):
                rs=[r for r in selected if r['method']==method and r['renderer']==renderer]
                rsummary[condition][method+'_'+renderer]={key:distribution([r[key] for r in rs]) for key in ('mse','dark_mse','bright_mse')}
    ssummary={}
    for condition in ('homogeneous_dark','homogeneous_bright'):
        ssummary[condition]={}
        for subset in ('all','active'):
            rs=[r for r in surfaces if r['condition']==condition and (subset=='all' or r['active_count']>0)]
            ssummary[condition][subset]=dict(n=len(rs),spearman=distribution([r['summary']['spearman'] for r in rs]),
                 ev_distance=distribution([r['summary']['ev_distance'] for r in rs]),
                 gamma_distance=distribution([r['summary']['gamma_distance'] for r in rs]),
                 semantic_argmin_worse_than_direct_fraction=mean([r['summary']['semantic_argmin_worse_than_fixed_direct'] for r in rs]),
                 crossing_counts={k:sum(r['summary']['crossing_category']==k for r in rs) for k in ('before','near','beyond','none')})
    pooled=groups['pooled_nonclean'];a=pooled['2_ev_gamma']['alignment'];b=pooled['2_ev_gamma']['trajectory']
    objective_failure=a['positive_fraction']<.6 or a['cosine']['median']<=0
    stopping_failure=not objective_failure and b['step1_improves']>=.7 and b['strictly_better_earlier_step_fraction']>=.3
    gamma={}
    for size in (1,2):
        ev=groups['pooled_all'][f'{size}_ev_only'];both=groups['pooled_all'][f'{size}_ev_gamma']
        mse_ratio=ev['final_mse']['mean']/both['final_mse']['mean']
        retained=ev['semantic_reduction']['mean']/both['semantic_reduction']['mean'] if both['semantic_reduction']['mean']>0 else None
        gamma[str(size)]=dict(mse_ratio=mse_ratio,semantic_reduction_retained=retained,
                              fires=mse_ratio<=.9 and retained is not None and retained>=.9)
    quadrant=rsummary['quadrants']
    renderer_ratios={m:quadrant[m+'_piecewise2']['mse']['mean']/quadrant[m+'_bilinear2']['mse']['mean'] for m in ('direct','oracle')}
    rules=dict(objective_gradient_failure=objective_failure,overcorrection_stopping_failure=stopping_failure,
               gamma_failure=gamma['2']['fires'],renderer_failure=all(v<=.85 for v in renderer_ratios.values()))
    return dict(groups=groups,surfaces=ssummary,renderers=rsummary,rules=rules,fired=[k for k,v in rules.items() if v],
                gamma_rule_details=gamma,renderer_rule_ratios=renderer_ratios,
                primary_pool='spatial2 ev_gamma active nonclean for A/B; all five conditions for C; quadrants for E',development_only=True)


def markdown(report):
    lines=['# T009 development geometry diagnosis','',f"Fired rules: {report['fired']}",'',
      '| Pool | State | Coordinate | Cosine positive | Cosine median | Step1 improves | Earlier better | Final MSE | Oracle-step MSE |',
      '|---|---|---|---:|---:|---:|---:|---:|---:|']
    fmt=lambda x:'null' if x is None else f'{x:.6f}'
    for pool in ('pooled_nonclean','homogeneous_dark','homogeneous_bright','left_right','quadrants','clean'):
        for key,g in report['groups'][pool].items():
            state,coordinate=key.split('_',1);a=g['alignment'];t=g['trajectory']
            values=(a['positive_fraction'],a['cosine']['median'],t['step1_improves'],t['strictly_better_earlier_step_fraction'],g['final_mse']['mean'],t['offline_oracle_best_mse']['mean'])
            lines.append('| '+pool+' | '+state+' | '+coordinate+' | '+' | '.join(map(fmt,values))+' |')
    lines+=['','Development/oracle results only. No T010 or fresh held-out qualification is implied. Full renderer, surface, coordinate and correlation summaries are in summary.json.']
    return '\n'.join(lines)+'\n'
