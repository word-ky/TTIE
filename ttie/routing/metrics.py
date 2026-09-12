"""T015 reference-only ten-clause audit; offset is a primary condition."""
from collections import Counter
from ..residual_metrics import aggregate,PRIMARY as CONDITIONS,STRESS
from ..restoration_metrics import stats
from .core import BASES,PRIMARY,ORACLE

SPATIAL=('left_right','quadrants',STRESS)


def summarize_trajectories(entries):
    # Port of T014's report reducer with the three T015 bases; frozen donor unchanged.
    result={}
    for method in BASES:
        result[method]={}
        for condition in ('all',*dict.fromkeys(e['condition'] for e in entries)):
            ds=[e['trajectory_diagnostics'][method] for e in entries if condition=='all' or e['condition']==condition]
            sums={k:sum(d[k] for d in ds) for k in ('updates','projected_updates','final_boundary_coordinates','final_coordinates','movable_boundary_coordinates','movable_coordinates')}
            result[method][condition]=dict(count=len(ds),selected_steps=dict(sorted(Counter(d['selected_step'] for d in ds).items())),
                no_active=sum(d['no_active'] for d in ds),**sums,
                projected_update_fraction=sums['projected_updates']/sums['updates'] if sums['updates'] else 0.,
                final_boundary_fraction=sums['final_boundary_coordinates']/sums['final_coordinates'],
                movable_boundary_fraction=sums['movable_boundary_coordinates']/sums['movable_coordinates'] if sums['movable_coordinates'] else 0.)
    return result


def ratio(a,b):
    return a/b if b else None


def summarize(rows):
    groups=aggregate(rows);spatial=[r for r in rows if r['condition'] in SPATIAL]
    groups['spatial_pool']={m:{k:stats([r[k] for r in spatial if r['method']==m])
        for k in groups['heterogeneous'][m]} for m in groups['heterogeneous']}
    mse=lambda c,m:groups[c][m]['mse']['mean']
    best=min(BASES,key=lambda m:mse('spatial_pool',m));p=PRIMARY
    specs=[('clean_mean',mse('clean',p),.003),('clean_p95',groups['clean'][p]['mse']['p95'],.005),
        ('homogeneous_dark',mse('homogeneous_dark',p),.60*mse('homogeneous_dark','identity')),
        ('homogeneous_bright',mse('homogeneous_bright',p),.60*mse('homogeneous_bright','identity')),
        ('beyond_best_fixed',mse('spatial_pool',p),.97*mse('spatial_pool',best)),
        ('beyond_discrete',mse('spatial_pool',p),.95*mse('spatial_pool','region2_discrete_projected')),
        ('beyond_fixed16',mse('spatial_pool',p),.95*mse('spatial_pool','fixed_step_source')),
        ('offset_noninferiority',mse(STRESS,p),1.01*mse(STRESS,BASES[1])),
        ('aligned_noninferiority',mse('heterogeneous',p),1.01*mse('heterogeneous',BASES[2])),
        ('oracle_regret',mse('spatial_pool',p),1.05*mse('spatial_pool',ORACLE))]
    criteria={name:observed<=bound for name,observed,bound in specs}
    values=dict(clean_mean=mse('clean',p),clean_p95=groups['clean'][p]['mse']['p95'],
        dark_ratio=ratio(mse('homogeneous_dark',p),mse('homogeneous_dark','identity')),
        bright_ratio=ratio(mse('homogeneous_bright',p),mse('homogeneous_bright','identity')),
        best_fixed_ratio=ratio(mse('spatial_pool',p),mse('spatial_pool',best)),
        discrete_ratio=ratio(mse('spatial_pool',p),mse('spatial_pool','region2_discrete_projected')),
        fixed16_ratio=ratio(mse('spatial_pool',p),mse('spatial_pool','fixed_step_source')),
        offset_bilinear_ratio=ratio(mse(STRESS,p),mse(STRESS,BASES[1])),
        aligned_region2_ratio=ratio(mse('heterogeneous',p),mse('heterogeneous',BASES[2])),
        oracle_regret_ratio=ratio(mse('spatial_pool',p),mse('spatial_pool',ORACLE)))
    return dict(task='T015',groups=groups,spatial_pool=list(SPATIAL),offset_is_primary=True,
        best_fixed_basis_spatial=best,best_fixed_basis_evaluation_only=True,values=values,criteria=criteria,
        clauses=[dict(name=n,observed=a,upper_bound=b,passed=a<=b) for n,a,b in specs],
        qualified=all(criteria.values()),failed=[k for k,v in criteria.items() if not v],
        oracle=dict(spatial_over_best_fixed=ratio(mse('spatial_pool',ORACLE),mse('spatial_pool',best)),
            beats_best_fixed_3pct=mse('spatial_pool',ORACLE)<=.97*mse('spatial_pool',best),
            offset_over_bilinear=ratio(mse(STRESS,ORACLE),mse(STRESS,BASES[1]))))


def routing_diagnostics(entries):
    groups={}
    for condition in ('all',*CONDITIONS,STRESS,'spatial_pool','heterogeneous'):
        selected=[e for e in entries if condition=='all' or e['condition']==condition or
            (condition=='spatial_pool' and e['condition'] in SPATIAL) or
            (condition=='heterogeneous' and e['condition'] in SPATIAL[:2])]
        if not selected:continue
        regret={}
        for basis in BASES:
            records=[e['oracle'] for e in selected if e['routing']['selected_basis']==basis]
            a=[r['routed_mse'] for r in records];b=[r['oracle_mse'] for r in records]
            regret[basis]=dict(count=len(records),routed_mse=stats(a),oracle_mse=stats(b),
                mse_regret=stats([r['mse_regret'] for r in records]),ratio_of_mean_mse=ratio(sum(a),sum(b)),
                zero_oracle_cases=sum(v==0 for v in b),positive_regret_zero_oracle_cases=sum(x>0 and y==0 for x,y in zip(a,b)))
        margins=[dict(image_id=e['image_id'],condition=e['condition'],**e['routing']) for e in selected]
        groups[condition]=dict(count=len(selected),basis_counts={m:sum(e['routing']['selected_basis']==m for e in selected) for m in BASES},
            oracle_basis_counts={m:sum(e['oracle']['selected_basis']==m for e in selected) for m in BASES},
            disagreement_count=sum(e['oracle']['disagreement'] for e in selected),
            disagreement_rate=sum(e['oracle']['disagreement'] for e in selected)/len(selected),
            energy_margin=stats([e['routing']['winner_runner_up_margin'] for e in selected]),full_energy_margins=margins,
            regret_by_routed_basis=regret,zero_denominator_ratio='null; inspect absolute regret and zero-oracle counts')
    return groups


def markdown(report):
    lines=['# T015 frozen cross-basis routing','',f"Qualified: {report['qualified']}; failed: {report['failed']}",'',
        '| Clause | Observed MSE | Upper bound MSE | Pass |','|---|---:|---:|---|']
    for r in report['clauses']:lines.append(f"| {r['name']} | {r['observed']:.12g} | {r['upper_bound']:.12g} | {r['passed']} |")
    lines+=['','Best fixed spatial basis (reference-only): '+report['best_fixed_basis_spatial'],'',str(report['oracle']),'',
        '| Group | Method | Mean MSE | P95 MSE | Mean finite PSNR |','|---|---|---:|---:|---:|']
    for c,methods in report['groups'].items():
        for m,d in methods.items():
            vals=[d[k][s] for k,s in [('mse','mean'),('mse','p95'),('psnr_db','mean')]]
            lines.append('| '+c+' | '+m+' | '+' | '.join('null' if x is None else f'{x:.12g}' for x in vals)+' |')
    lines+=['','All selection counts, full per-input energy margins, oracle disagreements and selected-basis regret are in routing_diagnostics.json. Offset is primary. No new task or retuning.']
    return '\n'.join(lines)+'\n'
