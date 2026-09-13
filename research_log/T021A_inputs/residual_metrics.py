"""T010 offline source selection and held-out qualification, never in updates."""
from .restoration_metrics import stats
from .residual_ttt import RHO_GRID

PRIMARY=('clean','homogeneous_dark','homogeneous_bright','left_right','quadrants')
STRESS='offset_left_right_40'
PAIRS=tuple((d,b) for d in RHO_GRID for b in RHO_GRID)


def candidate_name(d,b):return f'rho_d{round(d*100):03d}_b{round(b*100):03d}'


def aggregate(rows):
    methods=list(dict.fromkeys(r['method'] for r in rows));groups={}
    for condition in (*PRIMARY,STRESS,'heterogeneous'):
        subset=[r for r in rows if r['condition'] in ('left_right','quadrants')] if condition=='heterogeneous' else [r for r in rows if r['condition']==condition]
        if not subset:continue
        groups[condition]={}
        for method in methods:
            selected=[r for r in subset if r['method']==method]
            keys=('mse','psnr_db','clean_drift_mse','dark_region_mse','bright_region_mse','loss_before','loss_after','steps','recovery_ratio')
            groups[condition][method]={k:stats([r[k] for r in selected]) for k in keys}
    return groups


def stage_a(rows):
    groups=aggregate(rows);candidates=[]
    mse=lambda c,m:groups[c][m]['mse']['mean']
    for d,b in PAIRS:
        name=candidate_name(d,b);clean=groups['clean'][name]['mse']
        values=dict(clean_mean=clean['mean'],clean_p95=clean['p95'],
                    dark_ratio=mse('homogeneous_dark',name)/mse('homogeneous_dark','identity'),
                    bright_ratio=mse('homogeneous_bright',name)/mse('homogeneous_bright','identity'),
                    heterogeneous_mse=mse('heterogeneous',name),
                    heterogeneous_direct_ratio=mse('heterogeneous',name)/mse('heterogeneous','region2_direct'))
        criteria=dict(clean_mean=values['clean_mean']<=.003,clean_p95=values['clean_p95']<=.005,
                      homogeneous_dark=values['dark_ratio']<=.7,homogeneous_bright=values['bright_ratio']<=.7,
                      beyond_direct=values['heterogeneous_direct_ratio']<=.95)
        candidates.append(dict(method=name,rho_dark=d,rho_bright=b,**values,criteria=criteria,feasible=all(criteria.values())))
    feasible=[r for r in candidates if r['feasible']];selected=None
    if feasible:
        best=min(r['heterogeneous_mse'] for r in feasible)
        tied=[r for r in feasible if r['heterogeneous_mse']<=best+1e-6]
        selected=max(tied,key=lambda r:((r['rho_dark']+r['rho_bright'])/2,r['rho_bright'],r['rho_dark']))
    return dict(stage='A',development_only=True,groups=groups,candidates=candidates,selected=selected,
                feasible_count=len(feasible),passes=bool(feasible),selection='min heterogeneous MSE; within 1e-6 prefer mean rho, bright rho, dark rho descending')


def stage_b(rows,*,primary='region2_ttt_rho',global_method='global_ttt_rho',
            discrete='region2_discrete_rho',bilinear='bilinear2_ttt_rho'):
    groups=aggregate(rows);clean=groups['clean'][primary]['mse']
    mse=lambda c,m:groups[c][m]['mse']['mean']
    ratio=lambda baseline:mse('heterogeneous',primary)/mse('heterogeneous',baseline)
    h=groups['heterogeneous']
    values=dict(clean_mean=clean['mean'],clean_p95=clean['p95'],
        dark_ratio=mse('homogeneous_dark',primary)/mse('homogeneous_dark','identity'),
        bright_ratio=mse('homogeneous_bright',primary)/mse('homogeneous_bright','identity'),
        global_ratio=ratio(global_method),direct_ratio=ratio('region2_direct'),discrete_ratio=ratio(discrete),
        dark_region_ratio=h[primary]['dark_region_mse']['mean']/h['identity']['dark_region_mse']['mean'],
        bright_region_ratio=h[primary]['bright_region_mse']['mean']/h['identity']['bright_region_mse']['mean'],
        quadrant_bilinear_ratio=mse('quadrants',primary)/mse('quadrants',bilinear))
    criteria=dict(clean_mean=values['clean_mean']<=.003,clean_p95=values['clean_p95']<=.005,
        homogeneous_dark=values['dark_ratio']<=.6,homogeneous_bright=values['bright_ratio']<=.6,
        spatial_value=values['global_ratio']<=.85,beyond_direct=values['direct_ratio']<=.95,
        beyond_discrete=values['discrete_ratio']<=.95,dark_region_safety=values['dark_region_ratio']<=1.05,
        bright_region_safety=values['bright_region_ratio']<=1.05,renderer_support=values['quadrant_bilinear_ratio']<=.9)
    return dict(stage='B',groups=groups,values=values,criteria=criteria,qualified=all(criteria.values()),
                failed=[k for k,v in criteria.items() if not v],stress_report_only=STRESS)


def markdown(report):
    lines=[f"# T010 Stage {report['stage']}",'']
    if report['stage']=='A':
        lines += [f"Passes: {report['passes']}; feasible candidates: {report['feasible_count']}; selected: {report['selected']}",'',
          '| Dark rho | Bright rho | Clean mean | Clean p95 | Dark ratio | Bright ratio | Hetero MSE | Direct ratio | Feasible |',
          '|---|---|---|---|---|---|---|---|---|']
        for r in report['candidates']:
            lines.append('| '+' | '.join(str(r[k]) for k in ('rho_dark','rho_bright','clean_mean','clean_p95','dark_ratio','bright_ratio','heterogeneous_mse','heterogeneous_direct_ratio','feasible'))+' |')
    else:lines += [f"Qualified: {report['qualified']}; failed: {report['failed']}",'',str(report['values'])]
    lines+=['','Full per-condition/per-method metrics are in summary.json. No automatic next task is authorized.']
    return '\n'.join(lines)+'\n'
