"""T012 reference-only source fixed depth selection and frozen qualification."""
from collections import Counter
from .residual_metrics import aggregate,stage_b,PRIMARY,STRESS

FIXED_STEPS=(0,1,2,4,8,16,40)
PRIMARY_METHOD='region2_ttt_learned_stop'


def fixed_name(step):return f'fixed_candidate_{step:02d}'


def choose_fixed_step(rows):
    groups=aggregate(rows);candidates=[]
    for step in FIXED_STEPS:
        name=fixed_name(step)
        candidates.append(dict(step=step,clean_p95=groups['clean'][name]['mse']['p95'],
            heterogeneous_mse=groups['heterogeneous'][name]['mse']['mean']))
    eligible=[r for r in candidates if r['clean_p95']<=.005]
    selected=min(eligible,key=lambda r:(r['heterogeneous_mse'],r['step']))
    return dict(selected_step=selected['step'],candidates=candidates,rule='clean p95<=.005; min hetero MSE; exact ties earliest step')


def selection_histograms(rows):
    return {c:dict(sorted(Counter(str(r['selected_step']) for r in rows if r['condition']==c and r['method']==PRIMARY_METHOD).items(),key=lambda x:int(x[0])))
            for c in (*PRIMARY,STRESS) if any(r['condition']==c for r in rows)}


def stage_a(rows):
    groups=aggregate(rows);p=PRIMARY_METHOD
    mse=lambda c,m:groups[c][m]['mse']['mean']
    values=dict(clean_p95=groups['clean'][p]['mse']['p95'],
        dark_ratio=mse('homogeneous_dark',p)/mse('homogeneous_dark','identity'),
        bright_ratio=mse('homogeneous_bright',p)/mse('homogeneous_bright','identity'),
        discrete_ratio=mse('heterogeneous',p)/mse('heterogeneous','region2_discrete_projected'),
        fixed_step_ratio=mse('heterogeneous',p)/mse('heterogeneous','fixed_step_source'),
        oracle_regret_ratio=mse('heterogeneous',p)/mse('heterogeneous','oracle_best_checkpoint'))
    criteria=dict(clean_p95=values['clean_p95']<=.005,homogeneous_dark=values['dark_ratio']<=.65,
        homogeneous_bright=values['bright_ratio']<=.65,beyond_discrete=values['discrete_ratio']<=.95,
        beyond_fixed_step=values['fixed_step_ratio']<=.95)
    return dict(stage='A',groups=groups,values=values,criteria=criteria,passes=all(criteria.values()),
                failed=[k for k,v in criteria.items() if not v],selected_step_histogram=selection_histograms(rows))


def stage_b_stop(rows):
    report=stage_b(rows,primary=PRIMARY_METHOD,global_method='global_ttt_projected',
                   discrete='region2_discrete_projected',bilinear='bilinear2_ttt_projected')
    g=report['groups']['heterogeneous'];ratio=g[PRIMARY_METHOD]['mse']['mean']/g['fixed_step_source']['mse']['mean']
    report['values']['fixed_step_ratio']=ratio;report['criteria']['beyond_fixed_step']=ratio<=.95
    report['values']['oracle_regret_ratio']=g[PRIMARY_METHOD]['mse']['mean']/g['oracle_best_checkpoint']['mse']['mean']
    report['qualified']=all(report['criteria'].values());report['failed']=[k for k,v in report['criteria'].items() if not v]
    report['selected_step_histogram']=selection_histograms(rows)
    return report


def markdown(report):
    lines=[f"# T012 Stage {report['stage']}",'',f"Pass: {report.get('passes',report.get('qualified'))}; failed: {report['failed']}",'',
           '| Metric / ratio | Value |','|---|---:|']
    for key,value in report['values'].items():lines.append(f'| {key} | {value:.10f} |')
    lines+=['','| Condition | Method | Mean MSE | P95 MSE | Mean finite PSNR |','|---|---|---:|---:|---:|']
    for c,methods in report['groups'].items():
        for m,g in methods.items():
            values=(g['mse']['mean'],g['mse']['p95'],g['psnr_db']['mean'])
            lines.append('| '+c+' | '+m+' | '+' | '.join('null' if v is None else f'{v:.9f}' for v in values)+' |')
    lines+=['','Oracle uses references only after label-free outputs are persisted. Offset stress is report-only. No tuning or automatic next task.']
    return '\n'.join(lines)+'\n'
