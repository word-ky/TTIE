"""T011 offline conjunction, same ten predeclared thresholds as T010 Stage B."""
from .residual_metrics import stage_b,PRIMARY,STRESS


def summarize(rows):
    report=stage_b(rows,primary='region2_ttt_projected',global_method='global_ttt_projected',
                   discrete='region2_discrete_projected',bilinear='bilinear2_ttt_projected')
    report.pop('stage');report['task']='T011'
    report['one_step_diagnostic']={condition:dict(
        one_step_mse=g['region2_ttt_projected_1step']['mse']['mean'],
        full_mse=g['region2_ttt_projected']['mse']['mean'],
        one_step_better=g['region2_ttt_projected_1step']['mse']['mean']<g['region2_ttt_projected']['mse']['mean'])
        for condition,g in report['groups'].items()}
    return report


def markdown(report):
    lines=['# T011 projected spatial TTT','',f"Qualified: {report['qualified']}; failed: {report['failed']}",'',
           '| Criterion | Value | Pass |','|---|---:|---|']
    keys=('clean_mean','clean_p95','dark_ratio','bright_ratio','global_ratio','direct_ratio','discrete_ratio',
          'dark_region_ratio','bright_region_ratio','quadrant_bilinear_ratio')
    for (clause,passed),key in zip(report['criteria'].items(),keys):
        lines.append(f"| {clause} | {report['values'][key]:.9f} | {passed} |")
    lines+=['','| Condition | Method | Mean MSE | P95 MSE | Mean PSNR (finite) | Mean updates |','|---|---|---:|---:|---:|---:|']
    for condition,methods in report['groups'].items():
        for name,g in methods.items():
            values=[g['mse']['mean'],g['mse']['p95'],g['psnr_db']['mean'],g['steps']['mean']]
            lines.append('| '+condition+' | '+name+' | '+' | '.join('null' if v is None else f'{v:.8f}' for v in values)+' |')
    lines+=['','One-step is diagnostic only; offset_left_right_40 is report-only and excluded from qualification.',
            'Perfect PSNR is infinite (null in JSON); no automatic method promotion, tuning, or future task.']
    return '\n'.join(lines)+'\n'
