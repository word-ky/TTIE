"""Offline T008 metrics. Receives finalized outputs and evaluation truth only."""
import itertools
import math
import statistics
import torch

from .semantic_ttt import METHODS
from .natural import CONDITIONS


def evaluate_outputs(results, clean, *, image_id, condition):
    clean=clean.cpu()
    identity=results['identity']['image'].cpu()
    base=float((identity-clean).square().mean())
    h,w=clean.shape[-2:]
    y,x=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
    mask=x<w//2 if condition=='left_right' else ((x>=w//2).int()+(y>=h//2).int())%2==0
    rows=[]
    for method,result in results.items():
        squared=(result['image'].cpu()-clean).square()
        mse=float(squared.mean());diag=result['diagnostics']
        rows.append(dict(image_id=image_id,condition=condition,method=method,mse=mse,
                         psnr_db=-10*math.log10(mse) if mse else None,
                         recovery_ratio=1-mse/base if condition!='clean' and base else None,
                         clean_drift_mse=mse if condition=='clean' else None,
                         dark_region_mse=float(squared[...,mask].mean()) if condition in ('left_right','quadrants') else None,
                         bright_region_mse=float(squared[...,~mask].mean()) if condition in ('left_right','quadrants') else None,
                         loss_before=diag['loss_before'],loss_after=diag['loss_after'],active_count=diag['active_count'],
                         steps=diag['steps'],stop_reason=diag['stop_reason'],ev_min=diag['final_ranges']['min'][0],
                         ev_max=diag['final_ranges']['max'][0],gamma_min=diag['final_ranges']['min'][1],gamma_max=diag['final_ranges']['max'][1]))
    return rows


def stats(values):
    values=[v for v in values if v is not None]
    if not values:return dict(count=0,mean=None,median=None,p95=None)
    return dict(count=len(values),mean=statistics.mean(values),median=statistics.median(values),
                p95=float(torch.quantile(torch.tensor(values,dtype=torch.float64),.95)))


def summarize_restoration(rows, immutable=True):
    groups={}
    for condition in (*CONDITIONS,'heterogeneous'):
        group=[r for r in rows if r['condition'] in ('left_right','quadrants')] if condition=='heterogeneous' else [r for r in rows if r['condition']==condition]
        groups[condition]={}
        for method in METHODS:
            selected=[r for r in group if r['method']==method]
            keys=('mse','psnr_db','recovery_ratio','clean_drift_mse','dark_region_mse','bright_region_mse','loss_before','loss_after','active_count','steps')
            groups[condition][method]={k:stats([r[k] for r in selected]) for k in keys}
            groups[condition][method]['perfect_psnr_count']=sum(r['mse']==0 for r in selected)
    mse=lambda c,m:groups[c][m]['mse']['mean']
    clean=groups['clean']['spatial2_ttt']['clean_drift_mse']
    hetero=groups['heterogeneous']
    criteria=dict(clean_mean=clean['mean']<=1e-3,clean_p95=clean['p95']<=5e-3,
        dark_usefulness=mse('homogeneous_dark','spatial2_ttt')<=.9*mse('homogeneous_dark','identity'),
        bright_usefulness=mse('homogeneous_bright','spatial2_ttt')<=.9*mse('homogeneous_bright','identity'),
        spatial_vs_global=mse('heterogeneous','spatial2_ttt')<=.85*mse('heterogeneous','global_ttt'),
        heterogeneous_usefulness=mse('heterogeneous','spatial2_ttt')<=.9*mse('heterogeneous','identity'),
        beyond_direct=mse('heterogeneous','spatial2_ttt')<=.95*mse('heterogeneous','spatial2_direct'),
        competitive_discrete=mse('heterogeneous','spatial2_ttt')<=1.05*mse('heterogeneous','spatial2_discrete'),
        dark_region_safety=hetero['spatial2_ttt']['dark_region_mse']['mean']<=1.1*hetero['identity']['dark_region_mse']['mean'],
        bright_region_safety=hetero['spatial2_ttt']['bright_region_mse']['mean']<=1.1*hetero['identity']['bright_region_mse']['mean'],
        leakage_immutability=immutable)
    paired=[]
    cases=sorted({(r['image_id'],r['condition']) for r in rows})
    for image_id,condition in cases:
        by_method={r['method']:r for r in rows if r['image_id']==image_id and r['condition']==condition}
        for a,b in itertools.combinations(METHODS,2):
            paired.append(dict(image_id=image_id,condition=condition,method_a=a,method_b=b,mse_a_minus_b=by_method[a]['mse']-by_method[b]['mse']))
    paired_summary={}
    for condition in (*CONDITIONS,'heterogeneous'):
        items=[r for r in paired if r['condition'] in ('left_right','quadrants')] if condition=='heterogeneous' else [r for r in paired if r['condition']==condition]
        paired_summary[condition]={a+' minus '+b:stats([r['mse_a_minus_b'] for r in items if r['method_a']==a and r['method_b']==b]) for a,b in itertools.combinations(METHODS,2)}
    return dict(groups=groups,criteria=criteria,qualifies_later_detector=all(criteria.values()),
                failed_criteria=[k for k,v in criteria.items() if not v],paired_summary=paired_summary),paired


def markdown_summary(report):
    lines=['# T008 fixed restoration pilot','',f"Qualified: {report['qualifies_later_detector']}; failed: {report['failed_criteria']}",'',
           '| Condition | Method | Mean MSE | Median MSE | P95 MSE | Mean finite PSNR | Mean recovery |', '|---|---|---:|---:|---:|---:|---:|']
    for condition,methods in report['groups'].items():
        for name,g in methods.items():
            vals=(g['mse']['mean'],g['mse']['median'],g['mse']['p95'],g['psnr_db']['mean'],g['recovery_ratio']['mean'])
            lines.append('| '+condition+' | '+name+' | '+' | '.join('null' if v is None else f'{v:.8f}' for v in vals)+' |')
    lines+=['','Perfect reconstruction PSNR is +infinity (null in JSON), with perfect counts saved separately. Smooth gradient is report-only. All per-image pairwise MSE differences are saved, positive A-minus-B means B is better.']
    return '\n'.join(lines)+'\n'
