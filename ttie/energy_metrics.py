"""T013 reference-only restoration qualification and gradient diagnostics."""
import statistics
import torch
from .semantic_ttt import Region2
from .residual_metrics import aggregate,stage_b,PRIMARY,STRESS

PRIMARY_METHOD='region2_ttt_energy'
ORACLE='oracle_best_energy_checkpoint'


def alignment(image,clean,trajectory):
    active=torch.tensor(trajectory['gate']['active'],device=image.device)
    model=Region2(active).to(image)
    with torch.no_grad():model.raw.copy_(trajectory['states'][0].to(image))
    reference=((model(image)-clean.to(image)).square().mean()+1e-6).log()
    grad,=torch.autograd.grad(reference,model.raw)
    learned=torch.tensor(trajectory['diagnostics']['gradient_vectors'][0],dtype=torch.float64).flatten()
    truth=grad.detach().cpu().double().flatten();denom=learned.norm()*truth.norm()
    cosine=float((learned@truth)/denom) if float(denom)>0 else 0.
    return dict(cosine=cosine,positive=cosine>0,energy_gradient=learned.tolist(),reference_gradient=truth.tolist(),
                energy_gradient_norm=float(learned.norm()),reference_gradient_norm=float(truth.norm()),
                reference_only=True,coordinates='Region2 raw EV/gamma at identity',zero_norm_rule='cosine 0, counts nonpositive')


def oracle_diagnostics(groups,primary=PRIMARY_METHOD,oracle=ORACLE):
    h=groups['heterogeneous'];mse=lambda m:h[m]['mse']['mean']
    return dict(oracle_regret_ratio=mse(primary)/mse(oracle),
        oracle_over_discrete=mse(oracle)/mse('region2_discrete_projected'),
        oracle_over_fixed16=mse(oracle)/mse('fixed_step_source'),
        oracle_beats_discrete_5pct=mse(oracle)<=.95*mse('region2_discrete_projected'),
        oracle_beats_fixed16_5pct=mse(oracle)<=.95*mse('fixed_step_source'))


def stage_a(rows,alignments,*,primary=PRIMARY_METHOD,oracle=ORACLE):
    groups=aggregate(rows);p=primary;mse=lambda c,m:groups[c][m]['mse']['mean']
    cosines=[a['cosine'] for a in alignments]
    values=dict(positive_cosine_fraction=sum(v>0 for v in cosines)/len(cosines) if cosines else 0.,
        median_cosine=statistics.median(cosines) if cosines else 0.,
        clean_p95=groups['clean'][p]['mse']['p95'],dark_ratio=mse('homogeneous_dark',p)/mse('homogeneous_dark','identity'),
        bright_ratio=mse('homogeneous_bright',p)/mse('homogeneous_bright','identity'),
        discrete_ratio=mse('heterogeneous',p)/mse('heterogeneous','region2_discrete_projected'),
        fixed_step_ratio=mse('heterogeneous',p)/mse('heterogeneous','fixed_step_source'))
    criteria=dict(gradient_positive=values['positive_cosine_fraction']>=.8,gradient_median=values['median_cosine']>=.5,
        clean_p95=values['clean_p95']<=.005,homogeneous_dark=values['dark_ratio']<=.65,homogeneous_bright=values['bright_ratio']<=.65,
        beyond_discrete=values['discrete_ratio']<=.95,beyond_fixed_step=values['fixed_step_ratio']<=.95)
    return dict(stage='A',groups=groups,values=values,criteria=criteria,passes=all(criteria.values()),
        failed=[k for k,v in criteria.items() if not v],alignment_count=len(cosines),oracle=oracle_diagnostics(groups,primary,oracle))


def stage_b_energy(rows,*,primary=PRIMARY_METHOD,oracle=ORACLE,global_method='global_ttt_energy',bilinear='bilinear2_ttt_energy'):
    report=stage_b(rows,primary=primary,global_method=global_method,bilinear=bilinear,discrete='region2_discrete_projected')
    h=report['groups']['heterogeneous'];ratio=h[primary]['mse']['mean']/h['fixed_step_source']['mse']['mean']
    report['values']['fixed_step_ratio']=ratio;report['criteria']['beyond_fixed_step']=ratio<=.95
    report['qualified']=all(report['criteria'].values());report['failed']=[k for k,v in report['criteria'].items() if not v]
    report['oracle']=oracle_diagnostics(report['groups'],primary,oracle);return report


def markdown(report):
    lines=[f"# T013 Stage {report['stage']}",'',f"Pass: {report.get('passes',report.get('qualified'))}; failed: {report['failed']}",'',
           '| Value | Observed |','|---|---:|']
    for k,v in report['values'].items():lines.append(f'| {k} | {v:.12g} |')
    lines+=['','Oracle diagnostics (reference-only): '+str(report['oracle']),'',
        '| Condition | Method | Mean MSE | P95 MSE | Mean finite PSNR |','|---|---|---:|---:|---:|']
    for c,methods in report['groups'].items():
        for m,g in methods.items():
            values=(g['mse']['mean'],g['mse']['p95'],g['psnr_db']['mean'])
            lines.append('| '+c+' | '+m+' | '+' | '.join('null' if v is None else f'{v:.10g}' for v in values)+' |')
    return '\n'.join(lines)+'\n'
