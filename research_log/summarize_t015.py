"""Readable post-run T015 evidence, using verified immutable metrics only."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('audit',type=Path);a=p.parse_args()
read=lambda name:json.loads((a.audit/name).read_text())
s=read('summary.json');d=read('routing_diagnostics.json');remote=read('output_verification.json');local=read('local_verification.json')
bases=['global_ttt_energy_sobolev','bilinear2_ttt_energy_sobolev','region2_ttt_energy_sobolev']
lines=['# T015 verified frozen cross-basis routing report','',
    f"Qualification: {sum(s['criteria'].values())}/10 PASS; qualified={s['qualified']}; failed={s['failed']}.",'',
    '40 fresh images x 6 primary conditions. spatial_pool includes left_right, quadrants and offset_left_right_40 (120 inputs). Aligned heterogeneous contains only left_right and quadrants (80 inputs). No source fitting or score calibration.', '',
    '| Clause | Observed MSE | Required upper MSE | Pass |','|---|---:|---:|---|']
for r in s['clauses']:lines.append(f"| {r['name']} | {r['observed']:.16g} | {r['upper_bound']:.16g} | {r['passed']} |")
lines+=['','| Ratio/value | Observed |','|---|---:|']
for k,v in s['values'].items():lines.append(f'| {k} | {v} |')
lines+=['','Evaluation-only best fixed spatial basis: '+s['best_fixed_basis_spatial'], '',
    'Oracle diagnostics: '+str(s['oracle']), '',
    'The reference oracle selects among the three already-selected basis outputs. It does not select other checkpoints or change trajectories.', '',
    '## Full method MSE table', '', '| Method | '+' | '.join(s['groups'])+' |', '|---|'+'---:|'*len(s['groups'])]
for method in s['groups']['clean']:
    lines.append('| '+method+' | '+' | '.join(f"{g[method]['mse']['mean']:.12g}" for g in s['groups'].values())+' |')
lines+=['','## Basis counts and oracle disagreement','',
    'Counts below are ordered global / bilinear2 / region2. Exact reference-MSE ties use the same order; count disagreement can include equal-output ties, so absolute MSE regret is also reported.', '',
    '| Group | N | Routed counts | Oracle counts | Disagreements | Disagreement fraction |', '|---|---:|---|---|---:|---:|']
for c,g in d.items():
    lines.append(f"| {c} | {g['count']} | {[g['basis_counts'][b] for b in bases]} | {[g['oracle_basis_counts'][b] for b in bases]} | {g['disagreement_count']} | {g['disagreement_rate']:.12g} |")
lines+=['','## Winner-runner-up energy margins', '',
    'All raw scores, selected bases and per-input margins are preserved in routing_diagnostics.json. These are unadjusted frozen energy scores; no offsets or normalization are applied.', '',
    '| Group | N | Exact zero | Minimum | P10 | P25 | Median | P75 | P90 | P95 | Maximum | Mean |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
margin_distributions={}
for c,g in d.items():
    values=np.asarray([r['winner_runner_up_margin'] for r in g['full_energy_margins']],dtype=np.float64)
    qs=np.quantile(values,[0,.1,.25,.5,.75,.9,.95,1])
    row=dict(count=len(values),exact_zero=int((values==0).sum()),mean=float(values.mean()),
        quantiles=dict(zip(('min','p10','p25','p50','p75','p90','p95','max'),qs.tolist())))
    margin_distributions[c]=row
    lines.append('| '+c+f" | {row['count']} | {row['exact_zero']} | "+' | '.join(f'{v:.12g}' for v in [*qs,row['mean']])+' |')
lines+=['','## Regret conditioned on the routed basis', '',
    'The MSE ratio is a ratio of group means, not the mean of per-input ratios. A zero oracle denominator remains null, with absolute regret and zero-oracle counts exposed.', '',
    '| Group | Routed basis | N | Routed MSE | Oracle MSE | Mean absolute excess MSE | Ratio of means | Zero oracle | Positive excess with zero oracle |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for c,g in d.items():
    for b,r in g['regret_by_routed_basis'].items():
        vals=[r['routed_mse']['mean'],r['oracle_mse']['mean'],r['mse_regret']['mean'],r['ratio_of_mean_mse']]
        lines.append('| '+c+' | '+b+f" | {r['count']} | "+' | '.join('null' if v is None else f'{v:.12g}' for v in vals)+
            f" | {r['zero_oracle_cases']} | {r['positive_regret_zero_oracle_cases']} |")
lines+=['','## Frozen per-basis trajectory diagnostics', '',
    '| Basis | Group | Selected-step counts | No-active | Projected / updates | Movable boundary / coordinates |',
    '|---|---|---|---:|---|---|']
for b,groups in s['trajectory_distributions'].items():
    for c,r in groups.items():
        lines.append(f"| {b} | {c} | {r['selected_steps']} | {r['no_active']} | {r['projected_updates']}/{r['updates']} ({r['projected_update_fraction']:.12g}) | {r['movable_boundary_coordinates']}/{r['movable_coordinates']} ({r['movable_boundary_fraction']:.12g}) |")
lines+=['','## Verification receipts','','```json',json.dumps(dict(remote=remote,local=local),indent=2),'```','']
(a.audit/'final_distributions.md').write_text('\n'.join(lines),encoding='utf-8')
(a.audit/'margin_distributions.json').write_text(json.dumps(margin_distributions,indent=2)+'\n')
print(json.dumps(dict(qualified=s['qualified'],passed=sum(s['criteria'].values()),oracle=s['oracle'],
    spatial_margins=margin_distributions['spatial_pool'],remote=remote,local=local),indent=2))
