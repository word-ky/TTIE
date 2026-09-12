"""Final reporting only: distinct source-fit and calibration evidence from saved data."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
from ttie.sobolev_metrics import distribution

p=argparse.ArgumentParser();p.add_argument('audit',type=Path);args=p.parse_args();root=args.audit
read=lambda name:json.loads((root/name).read_text())
train=read('training.json');entries=read('training_manifest.json');summary=read('summary.json')
source={}
for head,stats in train['final_train_statistics'].items():
    cos=stats['cosines'];by_condition={};cursor=0
    for entry in entries:
        n=entry['direction_rows'];by_condition.setdefault(entry['condition'],[]).extend(cos[cursor:cursor+n]);cursor+=n
    assert cursor==len(cos)
    source[head]=dict(value_huber=stats['value_huber'],direction_loss=stats['direction_loss'],
        all=distribution(cos),per_condition={c:distribution(v) for c,v in by_condition.items()})
result=dict(source_train_only=source,calibration_only=summary['alignment_distributions'],
    calibration_causal_delta=summary['derivative_changes'],heterogeneous_sobolev_over_value_only=summary['values']['value_only_ratio'],
    calibration_trajectory_distributions=summary['trajectory_distributions'],oracle=summary['oracle'],
    source_fit_is_not_a_gate=True,method_and_gates_unchanged=True)
(root/'final_distributions.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# T014 separate training and calibration diagnostics','',
    'Source-fit diagnostics are not qualification evidence. Calibration gate/selection remain frozen.','',
    '| Source head | Value Huber | Direction loss | Eligible rows | Positive | Positive fraction | Median cosine |',
    '|---|---:|---:|---:|---:|---:|---:|']
for name,s in source.items():
    d=s['all'];lines.append(f"| {name} | {s['value_huber']:.12g} | {s['direction_loss']:.12g} | {d['count']} | {d['positive']} | {d['positive_fraction']:.12g} | {d['median']:.12g} |")
lines+=['','| Calibration head/group | N | Positive | Negative | Zero | Positive fraction | Mean | P10 | P25 | Median | P75 | P90 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for name,groups in summary['alignment_distributions'].items():
    for group,d in groups.items():
        q=d['quantiles'];values=[d[k] for k in ('count','positive','negative','zero','positive_fraction','mean')]+list(q.values())
        lines.append('| '+name+'/'+group+' | '+' | '.join(f'{v:.12g}' for v in values)+' |')
lines+=['','Calibration Sobolev-minus-value-only changes: '+str(summary['derivative_changes']),
        '', 'Heterogeneous Sobolev/value-only MSE ratio: '+str(summary['values']['value_only_ratio']),
        '', 'Reference-only oracle: '+str(summary['oracle'])]
for method,groups in summary['trajectory_distributions'].items():
    lines+=['','## '+method,'','| Group | Step histogram | No-active | Projected updates / updates | Final movable boundary / movable coordinates |',
            '|---|---|---:|---|---|']
    for group,d in groups.items():
        lines.append(f"| {group} | {d['selected_steps']} | {d['no_active']} | {d['projected_updates']}/{d['updates']} ({d['projected_update_fraction']:.8g}) | {d['movable_boundary_coordinates']}/{d['movable_coordinates']} ({d['movable_boundary_fraction']:.8g}) |")
(root/'final_distributions.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(source={h:{k:v for k,v in s.items() if k!='per_condition'} for h,s in source.items()},
                     calibration=summary['values'],criteria=summary['criteria'],oracle=summary['oracle']),indent=2))
