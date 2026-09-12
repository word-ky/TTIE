"""Requested offline T011 diagnostics; consume persisted records, never adapt outputs."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics
import torch

TOLERANCE=1e-6  # Physical absolute tolerance frozen in commit d2929e6 before diagnostic access.


def episode_counts(diag,gate):
    active=torch.tensor(gate['active'],dtype=torch.bool)
    grid=torch.tensor(diag['final_grid'],dtype=torch.float64)
    node_active=active.reshape(1,2,2) if grid.shape[-1]==2 else active.any().reshape(1,1,1)
    mask=node_active[:,None].expand_as(grid)
    counts=Counter(episodes=1,updates=diag['steps'])
    for projection in diag['projections']:
        changed=torch.tensor(projection['pre_raw'],dtype=torch.float64)!=torch.tensor(projection['post_raw'],dtype=torch.float64)
        counts['hit_updates']+=int(changed.any())
        for channel,name in enumerate(('ev','gamma')):
            hit=changed[:,channel];active_hit=hit[node_active]
            counts[name+'_hit_updates']+=int(hit.any())
            counts[name+'_coordinates']+=hit.numel();counts[name+'_coordinate_hits']+=int(hit.sum())
            counts[name+'_active_coordinates']+=active_hit.numel();counts[name+'_active_coordinate_hits']+=int(active_hit.sum())
    lower=torch.tensor(diag['action_box']['lower'],dtype=torch.float64)
    upper=torch.tensor(diag['action_box']['upper'],dtype=torch.float64)
    boundary=((grid-lower).abs()<=TOLERANCE)|((grid-upper).abs()<=TOLERANCE)
    counts['final_active_coordinates']=int(mask.sum())
    counts['final_boundary_coordinates']=int((boundary&mask).sum())
    for channel,name in enumerate(('ev','gamma')):
        counts['final_'+name+'_active_coordinates']=int(node_active.sum())
        counts['final_'+name+'_boundary_coordinates']=int((boundary[:,channel]&node_active).sum())
    return counts


def fractions(counts):
    report=dict(counts)
    ratio=lambda a,b:counts[a]/counts[b] if counts[b] else None
    report['projection_hit_rate']=ratio('hit_updates','updates')
    report['final_boundary_occupancy']=ratio('final_boundary_coordinates','final_active_coordinates')
    for name in ('ev','gamma'):
        report[name+'_update_hit_rate']=ratio(name+'_hit_updates','updates')
        report[name+'_coordinate_hit_rate_all']=ratio(name+'_coordinate_hits',name+'_coordinates')
        report[name+'_coordinate_hit_rate_active']=ratio(name+'_active_coordinate_hits',name+'_active_coordinates')
        report['final_'+name+'_boundary_occupancy']=ratio('final_'+name+'_boundary_coordinates','final_'+name+'_active_coordinates')
    return report


def diagnose(root):
    entries=json.loads((root/'artifact_manifest.json').read_text());groups=defaultdict(lambda:defaultdict(Counter))
    gate_counts=defaultdict(Counter)
    for entry in entries:
        decisions=json.loads((root/entry['directory']/'decisions.json').read_text());gate=decisions['gate']
        winners=[w for w,a in zip(gate['winner'],gate['active']) if a]
        mode='no_active' if not winners else 'agree_dark' if all(w==0 for w in winners) else 'agree_bright' if all(w==1 for w in winners) else 'conflict'
        for condition in (entry['condition'],'overall'):
            gate_counts[condition][mode]+=1;gate_counts[condition]['inputs']+=1
        for method,diag in decisions['methods'].items():
            if 'ttt_projected' not in method:continue
            counts=episode_counts(diag,gate)
            for condition in (entry['condition'],'overall'):
                groups[condition][method].update(counts)
                groups[condition]['all_projected_methods'].update(counts)
    rows=json.loads((root/'metrics.json').read_text());paired=[]
    for entry in entries:
        subset={r['method']:r for r in rows if r['image_id']==entry['image_id'] and r['condition']==entry['condition']}
        paired.append(dict(image_id=entry['image_id'],condition=entry['condition'],
            one_step_minus_full_mse=subset['region2_ttt_projected_1step']['mse']-subset['region2_ttt_projected']['mse']))
    one_step={}
    for condition in ('clean','homogeneous_dark','homogeneous_bright','left_right','quadrants','heterogeneous','offset_left_right_40'):
        values=[r['one_step_minus_full_mse'] for r in paired if r['condition'] in ('left_right','quadrants')] if condition=='heterogeneous' else [r['one_step_minus_full_mse'] for r in paired if r['condition']==condition]
        one_step[condition]=dict(count=len(values),mean_one_step_minus_full_mse=statistics.mean(values),
            median_one_step_minus_full_mse=statistics.median(values),one_step_better_count=sum(v<0 for v in values),equal_count=sum(v==0 for v in values))
    return dict(task='T011',declaration_commit='d2929e6',physical_boundary_absolute_tolerance=TOLERANCE,
        hit_definition='Any exact inequality of persisted pre/post raw coordinate; monotone map to EV/gamma',
        boundary_denominator='Active fast coordinates only, excluding no-active global; EV sign-zero endpoint counts as boundary',
        groups={c:{m:fractions(n) for m,n in methods.items()} for c,methods in groups.items()},
        global_original_gate={c:{**n,**{k+'_fraction':n[k]/n['inputs'] for k in ('agree_dark','agree_bright','conflict','no_active')}} for c,n in gate_counts.items()},
        one_step_vs_full=one_step,paired_one_step_differences=paired,stress_report_only=True,no_method_selection=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('audit',type=Path);args=parser.parse_args()
    report=diagnose(args.audit)
    (args.audit/'projection_diagnostics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:report[k] for k in ('global_original_gate','one_step_vs_full')},indent=2))
