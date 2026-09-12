"""T014 literal gates and offline distributions, never used by the optimizer."""
from collections import Counter
import statistics
import torch
from .energy_metrics import stage_a as energy_a,stage_b_energy,markdown
from .sobolev_io import PRIMARY_METHOD,CONTROL,ORACLE,METHODS


def distribution(values):
    if not values:return dict(count=0,positive_fraction=0.,median=0.)
    return dict(count=len(values),positive=sum(v>0 for v in values),negative=sum(v<0 for v in values),zero=sum(v==0 for v in values),
        positive_fraction=sum(v>0 for v in values)/len(values),median=statistics.median(values),mean=statistics.mean(values),
        minimum=min(values),maximum=max(values),quantiles=dict(zip(('p10','p25','p50','p75','p90'),
        torch.quantile(torch.tensor(values,dtype=torch.float64),torch.tensor([.1,.25,.5,.75,.9],dtype=torch.float64)).tolist())))


def alignment_distributions(rows):
    return {m:{c:distribution([r['cosine'] for r in rows if r['method']==m and (c=='all' or r['condition']==c)])
               for c in ('all',*dict.fromkeys(r['condition'] for r in rows))} for m in (CONTROL,PRIMARY_METHOD)}


def add_control(report):
    h=report['groups']['heterogeneous'];ratio=h[PRIMARY_METHOD]['mse']['mean']/h[CONTROL]['mse']['mean']
    report['values']['value_only_ratio']=ratio;report['criteria']['beyond_value_only']=ratio<=.95
    report['passes' if report['stage']=='A' else 'qualified']=all(report['criteria'].values())
    report['failed']=[k for k,v in report['criteria'].items() if not v]
    return report


def stage_a(rows,alignments):
    report=energy_a(rows,[a for a in alignments if a['method']==PRIMARY_METHOD],primary=PRIMARY_METHOD,oracle=ORACLE)
    report['alignment_distributions']=alignment_distributions(alignments)
    d=report['alignment_distributions'];p=d[PRIMARY_METHOD]['all'];c=d[CONTROL]['all']
    report['derivative_changes']=dict(positive_fraction=p['positive_fraction']-c['positive_fraction'],median=p['median']-c['median'])
    return add_control(report)


def stage_b(rows):
    return add_control(stage_b_energy(rows,primary=PRIMARY_METHOD,oracle=ORACLE,
        global_method='global_ttt_energy_sobolev',bilinear='bilinear2_ttt_energy_sobolev'))


def trajectory_diagnostics(trajectories,decisions):
    result={}
    for m,t in trajectories.items():
        diag=t['diagnostics'];events=diag['projections'];lo=torch.tensor(diag['action_box']['lower']);hi=torch.tensor(diag['action_box']['upper'])
        end=t['grids'][-1];boundary=(torch.isclose(end,lo,atol=1e-6,rtol=0)|torch.isclose(end,hi,atol=1e-6,rtol=0))
        free=lo<hi
        hits=sum(torch.tensor(e['pre_raw']).ne(torch.tensor(e['post_raw'])).any().item() for e in events)
        result[m]=dict(selected_step=decisions[m]['selected_step'],no_active=decisions[m]['bypass']=='no_active',
            updates=len(events),projected_updates=hits,final_boundary_coordinates=int(boundary.sum()),final_coordinates=end.numel(),
            movable_boundary_coordinates=int((boundary&free).sum()),movable_coordinates=int(free.sum()))
    return result


def summarize_trajectories(entries):
    result={}
    for method in METHODS:
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
