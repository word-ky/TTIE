"""T016-C reference-only OOF metrics. Rank/tie equations reuse T016-B 4062e01c."""
import math
import statistics

CONDITIONS=('left_right','quadrants','offset_left_right_40')


def ratio(x,y): return x/y if y else None
def ranks(values): return [sum(y<x for y in values)+(sum(y==x for y in values)+1)/2 for x in values]


def spearman(a,b):
    x,y=ranks(a),ranks(b); mx,my=statistics.mean(x),statistics.mean(y)
    return ratio(sum((u-mx)*(v-my) for u,v in zip(x,y)),
                 math.sqrt(sum((u-mx)**2 for u in x)*sum((v-my)**2 for v in y)))


def describe(values):
    kept=[v for v in values if v is not None]
    return dict(count=len(values),null_count=len(values)-len(kept),
                mean=statistics.mean(kept) if kept else None,median=statistics.median(kept) if kept else None,
                min=min(kept) if kept else None,max=max(kept) if kept else None)


def interpretation(probe28_passes,probe30_passes):
    if probe28_passes:return 'existing_28d_development_sufficient'
    if probe30_passes:return 'explicit_geometry_restores_development_rankability'
    return 'neither_probe_establishes_development_rankability'


def evaluate(oof,reference):
    lookup={r['episode']:r for r in reference}; rows=[]
    for prediction in oof:
        old=lookup[prediction['episode']]; mse=old['reference_mse']; selected=prediction['selected_index']
        oracle=min(range(9),key=lambda i:(mse[i],i))
        rows.append(dict(**prediction,image_id=old['image_id'],condition=old['condition'],
            selected_mse=mse[selected],region2_mse=old['region2_mse'],hard_oracle_mse=min(mse),
            frozen_t016b_mse=old['selected_mse'],oracle_index=oracle,disagreement=selected!=oracle,
            outside_oracle_tie_set=mse[selected]!=min(mse),spearman=spearman(prediction['predictions'],mse),
            prediction_minimum_ties=prediction['predictions'].count(min(prediction['predictions'])),
            reference_minimum_ties=mse.count(min(mse))))
    groups={}
    for condition in ('spatial_pool',*CONDITIONS):
        cases=rows if condition=='spatial_pool' else [r for r in rows if r['condition']==condition]
        g={k:statistics.mean(r[k] for r in cases) for k in ('selected_mse','region2_mse','hard_oracle_mse','frozen_t016b_mse')}
        g.update(count=len(cases),ratios={f'selected_over_{k}':ratio(g['selected_mse'],g[k+'_mse'])
                  for k in ('region2','hard_oracle','frozen_t016b')},
            selected_counts=[sum(r['selected_index']==i for r in cases) for i in range(9)],
            oracle_counts=[sum(r['oracle_index']==i for r in cases) for i in range(9)],
            disagreement_rate=statistics.mean(r['disagreement'] for r in cases),
            outside_oracle_tie_set_rate=statistics.mean(r['outside_oracle_tie_set'] for r in cases),
            prediction_tied_episodes=sum(r['prediction_minimum_ties']>1 for r in cases),
            oracle_tied_episodes=sum(r['reference_minimum_ties']>1 for r in cases),
            spearman=describe([r['spearman'] for r in cases]))
        groups[condition]=g
    a=groups['spatial_pool']; o=groups[CONDITIONS[2]]; l=groups[CONDITIONS[0]]; q=groups[CONDITIONS[1]]
    clauses=dict(spatial_improves_region2_3pct=a['selected_mse']<=.97*a['region2_mse'],
        spatial_within_hard_oracle_5pct=a['selected_mse']<=1.05*a['hard_oracle_mse'],
        offset_improves_region2_5pct=o['selected_mse']<=.95*o['region2_mse'],
        left_right_no_more_than_1pct_worse=l['selected_mse']<=1.01*l['region2_mse'],
        quadrants_no_more_than_1pct_worse=q['selected_mse']<=1.01*q['region2_mse'])
    folds=[]
    for f in range(5):
        cases=[r for r in rows if r['fold']==f]
        if cases:
            folds.append(dict(fold=f,count=len(cases),image_ids=sorted({r['image_id'] for r in cases}),
                selected_mse=statistics.mean(r['selected_mse'] for r in cases),
                per_condition={c:statistics.mean(r['selected_mse'] for r in cases if r['condition']==c) for c in CONDITIONS}))
    return dict(groups=groups,folds=folds,clauses=clauses,passed=sum(clauses.values()),qualified=all(clauses.values())),rows


def report_markdown(report):
    lines=['# T016-C: grouped OOF feature-sufficiency probe','',
        'Result: **'+report['interpretation']+'**. Development-only supervised diagnostic; not fresh qualification or a deployable selector.','',
        'Forty fixed development IDs, sorted-ID modulo five folds; 32 training / 8 held-out IDs per fold. All conditions and candidates for an image stay together. Ten heads, CPU, exact fixed T014 value-head recipe. Train-only normalization; no model/epoch selection. All OOF predictions were frozen before reference evaluation.','']
    for name,p in report['probes'].items():
        lines += ['## '+name,'',f"Fixed clauses: {p['passed']}/5; {p['clauses']}",'',
            '| Group | Selected MSE | Region2 | Hard oracle | Frozen T016-B | /Region2 | /Hard oracle | /Frozen T016-B |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for condition,g in p['groups'].items():
            values=[g[k] for k in ('selected_mse','region2_mse','hard_oracle_mse','frozen_t016b_mse')]+list(g['ratios'].values())
            lines.append('| '+condition+' | '+' | '.join('null' if v is None else f'{v:.12g}' for v in values)+' |')
        lines += ['','| Group | Selected counts | Oracle counts | Disagreement | Outside oracle ties | Spearman summary |','|---|---|---|---:|---:|---|']
        for condition,g in p['groups'].items():
            lines.append(f"| {condition} | {g['selected_counts']} | {g['oracle_counts']} | {g['disagreement_rate']} | {g['outside_oracle_tie_set_rate']} | {g['spearman']} |")
        lines += ['','| Fold | Held-out IDs | Selected MSE | LR / Quadrants / Offset MSE | Final train Huber |','|---|---|---:|---|---:|']
        for f in p['folds']:
            lines.append(f"| {f['fold']} | {f['image_ids']} | {f['selected_mse']:.12g} | {f['per_condition']} | {p['final_train_huber'][f['fold']]:.12g} |")
        lines += ['', 'Tie/null details per condition: '+str({c:{k:g[k] for k in ('prediction_tied_episodes','oracle_tied_episodes')} for c,g in p['groups'].items()})+'.', '']
    lines += ['## Two-probe comparison','',str(report['comparison']), '',
        'Spearman uses average ranks for ties; constant ranks are null. Exact predicted-value and oracle ties use the first lexicographic boundary. Head inputs exclude IDs/condition/candidate IDs/reference MSE; probe30 appends only the prescribed gx/gy coordinates.','',
        'No threshold, fold, epoch, architecture or loss was changed. No new images, CLIP/A6000 scoring, rendering, TTT, deployable boundary model, or follow-on experiment. Stop for research-lead review.','']
    return '\n'.join(lines)
