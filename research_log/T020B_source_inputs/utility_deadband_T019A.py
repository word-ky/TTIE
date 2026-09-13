"""T019-A fixed development-only reference target audit; no model or images."""
import argparse
import json
from pathlib import Path
import platform
from .hard_local import (SOURCES as BASE_SOURCES, CROSS_INDICES, HARD_INDICES,
    load_inputs, precheck, measure, report_groups)
from .local_geometry import CANDIDATES, local_choice, now, sha, write
from .routing.provenance import verify_source

DELTA = .01
SOURCES = (*BASE_SOURCES, 'ttie/utility_deadband.py')
ORIGIN_PACK = Path('research_log/T019A_origin_objects.pack')


def axis(center, lower, upper):
    gains = ((center-lower)/center, (center-upper)/center)
    moves = max(gains) >= DELTA
    value = (.4 if lower <= upper else .6) if moves else .5
    return dict(value=value, gains=list(gains), threshold_equal=max(gains)==DELTA,
                noncenter_tie=lower==upper, moving_tie=moves and lower==upper)


def choose(table):
    result=[]
    for i,row in enumerate(table):
        cross=[row['candidate_mse'][j] for j in CROSS_INDICES]
        x=axis(cross[0],cross[1],cross[2]);y=axis(cross[0],cross[3],cross[4])
        original=local_choice(cross)
        result.append(dict(row_index=i,bx=x['value'],by=y['value'],hard_index=CANDIDATES.index((x['value'],y['value'],0.)),
            cross_values=cross,x=x,y=y,original_bx=original['bx'],original_by=original['by']))
    return result


def summarize(quantities, table, decisions):
    report,rows=report_groups(quantities,table)
    for name,g in report['groups'].items():
        selected=rows if name=='spatial_pool' else [r for r in rows if r['condition']==name]
        choices=[decisions[r['row_index']] for r in selected]
        g['targets']=dict(x={str(v):sum(d['bx']==v for d in choices) for v in (.4,.5,.6)},
            y={str(v):sum(d['by']==v for d in choices) for v in (.4,.5,.6)},
            joint={f'{x},{y}':sum(d['bx']==x and d['by']==y for d in choices) for x in (.4,.5,.6) for y in (.4,.5,.6)})
        sx=sum(d['original_bx']!=.5 and d['bx']==.5 for d in choices)
        sy=sum(d['original_by']!=.5 and d['by']==.5 for d in choices)
        g['suppressed_to_center']=dict(x=sx,y=sy,axes_total=sx+sy,
            episodes=sum((d['original_bx']!=.5 and d['bx']==.5) or (d['original_by']!=.5 and d['by']==.5) for d in choices))
    report['tie_cases']=[dict(row_index=d['row_index'],condition=table[d['row_index']]['condition'],axis=a,
        center=d['cross_values'][0],gains=d[a]['gains'],choice=d[a]['value'],threshold_equal=d[a]['threshold_equal'],
        noncenter_tie=d[a]['noncenter_tie'],moving_tie=d[a]['moving_tie'])
        for d in decisions for a in ('x','y') if d[a]['threshold_equal'] or d[a]['noncenter_tie']]
    report['interaction_failures']=[dict(**r,x_gains=decisions[r['row_index']]['x']['gains'],y_gains=decisions[r['row_index']]['y']['gains'])
        for r in rows if r['harmful'] and r['bx']!=.5 and r['by']!=.5]
    report['zero_harmful']=report['groups']['spatial_pool']['gains']['harmful']==0
    report['acceptance_vector']=[*report['clauses'].values(),report['zero_harmful']]
    report['accepted']=all(report['acceptance_vector'])
    report['interpretation']='T019-A utility-deadband target viable' if report['accepted'] else 'T019-A utility-deadband target negative'
    report['delta']=DELTA;report['H1_meaning']='H_delta';report['fresh_E_reference_inputs']=False
    return report,rows


def finish(output, decisions, table, config_sha):
    write(output/'decisions.json',decisions)
    write(output/'decisions_frozen.json',dict(finalized_utc=now(),decisions_sha256=sha(output/'decisions.json'),
        config_sha256=config_sha,count=len(decisions),family_labels_used=False,delta=DELTA))
    quantities=measure(decisions,table);write(output/'quantities.json',quantities)
    write(output/'quantities_frozen.json',dict(finalized_utc=now(),quantities_sha256=sha(output/'quantities.json'),
        decisions_sha256=sha(output/'decisions.json'),count=len(quantities),family_labels_used=False))
    started=now();report,rows=summarize(quantities,table,decisions)
    write(output/'summary.json',report);write(output/'evaluation.json',rows)
    write(output/'report_receipt.json',dict(reporting_started_utc=started,completed_utc=now(),config_sha256=config_sha,
        **{name+'_sha256':sha(output/(name+'.json')) for name in ('decisions','decisions_frozen','quantities','quantities_frozen','summary','evaluation')}))
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-sha',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    code=verify_source(a.source_sha,SOURCES);data,hashes=load_inputs();baseline=precheck(data)
    a.output.mkdir(parents=True)
    write(a.output/'baseline.json',baseline)
    write(a.output/'config.json',dict(task='T019-A',source_sha=a.source_sha,source_code_sha256=code,input_artifact_hashes=hashes,
        origin_objects_sha256=sha(ORIGIN_PACK),delta=DELTA,cross_indices=CROSS_INDICES,hard_indices=HARD_INDICES,
        noncenter_tie_order=[.4,.6],original_tie_order=[.5,.4,.6],H1_meaning='H_delta',
        runtime=dict(python=platform.python_version(),device='cpu'),reference_only=True,fresh_E_reference_inputs=False))
    result=finish(a.output,choose(data['candidate_metrics']),data['candidate_metrics'],sha(a.output/'config.json'))
    print(json.dumps({k:result[k] for k in ('interpretation','clauses','zero_harmful','acceptance_vector')}))


if __name__=='__main__':main()
