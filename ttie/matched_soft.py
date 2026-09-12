"""T017-C: reference-only matched-soft audit of immutable T017-A choices."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
from .local_geometry import CANDIDATES,choices,distribution,ratio,now,sha,write
from .geometry_attribution import measure
from .routing.provenance import verify_source

MERGED_B='0b052a0fd04acb12cdaa0ad69b9207c18c063119'
SOURCES=('ttie/__init__.py','ttie/routing/__init__.py','ttie/routing/provenance.py',
         'ttie/local_geometry.py','ttie/geometry_attribution.py','ttie/matched_soft.py')


def load_inputs():
    data={};hashes={};raw={}
    def read(name,commit,path,expected=None):
        blob=subprocess.check_output(['git','show',commit+':'+path]);digest=hashlib.sha256(blob).hexdigest()
        if expected is not None:assert digest==expected
        data[name]=json.loads(blob);hashes[name]=dict(commit=commit,path=path,sha256=digest);raw[name]=blob
    for name in ('config','quantities','quantities_frozen','report_receipt'):
        read('B_'+name,MERGED_B,'research_log/T017B_run/'+name+'.json')
    for name in ('decisions','decisions_frozen','table_candidate_metrics','table_config'):
        item=data['B_config']['input_artifact_hashes'][name]
        read(name,item['commit'],item['path'],item['sha256'])
    assert hashes['decisions']['sha256']==data['decisions_frozen']['decisions_sha256']==data['B_report_receipt']['original_decisions_sha256']
    assert hashes['B_quantities']['sha256']==data['B_quantities_frozen']['quantities_sha256']==data['B_report_receipt']['quantities_sha256']
    assert hashes['B_quantities_frozen']['sha256']==data['B_report_receipt']['quantities_frozen_sha256']
    return data,hashes,raw['decisions']


def baseline(data):
    table=data['table_candidate_metrics'];assert len(table)==120 and all(len(r['candidate_mse'])==27 for r in table)
    assert data['table_config']['candidates']==[list(c) for c in CANDIDATES]
    assert choices(table)==data['decisions']
    measured=measure(data['decisions'],table);assert measured==data['B_quantities']
    return dict(passed=True,completed_utc=now(),choices_exact=120,B_quantities_exact=120,family_labels_used=False),measured


def decompose(measured):
    return [dict(**r,fixed_smoothing_gain=ratio(r['H0']-r['S0'],r['H0']),
        adaptive_soft_gain=ratio(r['S0']-r['S1'],r['S0']),soft_oracle_headroom=r['S0']-r['S_star'],
        soft_oracle_recovery=ratio(r['S0']-r['S1'],r['S0']-r['S_star'])) for r in measured]


def clauses_and_interpretation(groups):
    p=groups['spatial_pool']['mse'];o=groups['offset_left_right_40']['mse']
    l=groups['left_right']['mse'];q=groups['quadrants']['mse']
    clauses=dict(pooled_3pct_better_H0=p['S1']<=.97*p['H0'],pooled_within_3pct_soft_oracle=p['S1']<=1.03*p['S_star'],
        offset_5pct_better_H0=o['S1']<=.95*o['H0'],left_right_safety=l['S1']<=1.01*l['H0'],quadrants_safety=q['S1']<=1.01*q['H0'])
    adaptive=p['S1']<=.99*p['S0']
    result='matched_soft_insufficient' if not all(clauses.values()) else 'matched_soft_adaptive_geometry_viable' if adaptive else 'fixed_soft_dominant'
    return dict(clauses=clauses,passed=sum(clauses.values()),pooled_adaptive_1pct=adaptive,interpretation=result)


def report_groups(quantities,table):
    rows=[dict(**r,condition=table[r['row_index']]['condition'],image_id=table[r['row_index']]['image_id']) for r in quantities]
    groups={}
    for condition in ('spatial_pool','left_right','quadrants','offset_left_right_40'):
        group=rows if condition=='spatial_pool' else [r for r in rows if r['condition']==condition]
        m={k:statistics.mean(r[k] for r in group) for k in ('H0','S0','S1','S_star')}
        groups[condition]=dict(count=len(group),mse=m,
            ratios={a+'_over_'+b:ratio(m[a],m[b]) for a,b in (('S1','H0'),('S0','H0'),('S1','S0'),('S1','S_star'),('S0','S_star'))},
            fixed_smoothing_gain=ratio(m['H0']-m['S0'],m['H0']),adaptive_soft_gain=ratio(m['S0']-m['S1'],m['S0']),
            gains=dict(fixed_smoothing_mse=m['H0']-m['S0'],adaptive_soft_mse=m['S0']-m['S1'],total_mse=m['H0']-m['S1']),
            soft_oracle_headroom=m['S0']-m['S_star'],soft_oracle_recovery=ratio(m['S0']-m['S1'],m['S0']-m['S_star']),
            soft_oracle_recovery_distribution=distribution([r['soft_oracle_recovery'] for r in group]),
            zero_soft_oracle_headroom_count=sum(r['soft_oracle_headroom']==0 for r in group),
            S1_vs_S0=dict(beneficial=sum(r['S1']<r['S0'] for r in group),equal=sum(r['S1']==r['S0'] for r in group),harmful=sum(r['S1']>r['S0'] for r in group)))
    return dict(groups=groups,**clauses_and_interpretation(groups),reference_only=True,qualification_experiment=False),rows


def finish(output,quantities,table):
    write(output/'quantities.json',quantities)
    write(output/'quantities_frozen.json',dict(finalized_utc=now(),quantities_sha256=sha(output/'quantities.json'),
        decisions_sha256=sha(output/'frozen_A_decisions.json'),family_labels_attached=False,count=len(quantities)))
    started=now();report,rows=report_groups(quantities,table)
    write(output/'summary.json',report);write(output/'evaluation.json',rows)
    write(output/'report_receipt.json',dict(reporting_started_utc=started,quantities_sha256=sha(output/'quantities.json'),
        quantities_frozen_sha256=sha(output/'quantities_frozen.json'),decisions_sha256=sha(output/'frozen_A_decisions.json')))
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-sha',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    code=verify_source(a.source_sha,SOURCES);data,hashes,raw=load_inputs();receipt,measured=baseline(data)
    a.output.mkdir(parents=True);write(a.output/'baseline.json',receipt);(a.output/'frozen_A_decisions.json').write_bytes(raw)
    write(a.output/'config.json',dict(task='T017-C',source_sha=a.source_sha,source_code_sha256=code,input_artifact_hashes=hashes,
        reference_only=True,choice_mutations=0,runtime=dict(python=platform.python_version(),device='cpu'),merged_B=MERGED_B))
    report=finish(a.output,decompose(measured),data['table_candidate_metrics'])
    assert sha(a.output/'frozen_A_decisions.json')==hashes['decisions']['sha256']
    print(json.dumps({k:report[k] for k in ('clauses','passed','pooled_adaptive_1pct','interpretation')}))


if __name__=='__main__':main()
