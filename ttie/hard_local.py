"""T018-A reference-only local hard-direction target; not a deployable selector."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
from .local_geometry import A,BASE,A_HASHES,CANDIDATES,local_choice,ratio,distribution,now,sha,write
from .routing.provenance import verify_source

CROSS_INDICES=(12,3,21,9,15)
HARD_INDICES=tuple(range(0,27,3))
SOURCES=('ttie/__init__.py','ttie/routing/__init__.py','ttie/routing/provenance.py','ttie/local_geometry.py','ttie/hard_local.py')


def load_inputs():
    data={};hashes={}
    for name in ('candidate_metrics','config'):
        path=BASE+name+'.json';blob=subprocess.check_output(['git','show',A+':'+path]);digest=hashlib.sha256(blob).hexdigest()
        assert digest==A_HASHES[name];data[name]=json.loads(blob);hashes[name]=dict(commit=A,path=path,sha256=digest)
    return data,hashes


def precheck(data):
    table=data['candidate_metrics'];grid=data['config']['candidates']
    assert len(table)==120 and all(len(r['candidate_mse'])==27 for r in table)
    assert grid==[list(c) for c in CANDIDATES]
    cross=[(.5,.5,0.),(.4,.5,0.),(.6,.5,0.),(.5,.4,0.),(.5,.6,0.)]
    hard=[(x,y,0.) for x in (.4,.5,.6) for y in (.4,.5,.6)]
    assert [tuple(grid[i]) for i in CROSS_INDICES]==cross and all(grid.count(list(c))==1 for c in cross)
    assert [tuple(grid[i]) for i in HARD_INDICES]==hard and all(grid.count(list(c))==1 for c in hard)
    assert all(r['candidate_mse'][12]==r['region2_mse'] for r in table)
    return dict(passed=True,completed_utc=now(),episodes=120,candidates=27,hard_cross_entries=600,hard_entries=1080,canonical_exact=120,family_labels_used=False)


def choose(table):
    # Reuse only the accepted scalar center/lower/upper argmin; supplied values are all hard.
    return [dict(row_index=i,**local_choice([r['candidate_mse'][j] for j in CROSS_INDICES])) for i,r in enumerate(table)]


def measure(decisions,table):
    rows=[]
    for d in decisions:
        mse=table[d['row_index']]['candidate_mse'];hard=[mse[i] for i in HARD_INDICES]
        h0=mse[12];h1=mse[d['hard_index']];best=min(hard)
        hx=mse[CANDIDATES.index((d['bx'],.5,0.))];hy=mse[CANDIDATES.index((.5,d['by'],0.))]
        x=d['bx']!=.5;y=d['by']!=.5
        rows.append(dict(row_index=d['row_index'],bx=d['bx'],by=d['by'],hard_index=d['hard_index'],H0=h0,H1=h1,H_star=best,
            Hx_selected=hx,Hy_selected=hy,x_nonworse=hx<=h0,y_nonworse=hy<=h0,harmful=h1>h0,
            pure_interaction_failure=h1>h0 and hx<=h0 and hy<=h0,
            movement='both' if x and y else 'x_only' if x else 'y_only' if y else 'no_move',
            gain=h0-h1,factorization_regret=h1-best,oracle_headroom=h0-best,oracle_recovery=ratio(h0-h1,h0-best),
            oracle_first_index=HARD_INDICES[hard.index(best)],oracle_tie_count=hard.count(best),in_oracle_tie_set=h1==best))
    return rows


def qualify(groups):
    p=groups['spatial_pool']['mse'];o=groups['offset_left_right_40']['mse'];l=groups['left_right']['mse'];q=groups['quadrants']['mse']
    clauses=dict(pooled_3pct_better_H0=p['H1']<=.97*p['H0'],pooled_within_3pct_oracle=p['H1']<=1.03*p['H_star'],
        offset_5pct_better_H0=o['H1']<=.95*o['H0'],left_right_safety=l['H1']<=1.01*l['H0'],quadrants_safety=q['H1']<=1.01*q['H0'])
    return dict(clauses=clauses,passed=sum(clauses.values()),interpretation='hard_local_direction_target_viable' if all(clauses.values()) else 'hard_local_factorization_insufficient')


def report_groups(quantities,table):
    rows=[dict(**r,condition=table[r['row_index']]['condition']) for r in quantities];groups={}
    for name in ('spatial_pool','left_right','quadrants','offset_left_right_40'):
        group=rows if name=='spatial_pool' else [r for r in rows if r['condition']==name]
        m={k:statistics.mean(r[k] for r in group) for k in ('H0','H1','H_star')}
        groups[name]=dict(count=len(group),mse=m,ratios=dict(H1_over_H0=ratio(m['H1'],m['H0']),H1_over_H_star=ratio(m['H1'],m['H_star'])),
            oracle_headroom=m['H0']-m['H_star'],oracle_recovery=ratio(m['H0']-m['H1'],m['H0']-m['H_star']),
            oracle_recovery_distribution=distribution([r['oracle_recovery'] for r in group]),zero_headroom_count=sum(r['oracle_headroom']==0 for r in group),
            movements={k:sum(r['movement']==k for r in group) for k in ('no_move','x_only','y_only','both')},
            gains=dict(beneficial=sum(r['H1']<r['H0'] for r in group),equal=sum(r['H1']==r['H0'] for r in group),harmful=sum(r['H1']>r['H0'] for r in group)),
            oracle_tie_set_match_count=sum(r['in_oracle_tie_set'] for r in group),oracle_tie_set_match_fraction=statistics.mean(r['in_oracle_tie_set'] for r in group),
            tied_oracle_episodes=sum(r['oracle_tie_count']>1 for r in group),factorization_regret=distribution([r['factorization_regret'] for r in group]),
            pure_interaction_failures=sum(r['pure_interaction_failure'] for r in group),
            harmful_examples=[{k:r[k] for k in ('row_index','bx','by','H0','H1','Hx_selected','Hy_selected','x_nonworse','y_nonworse','pure_interaction_failure')} for r in group if r['harmful']])
    return dict(groups=groups,**qualify(groups),reference_only=True,deployable=False),rows


def finish(output,decisions,table):
    write(output/'decisions.json',decisions)
    write(output/'decisions_frozen.json',dict(finalized_utc=now(),decisions_sha256=sha(output/'decisions.json'),count=len(decisions),family_labels_used=False))
    quantities=measure(decisions,table);write(output/'quantities.json',quantities)
    write(output/'quantities_frozen.json',dict(finalized_utc=now(),quantities_sha256=sha(output/'quantities.json'),decisions_sha256=sha(output/'decisions.json'),count=len(quantities),family_labels_used=False))
    started=now();report,rows=report_groups(quantities,table)
    write(output/'summary.json',report);write(output/'evaluation.json',rows)
    write(output/'report_receipt.json',dict(reporting_started_utc=started,decisions_sha256=sha(output/'decisions.json'),
        decisions_frozen_sha256=sha(output/'decisions_frozen.json'),quantities_sha256=sha(output/'quantities.json'),quantities_frozen_sha256=sha(output/'quantities_frozen.json')))
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-sha',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    code=verify_source(a.source_sha,SOURCES);data,hashes=load_inputs();baseline=precheck(data);a.output.mkdir(parents=True)
    write(a.output/'baseline.json',baseline);write(a.output/'config.json',dict(task='T018-A',source_sha=a.source_sha,source_code_sha256=code,input_artifact_hashes=hashes,
        cross_indices=CROSS_INDICES,hard_indices=HARD_INDICES,tie_order=[.5,.4,.6],runtime=dict(python=platform.python_version(),device='cpu'),reference_only=True))
    report=finish(a.output,choose(data['candidate_metrics']),data['candidate_metrics'])
    print(json.dumps({k:report[k] for k in ('clauses','passed','interpretation')}))


if __name__=='__main__':main()
