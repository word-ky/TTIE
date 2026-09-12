"""T017-A reference-only local geometry diagnostic; never a label-free selector."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
from .routing.provenance import verify_source

COORDINATES=(.4,.5,.6)
CANDIDATES=tuple((x,y,t) for x in COORDINATES for y in COORDINATES for t in (0.,.05,.1))
CROSS=((.5,.5,.05),(.4,.5,.05),(.6,.5,.05),(.5,.4,.05),(.5,.6,.05))
CROSS_INDICES=tuple(CANDIDATES.index(c) for c in CROSS)


def local_choice(cross):
    """Only five soft reference values: center, left, right, down, up."""
    center,left,right,down,up=cross
    def axis(values):
        index=min(range(3),key=lambda i:(values[i],i!=1,COORDINATES[i]))
        return COORDINATES[index]
    bx=axis((left,center,right));by=axis((down,center,up))
    return dict(bx=bx,by=by,hard_index=CANDIDATES.index((bx,by,0.)),
        gx=(right-left)/.2,gy=(up-down)/.2,cross_values=list(cross))


def choices(rows):
    # Row ordinal binds the choice; image/family identifiers are never read here.
    return [dict(row_index=i,**local_choice([row['candidate_mse'][j] for j in CROSS_INDICES])) for i,row in enumerate(rows)]


A='ee5d8fdaf3ab48ee7ad3654d45bdc65419be8367'
BASE='research_log/remote_runs/20260913-000502-ttie-t016a-screen/artifacts/audit/'
A_HASHES=dict(candidate_metrics='6091a0c928f115940997a647693b6571d8608131e7f9567a75882f0233b9c41e',
    config='e32b8b48ec0a93749698f51ab33637c5945bc8d827ec133175134bff93b168c5',
    summary='529660f3e1e464ff88d4603f9d4909a107cea5d8443c710ff57c329c7e93734d',
    sanity='3e9cd51f68945dc1dc010e36e2c870edbea76cadafa6ff17754f0edc8499e0e5',
    local_verification='5d172eced18a2c96f3d0460882100f2fae9d6ba8191c1904718969fed3e3cf01')
INPUTS={key:(A,BASE+key+'.json',digest) for key,digest in A_HASHES.items()}
INPUTS['accepted_b']=('4062e01cb93de731c394015c5ac741d6c08e04d8',
    'research_log/remote_runs/20260913-013748-ttie-t016b-assets-ready/artifacts/evaluation/evaluation.json',
    '8a2204fc481da3559b337ec634ef9c87cf890d63bd7edf2cf5ae947aaea0e22e')
SOURCES=('ttie/__init__.py','ttie/routing/__init__.py','ttie/routing/provenance.py','ttie/local_geometry.py')
CONDITIONS=('left_right','quadrants','offset_left_right_40')


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def ratio(n,d):return n/d if d!=0 else None


def load_inputs():
    data={};hashes={}
    for key,(commit,path,digest) in INPUTS.items():
        blob=subprocess.check_output(['git','show',commit+':'+path]);actual=hashlib.sha256(blob).hexdigest()
        assert actual==digest;data[key]=json.loads(blob);hashes[key]=dict(commit=commit,path=path,sha256=actual)
    return data,hashes


def precheck(data):
    rows=data['candidate_metrics'];assert len(rows)==120
    assert data['config']['candidates']==[list(c) for c in CANDIDATES]
    assert data['sanity']['passed'] and data['sanity']['max_mse_diff']==0
    old={r['episode']:r for r in data['accepted_b']}
    assert len(old)==len({r['source_directory'] for r in rows})==120
    for row in rows:
        mse=row['candidate_mse'];assert len(mse)==27 and mse[12]==row['region2_mse']
        ref=old[row['source_directory']];assert mse[::3]==ref['reference_mse']
        assert row['region2_mse']==ref['region2_mse'] and min(mse[::3])==ref['hard_oracle_mse']
    region2=statistics.mean(r['region2_mse'] for r in rows)
    hard=statistics.mean(min(r['candidate_mse'][::3]) for r in rows)
    accepted=statistics.mean(r['hard_oracle_mse'] for r in old.values())
    assert hard==accepted
    full=statistics.mean(min(r['candidate_mse']) for r in rows)
    prior=data['summary']['groups']['spatial_pool']
    assert abs(full-prior['oracle_soft_mse'])<1e-15 and abs(region2-prior['region2_mse'])<1e-15
    return dict(passed=True,completed_utc=now(),episodes=120,candidates=27,canonical_exact=120,hard_values_exact=1080,
        region2_mse=region2,hard_oracle_mse=hard,accepted_b_hard_oracle_mse=accepted,full27_oracle_mse=full,
        family_labels_used=False)


def distribution(values):
    v=sorted(x for x in values if x is not None)
    def quantile(p):
        pos=p*(len(v)-1);lo=int(pos);hi=min(lo+1,len(v)-1)
        return v[lo]+(v[hi]-v[lo])*(pos-lo)
    return dict(count=len(values),null_count=len(values)-len(v),mean=statistics.mean(v) if v else None,
        median=statistics.median(v) if v else None,
        quantiles={str(int(p*100)):quantile(p) for p in (0.,.05,.25,.5,.75,.95,1.)} if v else {})


def gradient_summary(values):
    return dict(negative=sum(v<0 for v in values),zero=sum(v==0 for v in values),positive=sum(v>0 for v in values),
        distribution=distribution(values))


def evaluate(decisions,table):
    rows=[]
    for d in decisions:
        old=table[d['row_index']];mse=old['candidate_mse'];hard=mse[::3];oracle=min(range(9),key=lambda i:hard[i])
        selected=mse[d['hard_index']];canonical=old['region2_mse'];minimum=min(hard)
        rows.append(dict(**d,image_id=old['image_id'],condition=old['condition'],episode=old['source_directory'],
            selected_mse=selected,region2_mse=canonical,hard_oracle_mse=minimum,hard_oracle_index=oracle,
            local_boundary_index=d['hard_index']//3,oracle_minimum_ties=hard.count(minimum),
            disagreement=d['hard_index']//3!=oracle,outside_oracle_tie_set=selected!=minimum,
            absolute_gain=canonical-selected,relative_gain=ratio(canonical-selected,canonical),
            oracle_gap=selected-minimum,oracle_available_gain=canonical-minimum,
            oracle_gain_captured=ratio(canonical-selected,canonical-minimum)))
    groups={}
    for c in ('spatial_pool',*CONDITIONS):
        group=rows if c=='spatial_pool' else [r for r in rows if r['condition']==c]
        selected=statistics.mean(r['selected_mse'] for r in group);canonical=statistics.mean(r['region2_mse'] for r in group);oracle=statistics.mean(r['hard_oracle_mse'] for r in group)
        groups[c]=dict(count=len(group),selected_mse=selected,region2_mse=canonical,hard_oracle_mse=oracle,
            ratios=dict(selected_over_region2=ratio(selected,canonical),selected_over_hard_oracle=ratio(selected,oracle)),
            oracle_gap=selected-oracle,oracle_gain_captured=ratio(canonical-selected,canonical-oracle),
            gain_distributions={key:distribution([r[key] for r in group]) for key in ('absolute_gain','relative_gain','oracle_gap','oracle_available_gain','oracle_gain_captured')},
            gains=dict(beneficial=sum(r['absolute_gain']>0 for r in group),harmful=sum(r['absolute_gain']<0 for r in group),zero=sum(r['absolute_gain']==0 for r in group)),
            zero_oracle_gain_episodes=sum(r['oracle_available_gain']==0 for r in group),
            tied_oracle_episodes=sum(r['oracle_minimum_ties']>1 for r in group),
            local_counts=[sum(r['local_boundary_index']==i for r in group) for i in range(9)],
            oracle_counts=[sum(r['hard_oracle_index']==i for r in group) for i in range(9)],
            disagreement_rate=statistics.mean(r['disagreement'] for r in group),
            outside_oracle_tie_set_rate=statistics.mean(r['outside_oracle_tie_set'] for r in group),
            inside_oracle_tie_set_rate=statistics.mean(not r['outside_oracle_tie_set'] for r in group),
            gx=gradient_summary([r['gx'] for r in group]),gy=gradient_summary([r['gy'] for r in group]))
    a=groups['spatial_pool'];o=groups[CONDITIONS[2]];l=groups[CONDITIONS[0]];q=groups[CONDITIONS[1]]
    clauses=dict(spatial_improves_region2_3pct=a['selected_mse']<=.97*a['region2_mse'],
        spatial_within_hard_oracle_3pct=a['selected_mse']<=1.03*a['hard_oracle_mse'],
        offset_improves_region2_5pct=o['selected_mse']<=.95*o['region2_mse'],
        left_right_no_more_than_1pct_worse=l['selected_mse']<=1.01*l['region2_mse'],
        quadrants_no_more_than_1pct_worse=q['selected_mse']<=1.01*q['region2_mse'])
    report=dict(groups=groups,clauses=clauses,passed=sum(clauses.values()),qualified=all(clauses.values()),
        interpretation='reference_soft_cross_supports_local_geometry_viability' if all(clauses.values()) else 'coarse_soft_cross_does_not_yet_justify_continuous_geometry_optimization')
    return report,rows


def finish(output,decisions,table):
    write(output/'decisions.json',decisions)
    write(output/'decisions_frozen.json',dict(finalized_utc=now(),decisions_sha256=sha(output/'decisions.json'),
        choices=len(decisions),family_labels_attached=False,reference_only=True))
    started=now();report,rows=evaluate(decisions,table)
    write(output/'summary.json',report);write(output/'evaluation.json',rows)
    write(output/'evaluation_receipt.json',dict(family_reporting_started_utc=started,decisions_sha256=sha(output/'decisions.json'),
        decisions_frozen_sha256=sha(output/'decisions_frozen.json')))
    return report


def markdown(report):
    lines=['# T017-A: reference-only local geometry landscape','',
        f"Result: **{report['interpretation']}**, **{report['passed']}/5**.",'',
        'This rule uses reference MSE, not a label-free or deployable signal. It queries only five pre-rendered tau=.05 cross values and evaluates the selected pre-rendered hard candidate. No training, images or rendering.',
        '', 'Literalclauses: '+str(report['clauses']), '',
        '| Group | Selected MSE | Region2 | Hard oracle | /Region2 | /Oracle | Oracle gap | Oracle gain captured | Beneficial/Harmful/Zero |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for c,g in report['groups'].items():
        v=[g[k] for k in ('selected_mse','region2_mse','hard_oracle_mse')]+list(g['ratios'].values())+[g['oracle_gap'],g['oracle_gain_captured']]
        lines.append('| '+c+' | '+' | '.join('null' if x is None else f'{x:.12g}' for x in v)+' | '+str(g['gains'])+' |')
    lines+=['','Boundary counts use lexicographic bx/by on the nine hard candidates. Oracle exact ties use first lexicographic; the local axis rule prefers canonical, then the lower coordinate.',
        '', '| Group | Local / Oracle counts | Disagreement / Outside ties / Inside ties | Zero oracle gain / Tied oracle episodes |','|---|---|---|---|']
    for c,g in report['groups'].items():
        lines.append(f"| {c} | {g['local_counts']} / {g['oracle_counts']} | {g['disagreement_rate']} / {g['outside_oracle_tie_set_rate']} / {g['inside_oracle_tie_set_rate']} | {g['zero_oracle_gain_episodes']} / {g['tied_oracle_episodes']} |")
    for c,g in report['groups'].items():
        lines+=['','## '+c+' distributions','', 'gx: '+str(g['gx']), '', 'gy: '+str(g['gy'])]
        for key,dist in g['gain_distributions'].items():lines+=['',key+': '+str(dist)]
    lines+=['','Group captured gain is a ratio of aggregate gains; per-episode captured-gain distribution is reported separately, with zero denominator as null (never clipped or forced to zero). Quantiles are linear0/5/25/50/75/95/100percent. Exactzero gains/oracle ties remain explicit.', '',
        'All120choices are frozen before family labels are attached. gx/gy are diagnostic finite differences only; they do not change the axis-minimum rule. The finite cross is not a proof of continuous differentiability or learned-objective feasibility. Stop after this reference-only development audit; no source objective or fresh experiment.','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-sha',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=verify_source(a.source_sha,SOURCES);data,hashes=load_inputs();baseline=precheck(data)
    a.output.mkdir(parents=True);write(a.output/'baseline.json',baseline)
    write(a.output/'config.json',dict(task='T017-A',source_sha=a.source_sha,source_code_sha256=source,input_artifact_hashes=hashes,
        candidates=CANDIDATES,cross=CROSS,cross_indices=CROSS_INDICES,reference_only=True,development_only=True,
        image_reads=0,rendering_calls=0,model_training=0,runtime=dict(python=platform.python_version(),device='cpu'),
        axis_ties='center then lower coordinate',finite_difference_denominator=.2,quantiles=[0,5,25,50,75,95,100]))
    decisions=choices(data['candidate_metrics']);report=finish(a.output,decisions,data['candidate_metrics'])
    (a.output/'T017A_analysis.md').write_text(markdown(report),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],interpretation=report['interpretation'])))


if __name__=='__main__':main()
