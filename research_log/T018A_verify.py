"""Independent hard-grid arithmetic; no imports from the audit implementation."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from datetime import datetime,timezone

root=Path('research_log/T018A_run')
read=lambda n:json.loads((root/(n+'.json')).read_text(encoding='utf-8'))
digest=lambda b:hashlib.sha256(b).hexdigest()
cfg=read('config');data={}
for name,item in cfg['input_artifact_hashes'].items():
    b=subprocess.check_output(['git','show',item['commit']+':'+item['path']]);assert digest(b)==item['sha256'];data[name]=json.loads(b)
for path,h in cfg['source_code_sha256'].items():
    assert digest(subprocess.check_output(['git','show',cfg['source_sha']+':'+path]))==h==digest(Path(path).read_bytes())
grid=data['config']['candidates'];table=data['candidate_metrics'];decisions=read('decisions');rows=read('quantities')
assert grid==[[x,y,t] for x in (.4,.5,.6) for y in (.4,.5,.6) for t in (0.,.05,.1)]
assert len(table)==len(decisions)==len(rows)==120
def div(a,b):return a/b if b!=0 else None
for i,(t,d,r) in enumerate(zip(table,decisions,rows)):
    assert len(t['candidate_mse'])==27
    values={tuple(c):v for c,v in zip(grid,t['candidate_mse'])}
    order=(.5,.4,.6)
    x=min(order,key=lambda x:values[x,.5,0.]);y=min(order,key=lambda y:values[.5,y,0.])
    cross=[values[x,y,0.] for x,y in [(.5,.5),(.4,.5),(.6,.5),(.5,.4),(.5,.6)]]
    index=grid.index([x,y,0.]);h0=values[.5,.5,0.];h1=values[x,y,0.];hx=values[x,.5,0.];hy=values[.5,y,0.]
    hard=[(grid.index([a,b,0.]),values[a,b,0.]) for a in (.4,.5,.6) for b in (.4,.5,.6)];best=min(v for j,v in hard)
    expected_d=dict(row_index=i,bx=x,by=y,hard_index=index,gx=(cross[2]-cross[1])/.2,gy=(cross[4]-cross[3])/.2,cross_values=cross)
    assert d==expected_d
    expected=dict(row_index=i,bx=x,by=y,hard_index=index,H0=h0,H1=h1,H_star=best,Hx_selected=hx,Hy_selected=hy,
        x_nonworse=hx<=h0,y_nonworse=hy<=h0,harmful=h1>h0,pure_interaction_failure=h1>h0 and hx<=h0 and hy<=h0,
        movement='no_move' if x==.5 and y==.5 else 'x_only' if y==.5 else 'y_only' if x==.5 else 'both',
        gain=h0-h1,factorization_regret=h1-best,oracle_headroom=h0-best,oracle_recovery=div(h0-h1,h0-best),
        oracle_first_index=next(j for j,v in hard if v==best),oracle_tie_count=sum(v==best for j,v in hard),in_oracle_tie_set=h1==best)
    assert r==expected and hx<=h0 and hy<=h0
def dist(items):
    v=sorted(x for x in items if x is not None)
    def q(p):
        z=p*(len(v)-1);i=int(z);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(z-i)
    return dict(count=len(items),null_count=len(items)-len(v),mean=statistics.mean(v) if v else None,median=statistics.median(v) if v else None,
        quantiles={str(int(p*100)):q(p) for p in (0.,.05,.25,.5,.75,.95,1.)} if v else {})
report=read('summary')
for name,g in report['groups'].items():
    subset=[r for r in rows if name=='spatial_pool' or table[r['row_index']]['condition']==name]
    assert g['count']==len(subset);m={k:statistics.mean(r[k] for r in subset) for k in ('H0','H1','H_star')};assert g['mse']==m
    assert g['ratios']==dict(H1_over_H0=div(m['H1'],m['H0']),H1_over_H_star=div(m['H1'],m['H_star']))
    assert g['oracle_headroom']==m['H0']-m['H_star'] and g['oracle_recovery']==div(m['H0']-m['H1'],m['H0']-m['H_star'])
    assert g['oracle_recovery_distribution']==dist([r['oracle_recovery'] for r in subset])
    assert g['factorization_regret']==dist([r['factorization_regret'] for r in subset])
    assert g['zero_headroom_count']==sum(r['oracle_headroom']==0 for r in subset)
    assert g['movements']=={k:sum(r['movement']==k for r in subset) for k in ('no_move','x_only','y_only','both')}
    assert g['gains']==dict(beneficial=sum(r['gain']>0 for r in subset),equal=sum(r['gain']==0 for r in subset),harmful=sum(r['gain']<0 for r in subset))
    count=sum(r['in_oracle_tie_set'] for r in subset);assert g['oracle_tie_set_match_count']==count and g['oracle_tie_set_match_fraction']==count/len(subset)
    assert g['tied_oracle_episodes']==sum(r['oracle_tie_count']>1 for r in subset)
    assert g['pure_interaction_failures']==sum(r['pure_interaction_failure'] for r in subset)
    assert g['harmful_examples']==[{k:r[k] for k in ('row_index','bx','by','H0','H1','Hx_selected','Hy_selected','x_nonworse','y_nonworse','pure_interaction_failure')} for r in subset if r['H1']>r['H0']]
p=report['groups']['spatial_pool']['mse'];o=report['groups']['offset_left_right_40']['mse'];l=report['groups']['left_right']['mse'];q=report['groups']['quadrants']['mse']
vector=[p['H1']<=.97*p['H0'],p['H1']<=1.03*p['H_star'],o['H1']<=.95*o['H0'],l['H1']<=1.01*l['H0'],q['H1']<=1.01*q['H0']]
assert list(report['clauses'].values())==vector and report['passed']==sum(vector)==5
assert report['interpretation']=='hard_local_direction_target_viable'
receipt=read('report_receipt');df=read('decisions_frozen');qf=read('quantities_frozen')
assert read('baseline')['completed_utc']<=df['finalized_utc']<=qf['finalized_utc']<=receipt['reporting_started_utc']
for name in ('decisions','quantities'):
    assert read(name+'_frozen')[name+'_sha256']==receipt[name+'_sha256']==digest((root/(name+'.json')).read_bytes())
    assert receipt[name+'_frozen_sha256']==digest((root/(name+'_frozen.json')).read_bytes())
    assert all('condition' not in r and 'image_id' not in r for r in read(name))
assert qf['decisions_sha256']==receipt['decisions_sha256']
assert read('evaluation')==[dict(**r,condition=table[r['row_index']]['condition']) for r in rows]
result=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),source_hashes=5,input_hashes=2,choices_exact=120,
    quantities_exact=120,all_group_statistics_exact=True,axis_choices_nonworse=120,harmful_combined_moves=0,
    choices_and_quantities_frozen_before_family_reporting=True,literal_clauses=vector,literal_passed=5,interpretation=report['interpretation'])
Path('research_log/T018A_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
