"""Independent arithmetic verification; does not import TTIE audit functions."""
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
from datetime import datetime,timezone

root=Path('research_log/T017C_run')
read=lambda name:json.loads((root/(name+'.json')).read_text(encoding='utf-8'))
digest=lambda b:hashlib.sha256(b).hexdigest()
cfg=read('config');inputs={}
for name,item in cfg['input_artifact_hashes'].items():
    blob=subprocess.check_output(['git','show',item['commit']+':'+item['path']])
    assert digest(blob)==item['sha256'];inputs[name]=json.loads(blob)
for path,expected in cfg['source_code_sha256'].items():
    blob=subprocess.check_output(['git','show',cfg['source_sha']+':'+path])
    assert digest(blob)==expected==digest(Path(path).read_bytes())
decisions=read('frozen_A_decisions');table=inputs['table_candidate_metrics'];rows=read('quantities')
assert decisions==inputs['decisions'] and len(rows)==len(table)==len(decisions)==120
assert digest((root/'frozen_A_decisions.json').read_bytes())==inputs['decisions_frozen']['decisions_sha256']==cfg['input_artifact_hashes']['decisions']['sha256']
grid=inputs['table_config']['candidates'];assert grid==[[x,y,t] for x in (.4,.5,.6) for y in (.4,.5,.6) for t in (0.,.05,.1)]
def divide(a,b):return None if b==0 else a/b
for d,t,r,prior in zip(decisions,table,rows,inputs['B_quantities']):
    assert len(t['candidate_mse'])==27
    values={tuple(c):v for c,v in zip(grid,t['candidate_mse'])}
    x=min((.4,.5,.6),key=lambda x:(values[x,.5,.05],x!=.5,x));y=min((.4,.5,.6),key=lambda y:(values[.5,y,.05],y!=.5,y))
    assert (x,y)==(d['bx'],d['by']) and grid[d['hard_index']]==[x,y,0.]
    h0=values[.5,.5,0.];h1=values[x,y,0.];s0=values[.5,.5,.05];s1=values[x,y,.05]
    best=min(v for (x,y,tau),v in values.items() if tau==.05)
    for k,v in dict(H0=h0,H1=h1,S0=s0,S1=s1,S_star=best).items():assert r[k]==prior[k]==v
    assert {k:r[k] for k in prior}==prior
    assert r['fixed_smoothing_gain']==divide(h0-s0,h0) and r['adaptive_soft_gain']==divide(s0-s1,s0)
    assert r['soft_oracle_headroom']==s0-best and r['soft_oracle_recovery']==divide(s0-s1,s0-best)
    assert 'condition' not in r and 'image_id' not in r
def distribution(values):
    v=sorted(x for x in values if x is not None)
    def quantile(p):
        pos=p*(len(v)-1);i=int(pos);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(pos-i)
    return dict(count=len(values),null_count=len(values)-len(v),mean=statistics.mean(v) if v else None,
        median=statistics.median(v) if v else None,quantiles={str(int(p*100)):quantile(p) for p in (0.,.05,.25,.5,.75,.95,1.)} if v else {})
report=read('summary')
for name,g in report['groups'].items():
    subset=[r for r in rows if name=='spatial_pool' or table[r['row_index']]['condition']==name]
    assert len(subset)==g['count'];m={k:statistics.mean(r[k] for r in subset) for k in ('H0','S0','S1','S_star')};assert g['mse']==m
    assert g['ratios']=={a+'_over_'+b:divide(m[a],m[b]) for a,b in [('S1','H0'),('S0','H0'),('S1','S0'),('S1','S_star'),('S0','S_star')]}
    assert g['fixed_smoothing_gain']==divide(m['H0']-m['S0'],m['H0']) and g['adaptive_soft_gain']==divide(m['S0']-m['S1'],m['S0'])
    assert g['gains']==dict(fixed_smoothing_mse=m['H0']-m['S0'],adaptive_soft_mse=m['S0']-m['S1'],total_mse=m['H0']-m['S1'])
    assert g['soft_oracle_headroom']==m['S0']-m['S_star'] and g['soft_oracle_recovery']==divide(m['S0']-m['S1'],m['S0']-m['S_star'])
    assert g['soft_oracle_recovery_distribution']==distribution([r['soft_oracle_recovery'] for r in subset])
    assert g['zero_soft_oracle_headroom_count']==sum(r['S0']==r['S_star'] for r in subset)
    assert g['S1_vs_S0']==dict(beneficial=sum(r['S1']<r['S0'] for r in subset),equal=sum(r['S1']==r['S0'] for r in subset),harmful=sum(r['S1']>r['S0'] for r in subset))
p=report['groups']['spatial_pool']['mse'];o=report['groups']['offset_left_right_40']['mse'];l=report['groups']['left_right']['mse'];q=report['groups']['quadrants']['mse']
vector=[p['S1']<=.97*p['H0'],p['S1']<=1.03*p['S_star'],o['S1']<=.95*o['H0'],l['S1']<=1.01*l['H0'],q['S1']<=1.01*q['H0']]
assert list(report['clauses'].values())==vector and report['passed']==sum(vector)==2
assert report['pooled_adaptive_1pct']==(p['S1']<=.99*p['S0'])
literal='matched_soft_insufficient' if not all(vector) else 'matched_soft_adaptive_geometry_viable' if p['S1']<=.99*p['S0'] else 'fixed_soft_dominant'
assert report['interpretation']==literal
freeze=read('quantities_frozen');receipt=read('report_receipt')
assert freeze['finalized_utc']<=receipt['reporting_started_utc']
assert freeze['quantities_sha256']==receipt['quantities_sha256']==digest((root/'quantities.json').read_bytes())
assert receipt['quantities_frozen_sha256']==digest((root/'quantities_frozen.json').read_bytes())
assert freeze['decisions_sha256']==receipt['decisions_sha256']==digest((root/'frozen_A_decisions.json').read_bytes())
assert read('evaluation')==[dict(**r,condition=table[r['row_index']]['condition'],image_id=table[r['row_index']]['image_id']) for r in rows]
assert read('baseline')['completed_utc']<=freeze['finalized_utc']
result=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),source_hashes=6,input_hashes=8,choices_exact=120,
    B_quantities_exact=120,all_C_quantities_and_group_statistics_exact=True,choice_bytes_unchanged=True,
    quantities_frozen_before_family_reporting=True,clauses=vector,literal_passed=2,interpretation=literal)
Path('research_log/T017C_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
