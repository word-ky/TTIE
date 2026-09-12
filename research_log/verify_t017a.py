"""Independent array-based recomputation of the frozen reference-only audit."""
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import subprocess
import numpy as np

root=Path('research_log/T017A_run');read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda b:hashlib.sha256(b).hexdigest()
cfg=read(root/'config.json');inputs={}
for name,item in cfg['input_artifact_hashes'].items():
    blob=subprocess.check_output(['git','show',item['commit']+':'+item['path']]);assert sha(blob)==item['sha256'];inputs[name]=json.loads(blob)
for path,digest in cfg['source_code_sha256'].items():
    blob=subprocess.check_output(['git','show',cfg['source_sha']+':'+path]);assert sha(blob)==digest==sha(Path(path).read_bytes())
expected=[[a,b,t] for a in (.4,.5,.6) for b in (.4,.5,.6) for t in (0.,.05,.1)]
assert cfg['candidates']==inputs['config']['candidates']==expected
table=inputs['candidate_metrics'];assert len(table)==120
baseline=read(root/'baseline.json');frozen=read(root/'decisions_frozen.json');receipt=read(root/'evaluation_receipt.json')
assert baseline['completed_utc']<=frozen['finalized_utc']<=receipt['family_reporting_started_utc']
assert sha((root/'decisions_frozen.json').read_bytes())==receipt['decisions_frozen_sha256']
assert sha((root/'decisions.json').read_bytes())==frozen['decisions_sha256']==receipt['decisions_sha256']
decisions=read(root/'decisions.json');rows=read(root/'evaluation.json');report=read(root/'summary.json');b={r['episode']:r for r in inputs['accepted_b']}
assert len(decisions)==120 and [d['row_index'] for d in decisions]==list(range(120))
def safe(n,d):return n/d if d else None
for i,(source,d,r) in enumerate(zip(table,decisions,rows)):
    assert 'image_id' not in d and 'condition' not in d
    assert len(source['candidate_mse'])==27
    array=np.array(source['candidate_mse'],dtype=np.float64).reshape(3,3,3)
    def axis(v):
        minimum=min(v)
        if v[1]==minimum:return 1
        return 0 if v[0]==minimum else 2
    x=axis(array[:,1,1]);y=axis(array[1,:,1]);index=(x*3+y)*3
    assert d['bx']==(.4,.5,.6)[x] and d['by']==(.4,.5,.6)[y] and d['hard_index']==index
    assert d['cross_values']==[array[1,1,1],array[0,1,1],array[2,1,1],array[1,0,1],array[1,2,1]]
    assert d['gx']==(array[2,1,1]-array[0,1,1])/.2 and d['gy']==(array[1,2,1]-array[1,0,1])/.2
    hard=array[:,:,0].ravel().tolist();ref=b[source['source_directory']]
    assert hard==ref['reference_mse'] and min(hard)==ref['hard_oracle_mse']
    canonical=source['region2_mse'];assert array[1,1,0]==canonical==ref['region2_mse']
    local=float(array[x,y,0]);oracle=min(hard);oi=hard.index(oracle)
    assert r['selected_mse']==local and r['region2_mse']==canonical and r['hard_oracle_mse']==oracle
    assert r['local_boundary_index']==index//3 and r['hard_oracle_index']==oi
    assert r['image_id']==source['image_id'] and r['condition']==source['condition'] and r['episode']==source['source_directory']
    assert r['oracle_minimum_ties']==hard.count(oracle) and r['disagreement']==(index//3!=oi)
    assert r['outside_oracle_tie_set']==(local!=oracle)
    for key,value in dict(absolute_gain=canonical-local,relative_gain=safe(canonical-local,canonical),oracle_gap=local-oracle,
                          oracle_available_gain=canonical-oracle,oracle_gain_captured=safe(canonical-local,canonical-oracle)).items():assert r[key]==value
assert baseline['hard_oracle_mse']==st.mean(r['hard_oracle_mse'] for r in rows)==baseline['accepted_b_hard_oracle_mse']
def dist(values,d):
    v=[x for x in values if x is not None];assert d['count']==len(values) and d['null_count']==len(values)-len(v)
    assert d['mean']==(st.mean(v) if v else None) and d['median']==(st.median(v) if v else None)
    for p,x in d['quantiles'].items():assert math.isclose(x,float(np.quantile(v,int(p)/100,method='linear')),rel_tol=1e-12,abs_tol=1e-12)
for c,g in report['groups'].items():
    group=rows if c=='spatial_pool' else [r for r in rows if r['condition']==c]
    for key in ('selected_mse','region2_mse','hard_oracle_mse'):assert g[key]==st.mean(r[key] for r in group)
    assert g['ratios']==dict(selected_over_region2=safe(g['selected_mse'],g['region2_mse']),selected_over_hard_oracle=safe(g['selected_mse'],g['hard_oracle_mse']))
    assert g['oracle_gap']==g['selected_mse']-g['hard_oracle_mse']
    assert g['oracle_gain_captured']==safe(g['region2_mse']-g['selected_mse'],g['region2_mse']-g['hard_oracle_mse'])
    assert g['gains']==dict(beneficial=sum(r['absolute_gain']>0 for r in group),harmful=sum(r['absolute_gain']<0 for r in group),zero=sum(r['absolute_gain']==0 for r in group))
    for key in ('absolute_gain','relative_gain','oracle_gap','oracle_available_gain','oracle_gain_captured'):dist([r[key] for r in group],g['gain_distributions'][key])
    assert g['zero_oracle_gain_episodes']==sum(r['oracle_available_gain']==0 for r in group)
    assert g['tied_oracle_episodes']==sum(r['oracle_minimum_ties']>1 for r in group)
    assert g['local_counts']==[sum(r['local_boundary_index']==j for r in group) for j in range(9)]
    assert g['oracle_counts']==[sum(r['hard_oracle_index']==j for r in group) for j in range(9)]
    assert g['disagreement_rate']==st.mean(r['disagreement'] for r in group)
    assert g['outside_oracle_tie_set_rate']==st.mean(r['outside_oracle_tie_set'] for r in group)
    assert g['inside_oracle_tie_set_rate']==st.mean(not r['outside_oracle_tie_set'] for r in group)
    for key in ('gx','gy'):
        v=[r[key] for r in group];d=g[key];assert (d['negative'],d['zero'],d['positive'])==(sum(x<0 for x in v),sum(x==0 for x in v),sum(x>0 for x in v));dist(v,d['distribution'])
g=report['groups'];a=g['spatial_pool'];o=g['offset_left_right_40'];l=g['left_right'];q=g['quadrants']
clauses=[a['selected_mse']<=.97*a['region2_mse'],a['selected_mse']<=1.03*a['hard_oracle_mse'],o['selected_mse']<=.95*o['region2_mse'],l['selected_mse']<=1.01*l['region2_mse'],q['selected_mse']<=1.01*q['region2_mse']]
assert list(report['clauses'].values())==clauses and report['passed']==sum(clauses) and report['qualified']==all(clauses)
result=dict(passed=True,source_files=len(cfg['source_code_sha256']),input_artifacts=len(inputs),table_shape=[120,27],canonical_equal=120,
    accepted_hard_values_equal=1080,local_choices_verified=120,finite_differences_verified=240,
    metrics_counts_ties_zero_denominators_quantiles_clauses_verified=True,decisions_frozen_before_family_reporting=True,
    zero_oracle_gain_episodes=a['zero_oracle_gain_episodes'],tied_oracle_episodes=a['tied_oracle_episodes'],reference_only=True,no_pixels_models_or_rendering=True)
Path('research_log/T017A_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
