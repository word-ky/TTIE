"""Independent table arithmetic cross-check, without importing the audit implementation."""
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
from datetime import datetime,timezone

root=Path('research_log/T017B_run')
read=lambda name:json.loads((root/(name+'.json')).read_text())
digest=lambda b:hashlib.sha256(b).hexdigest()
config=read('config');inputs={}
for name,item in config['input_artifact_hashes'].items():
    blob=subprocess.check_output(['git','show',item['commit']+':'+item['path']])
    assert digest(blob)==item['sha256'];inputs[name]=json.loads(blob)
for name,expected in config['source_code_sha256'].items():
    blob=subprocess.check_output(['git','show',config['source_sha']+':'+name])
    assert digest(blob)==expected==digest(Path(name).read_bytes())
assert len(inputs)==9
assert digest((root/'frozen_A_decisions.json').read_bytes())==config['input_artifact_hashes']['decisions']['sha256']
decisions=read('frozen_A_decisions');table=inputs['table_candidate_metrics'];measured=read('quantities')
assert decisions==inputs['decisions'] and len(decisions)==len(measured)==120
grid=inputs['table_config']['candidates'];assert len(grid)==27
assert grid==[[x,y,t] for x in (.4,.5,.6) for y in (.4,.5,.6) for t in (0.,.05,.1)]
for d,row,actual in zip(decisions,table,measured):
    landscape={tuple(c):v for c,v in zip(grid,row['candidate_mse'])}
    x=min((.4,.5,.6),key=lambda x:(landscape[x,.5,.05],x!=.5,x))
    y=min((.4,.5,.6),key=lambda y:(landscape[.5,y,.05],y!=.5,y))
    assert (x,y)==(d['bx'],d['by']) and grid[d['hard_index']]==[x,y,0.]
    s0=landscape[.5,.5,.05];s1=landscape[x,y,.05];h0=landscape[.5,.5,0.];h1=landscape[x,y,0.]
    soft=[landscape[a,b,.05] for a in (.4,.5,.6) for b in (.4,.5,.6)]
    sg=s0-s1;hg=h0-h1;best=min(soft)
    category=None if hg>=0 else 'transfer_flip' if sg>0 else 'soft_interaction_failure' if sg<0 else 'zero/tie'
    expected=dict(row_index=d['row_index'],hard_index=d['hard_index'],S0=s0,S1=s1,H0=h0,H1=h1,S_star=best,
        soft_joint_gain=sg,hard_gain=hg,soft_separability_regret=s1-best,harmful_category=category,
        soft_oracle_index=soft.index(best),soft_oracle_ties=soft.count(best),
        equals_first_soft_oracle=d['hard_index']//3==soft.index(best),in_soft_oracle_tie_set=s1==best)
    assert actual==expected
assert read('baseline')['clauses']==inputs['summary']['clauses']
assert list(read('baseline')['clauses'].values())==[True,True,True,True,False]
summary=read('summary');categories=('transfer_flip','soft_interaction_failure','zero/tie')
for name,g in summary['groups'].items():
    group=[r for r in measured if name=='spatial_pool' or table[r['row_index']]['condition']==name]
    n=len(group);harm=sum(r['hard_gain']<0 for r in group)
    counts={c:sum(r['harmful_category']==c for r in group) for c in categories}
    assert g['count']==n and g['harmful_hard_moves']==harm and g['category_counts']==counts
    assert g['category_fractions']=={c:v/harm if harm else None for c,v in counts.items()}
    def sign(v):return 'negative' if v<0 else 'positive' if v>0 else 'zero'
    for s,values in g['sign_contingency'].items():
        for h,v in values.items():assert v==sum(sign(r['soft_joint_gain'])==s and sign(r['hard_gain'])==h for r in group)
    regret=sorted(r['soft_separability_regret'] for r in group);pos=(n-1)*.95;i=int(pos)
    p95=regret[i]+(regret[min(i+1,n-1)]-regret[i])*(pos-i)
    assert g['soft_separability_regret']==dict(mean=statistics.mean(regret),median=statistics.median(regret),p95=p95,max=max(regret),zero_count=regret.count(0.))
    for prefix,field in [('first_soft_oracle_equal','equals_first_soft_oracle'),('in_soft_oracle_tie_set','in_soft_oracle_tie_set')]:
        count=sum(r[field] for r in group);assert g[prefix+'_count']==count and g[prefix+'_fraction']==count/n
    assert g['soft_oracle_tied_episodes']==sum(r['soft_oracle_ties']>1 for r in group)
    assert g['selected_in_tied_soft_oracle_count']==sum(r['soft_oracle_ties']>1 and r['in_soft_oracle_tie_set'] for r in group)
    expected='no_harmful_moves' if not harm else 'transfer_dominant' if 3*counts['transfer_flip']>=2*harm else 'interaction_dominant' if 3*counts['soft_interaction_failure']>=2*harm else 'mixed/inconclusive'
    assert g['dominance']==expected
assert summary['groups']['spatial_pool']['dominance']==summary['groups']['quadrants']['dominance']==summary['interpretation']=='transfer_dominant'
frozen=read('quantities_frozen');receipt=read('report_receipt')
assert frozen['finalized_utc']<=receipt['reporting_started_utc']
assert frozen['quantities_sha256']==receipt['quantities_sha256']==digest((root/'quantities.json').read_bytes())
assert receipt['quantities_frozen_sha256']==digest((root/'quantities_frozen.json').read_bytes())
assert receipt['original_decisions_sha256']==config['input_artifact_hashes']['decisions']['sha256']
assert all('condition' not in r and 'image_id' not in r for r in measured)
assert read('attribution')==[dict(**r,condition=table[r['row_index']]['condition'],image_id=table[r['row_index']]['image_id']) for r in measured]
result=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),source_hashes=5,input_hashes=9,
    choices_exact=120,per_episode_quantities_exact=120,all_group_statistics_exact=True,
    frozen_choice_hash_unchanged=True,quantities_frozen_before_family_reporting=True,interpretation='transfer_dominant')
Path('research_log/T017B_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
