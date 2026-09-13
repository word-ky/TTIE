"""Independent saved-score and post-freeze reference arithmetic."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from datetime import datetime,timezone

root=Path('research_log/T018B_run');read=lambda n:json.loads((root/(n+'.json')).read_text(encoding='utf-8'))
digest=lambda b:hashlib.sha256(b).hexdigest();cfg=read('config');pre=read('reference_precheck');scoring={};references={}
for entries,destination in [(cfg['score_inputs'],scoring),(pre['input_artifact_hashes'],references)]:
    for name,item in entries.items():
        b=subprocess.check_output(['git','show',item['commit']+':'+item['path']]);assert digest(b)==item['sha256'];destination[name]=json.loads(b)
for path,h in cfg['source_code_sha256'].items():assert digest(Path(path).read_bytes())==h==digest(subprocess.check_output(['git','show',cfg['source_sha']+':'+path]))
hard=[(x,y,0.) for x in (.4,.5,.6) for y in (.4,.5,.6)];grid=references['config']['candidates']
assert scoring['selection']['candidates']==scoring['config']['candidates']==[list(c) for c in hard]
assert grid==[[x,y,t] for x in (.4,.5,.6) for y in (.4,.5,.6) for t in (0.,.05,.1)]
decisions=read('decisions');scores=scoring['selection']['episodes'];table=references['candidate_metrics'];evaluated=read('evaluation')
assert len(decisions)==len(scores)==len(table)==len(evaluated)==120
# Recompute choices before accessing reference rows or target choices in this verifier.
expected_decisions=[]
for i,s in enumerate(scores):
    energy={c:v for c,v in zip(hard,s['energies'])};order=(.5,.4,.6)
    x=min(order,key=lambda x:energy[x,.5,0.]);y=min(order,key=lambda y:energy[.5,y,0.])
    cross=[energy[a,b,0.] for a,b in [(.5,.5),(.4,.5),(.6,.5),(.5,.4),(.5,.6)]]
    expected_decisions.append(dict(row_index=i,bx=x,by=y,hard_index=grid.index([x,y,0.]),score_index=hard.index((x,y,0.)),cross_energies=cross))
assert decisions==expected_decisions
lookup={r['source_directory']:i for i,r in enumerate(table)};assert len(lookup)==120 and set(lookup)=={s['episode'] for s in scores}
for d,s,actual in zip(decisions,scores,evaluated):
    index=lookup[s['episode']];t=table[index];target=references['target_decisions'][index];prior=references['target_quantities'][index]
    assert s['corners_sha256']==t['corners_sha256'];values={tuple(c):v for c,v in zip(grid,t['candidate_mse'])}
    x=d['bx'];y=d['by'];h0=values[.5,.5,0.];hs=values[x,y,0.];best=min(values[c] for c in hard)
    assert prior['H0']==h0 and prior['H_star']==best and prior['H1']==values[target['bx'],target['by'],0.]
    expected=dict(row_index=d['row_index'],reference_row_index=index,bx=x,by=y,hard_index=d['hard_index'],target_bx=target['bx'],target_by=target['by'],H0=h0,Hselected=hs,H_star=best,
        x_match=x==target['bx'],y_match=y==target['by'],joint_match=(x,y)==(target['bx'],target['by']),
        movement='no_move' if x==.5 and y==.5 else 'x_only' if y==.5 else 'y_only' if x==.5 else 'both',condition=t['condition'])
    assert actual==expected
summary=read('summary')
for name,g in summary['groups'].items():
    rows=[r for r in evaluated if name=='spatial_pool' or r['condition']==name]
    assert g['count']==len(rows);m={k:statistics.mean(r[k] for r in rows) for k in ('H0','Hselected','H_star')};assert m==g['mse']
    assert g['ratios']==dict(selected_over_H0=m['Hselected']/m['H0'],selected_over_H_star=m['Hselected']/m['H_star'])
    assert g['target_agreement']=={k:dict(count=sum(r[k] for r in rows),fraction=sum(r[k] for r in rows)/len(rows)) for k in ('x_match','y_match','joint_match')}
    assert g['movements']=={k:sum(r['movement']==k for r in rows) for k in ('no_move','x_only','y_only','both')}
    assert g['gains']==dict(beneficial=sum(r['Hselected']<r['H0'] for r in rows),equal=sum(r['Hselected']==r['H0'] for r in rows),harmful=sum(r['Hselected']>r['H0'] for r in rows))
    assert g['harmful_examples']==[{k:r[k] for k in ('row_index','bx','by','target_bx','target_by','H0','Hselected','H_star')} for r in rows if r['Hselected']>r['H0']]
p=summary['groups']['spatial_pool']['mse'];o=summary['groups']['offset_left_right_40']['mse'];l=summary['groups']['left_right']['mse'];q=summary['groups']['quadrants']['mse']
v=[p['Hselected']<=.97*p['H0'],p['Hselected']<=1.03*p['H_star'],o['Hselected']<=.95*o['H0'],l['Hselected']<=1.01*l['H0'],q['Hselected']<=1.01*q['H0']]
assert list(summary['clauses'].values())==v and summary['passed']==sum(v)==0 and summary['interpretation']=='frozen_energy_local_signal_insufficient'
freeze=read('decisions_frozen');receipt=read('evaluation_receipt')
assert freeze['finalized_utc']<receipt['reference_opened_utc']<=pre['completed_utc']<=receipt['completed_utc']
assert freeze['decisions_sha256']==receipt['decisions_sha256']==digest((root/'decisions.json').read_bytes())
assert freeze['config_sha256']==digest((root/'config.json').read_bytes())
assert receipt['decisions_frozen_sha256']==digest((root/'decisions_frozen.json').read_bytes())
assert all(not any(k in r for k in ('condition','image_id','target_bx','H0','reference_mse')) for r in decisions)
result=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),source_hashes=6,score_input_hashes=3,postfreeze_reference_input_hashes=8,
    decisions_exact=120,postfreeze_identity_and_metrics_exact=120,all_group_statistics_and_harmful_examples_exact=True,
    decisions_frozen_before_reference_open=True,decision_bytes_unchanged=True,reference_order_matches_score_order=all(r['row_index']==r['reference_row_index'] for r in evaluated),
    clauses=v,literal_passed=0,interpretation=summary['interpretation'])
Path('research_log/T018B_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
