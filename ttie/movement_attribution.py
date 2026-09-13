"""T020-D post-freeze reference-only diagnosis. Standard library only."""
import argparse
import hashlib
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

CATEGORIES=('correct_center','false_move','missed_move','correct_move_direction','wrong_move_direction')
ERRORS=('false_move','missed_move','wrong_move_direction')
GROUPS=('nonspatial_pool','clean','homogeneous_dark','homogeneous_bright')
HARD=tuple((x,y) for x in (.4,.5,.6) for y in (.4,.5,.6))
CLASSES=(.5,.4,.6)
DEFINITIONS=dict(A='Target center -> center; otherwise choose frozen lower/upper logit argmax, lower first on ties.',
    B='Keep frozen center. For frozen moves use target sign only if target moves; otherwise preserve frozen false-move sign.',
    acceptance='Four group means H<=1.01*H0, plus zero harmful clean episodes; all five required.',
    interpretation={'10':'necessity-dominant','01':'direction-dominant','11':'both individually sufficient / mixed',
                    '00':'neither sufficient / interaction-or-representation-limited'}, reference_only=True,deployable=False)

def now():return datetime.now(timezone.utc).isoformat()
def digest(raw):return hashlib.sha256(raw).hexdigest()
def write(path,value):path.write_bytes((json.dumps(value,indent=2)+'\n').encode())

def category(pred,target):
    if target==.5:return 'correct_center' if pred==.5 else 'false_move'
    if pred==.5:return 'missed_move'
    return 'correct_move_direction' if pred==target else 'wrong_move_direction'

def counterfactual(pred,target,logits):
    return (.5 if target==.5 else (.4 if logits[1]>=logits[2] else .6),
            target if pred!=.5 and target!=.5 else pred)

def movement(x,y):return 'no_move' if x==y==.5 else 'x_only' if y==.5 else 'y_only' if x==.5 else 'both'

def summarize(rows):
    groups={}
    for name in GROUPS:
        rs=rows if name==GROUPS[0] else [r for r in rows if r['condition']==name]
        means={k:statistics.mean(r[k] for r in rs) for k in ('H0','H','H_star')}
        groups[name]=dict(count=len(rs),mse=means,ratios=dict(H_over_H0=means['H']/means['H0'],H_over_H_star=means['H']/means['H_star']),
            outcomes=dict(beneficial=sum(r['H']<r['H0'] for r in rs),equal=sum(r['H']==r['H0'] for r in rs),harmful=sum(r['H']>r['H0'] for r in rs)),
            movements={v:sum(r['movement']==v for r in rs) for v in ('no_move','x_only','y_only','both')})
    clauses={name+'_safety':g['mse']['H']<=1.01*g['mse']['H0'] for name,g in groups.items()}
    clauses['clean_zero_harmful']=groups['clean']['outcomes']['harmful']==0
    return dict(groups=groups,clauses=clauses,acceptance_vector=list(clauses.values()),passed=sum(clauses.values()),all_pass=all(clauses.values()))

def load_inputs():
    origins=json.loads(Path('research_log/T020D_inputs.json').read_bytes());raw={}
    for key,item in origins.items():
        raw[key]=subprocess.check_output(['git','show',item['commit']+':'+item['path']])
        assert digest(raw[key])==item['sha256']
    cf=json.loads(raw['C_freeze']);br=json.loads(raw['B_receipt']);bf=json.loads(raw['B_freeze'])
    for key,field in [('C_decisions','decisions'),('C_config','config'),('C_folds','folds')]:assert digest(raw[key])==cf[field+'_sha256']
    for key,field in [('B_targets','evaluation'),('B_table','candidate_table'),('B_freeze','table_frozen')]:assert digest(raw[key])==br[field+'_sha256']
    assert bf['candidate_table_sha256']==digest(raw['B_table'])
    er=json.loads(raw['C_evaluation_receipt'])
    assert er['decisions_sha256']==digest(raw['C_decisions']) and er['reference_sha256']==digest(raw['B_targets'])
    return {k:json.loads(v) for k,v in raw.items()},origins

def run(output,source):
    paths=['ttie/movement_attribution.py','research_log/T020D_inputs.json','research_log/T020D_verify.py','tests/test_movement_attribution.py']
    code={}
    for path in paths:
        raw=subprocess.check_output(['git','show',source+':'+path]);assert raw==Path(path).read_bytes();code[path]=digest(raw)
    output.mkdir(parents=True)
    write(output/'definitions.json',dict(**DEFINITIONS,source_sha=source,source_code_sha256=code,frozen_utc=now()))
    data,origins=load_inputs();write(output/'input_provenance.json',dict(verified_utc=now(),origins=origins,receipt_links_verified=True))
    predictions=data['C_decisions'];targets=data['B_targets'];table=data['B_table']
    assert len(predictions)==len(targets)==len(table)==120
    attribution=[]; variants={k:[] for k in ('original','A','B')}
    for i,(p,t,ref) in enumerate(zip(predictions,targets,table)):
        assert p['row_index']==t['row_index']==ref['row_index']==i
        assert t['candidate_mse']==ref['candidate_mse'] and t['image_id']==ref['image_id'] and t['condition']==ref['condition']
        axes={};choices={'original':[], 'A':[], 'B':[]}
        for axis in ('x','y'):
            pred=p['b'+axis];target=t['b'+axis];logits=p[axis+'_logits']
            assert pred==CLASSES[p[axis+'_class']]==CLASSES[max(range(3),key=lambda j:logits[j])]
            axes[axis]=category(pred,target);a,b=counterfactual(pred,target,logits)
            for k,v in [('original',pred),('A',a),('B',b)]:choices[k].append(v)
        common=dict(row_index=i,image_id=t['image_id'],condition=t['condition'],fold=p['fold'])
        for key,(x,y) in choices.items():
            h=ref['candidate_mse'][HARD.index((x,y))]
            variants[key].append(dict(**common,bx=x,by=y,H0=t['H0'],H=h,H_star=t['H_star'],movement=movement(x,y)))
        attribution.append(dict(**common,x=axes['x'],y=axes['y'],harmful=variants['original'][-1]['H']>t['H0'],
            **{e:e in axes.values() for e in ERRORS}))
    for fold in data['C_folds']:
        assert [r['row_index'] for r in attribution if r['fold']==fold['fold']]==fold['heldout']
        assert sorted({r['image_id'] for r in attribution if r['fold']==fold['fold']})==fold['heldout_image_ids']
    harmful=[r for r in attribution if r['harmful']];assert len(harmful)==20
    counts={name:{a:{c:sum(r[a]==c for r in attribution if name==GROUPS[0] or r['condition']==name) for c in CATEGORIES} for a in ('x','y')} for name in GROUPS}
    overlap=dict(rows=harmful,any_error_counts={e:sum(r[e] for r in harmful) for e in ERRORS},
        exact_patterns={''.join(map(str,bits)):sum(tuple(int(r[e]) for e in ERRORS)==bits for r in harmful)
            for bits in ((a,b,c) for a in (0,1) for b in (0,1) for c in (0,1))},bit_order=list(ERRORS))
    reports={k:summarize(rows) for k,rows in variants.items()}
    interpretation=DEFINITIONS['interpretation'][str(int(reports['A']['all_pass']))+str(int(reports['B']['all_pass']))]
    for key,rows in variants.items():write(output/(key+'_decisions.json'),rows)
    write(output/'axis_errors.json',attribution);write(output/'axis_counts.json',counts);write(output/'harmful_overlap.json',overlap)
    write(output/'summary.json',dict(counterfactuals=reports,interpretation=interpretation,reference_only=True,deployable=False))
    # Re-read original Git bytes to establish preservation of frozen predictions.
    o=origins['C_decisions'];assert digest(subprocess.check_output(['git','show',o['commit']+':'+o['path']]))==o['sha256']
    write(output/'completed.json',dict(completed_utc=now(),prediction_sha256_unchanged=o['sha256'],output_sha256={p.name:digest(p.read_bytes()) for p in output.glob('*.json')}))
    print(interpretation,{k:v['acceptance_vector'] for k,v in reports.items()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source-sha',required=True);a=p.parse_args();run(a.output,a.source_sha)
