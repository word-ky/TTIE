"""Independent immutable-input replay; imports no TTIE implementation."""
import argparse
import hashlib
import itertools
import json
import statistics
import subprocess
from datetime import datetime,timezone
from pathlib import Path

def read(p):return json.loads(p.read_bytes())
def sha(b):return hashlib.sha256(b).hexdigest()

def verify(out):
    provenance=read(out/'input_provenance.json'); inputs={};raws={}
    for key,o in provenance['origins'].items():
        raw=subprocess.check_output(['git','show',o['commit']+':'+o['path']]);assert sha(raw)==o['sha256'];raws[key]=raw;inputs[key]=json.loads(raw)
    f=inputs['C_freeze'];br=inputs['B_receipt']
    for k,n in [('C_decisions','decisions'),('C_config','config'),('C_folds','folds')]:assert sha(raws[k])==f[n+'_sha256']
    for k,n in [('B_targets','evaluation'),('B_table','candidate_table'),('B_freeze','table_frozen')]:assert sha(raws[k])==br[n+'_sha256']
    assert inputs['B_freeze']['candidate_table_sha256']==sha(raws['B_table'])
    assert inputs['C_evaluation_receipt']['reference_sha256']==sha(raws['B_targets'])
    completed=read(out/'completed.json')
    for n,h in completed['output_sha256'].items():assert sha((out/n).read_bytes())==h
    defs=read(out/'definitions.json')
    for path,h in defs['source_code_sha256'].items():assert sha(subprocess.check_output(['git','show',defs['source_sha']+':'+path]))==h
    assert f['finalized_utc']<defs['frozen_utc']<provenance['verified_utc']<completed['completed_utc']
    classes=[.5,.4,.6];hard=list(itertools.product([.4,.5,.6],repeat=2));attrs=[];variants={k:[] for k in ('original','A','B')}
    # Explicit nine-entry truth table independent of the diagnostic branching.
    cats={(0,0):'correct_center',(0,1):'false_move',(0,2):'false_move',
          (1,0):'missed_move',(2,0):'missed_move',(1,1):'correct_move_direction',
          (2,2):'correct_move_direction',(1,2):'wrong_move_direction',(2,1):'wrong_move_direction'}
    errors=['false_move','missed_move','wrong_move_direction']
    for i,(p,t,table) in enumerate(zip(inputs['C_decisions'],inputs['B_targets'],inputs['B_table'])):
        assert p['row_index']==t['row_index']==table['row_index']==i and t['candidate_mse']==table['candidate_mse']
        assert (t['image_id'],t['condition'])==(table['image_id'],table['condition'])
        identity=dict(row_index=i,image_id=t['image_id'],condition=t['condition'],fold=p['fold'])
        values={k:[] for k in variants};axis={}
        for a in ('x','y'):
            pc=p[a+'_class'];tc=classes.index(t['b'+a]);logits=p[a+'_logits']
            assert classes[pc]==p['b'+a] and pc==max(range(3),key=logits.__getitem__)
            axis[a]=cats[tc,pc]
            ac=0 if tc==0 else 1+int(logits[2]>logits[1])
            bc=pc if pc==0 or tc==0 else tc
            for k,c in [('original',pc),('A',ac),('B',bc)]:values[k].append(classes[c])
        for k,xy in values.items():
            x,y=xy;moves=['no_move','y_only','x_only','both'][2*int(x!=.5)+int(y!=.5)]
            variants[k].append(dict(**identity,bx=x,by=y,H0=table['candidate_mse'][4],H=table['candidate_mse'][hard.index((x,y))],H_star=min(table['candidate_mse']),movement=moves))
        attrs.append(dict(**identity,**axis,harmful=variants['original'][-1]['H']>variants['original'][-1]['H0'],**{e:e in axis.values() for e in errors}))
    assert len(attrs)==120 and len({r['image_id'] for r in attrs})==40 and attrs==read(out/'axis_errors.json')
    for fold in inputs['C_folds']:
        assert [r['row_index'] for r in attrs if r['fold']==fold['fold']]==fold['heldout']
        assert sorted({r['image_id'] for r in attrs if r['fold']==fold['fold']})==fold['heldout_image_ids']
    counts=read(out/'axis_counts.json')
    for name,g in counts.items():
        rs=attrs if name=='nonspatial_pool' else [r for r in attrs if r['condition']==name]
        for axis,cs in g.items():assert cs=={c:sum(r[axis]==c for r in rs) for c in set(cats.values())}
    harmful=[r for r in attrs if r['harmful']];overlap=read(out/'harmful_overlap.json');assert len(harmful)==20 and overlap['rows']==harmful
    assert overlap['any_error_counts']=={e:sum(r[e] for r in harmful) for e in errors} and overlap['bit_order']==errors
    assert overlap['exact_patterns']=={''.join(map(str,bits)):sum(tuple(int(r[e]) for e in errors)==bits for r in harmful) for bits in itertools.product((0,1),repeat=3)}
    summary=read(out/'summary.json');passed={}
    for k,rows in variants.items():
        assert rows==read(out/(k+'_decisions.json'));report=summary['counterfactuals'][k];vector=[]
        for name,g in report['groups'].items():
            rs=rows if name=='nonspatial_pool' else [r for r in rows if r['condition']==name]
            means={n:statistics.mean(r[n] for r in rs) for n in ('H0','H','H_star')}
            assert means==g['mse'] and len(rs)==g['count']
            assert g['ratios']==dict(H_over_H0=means['H']/means['H0'],H_over_H_star=means['H']/means['H_star'])
            assert g['outcomes']==dict(beneficial=sum(r['H']<r['H0'] for r in rs),equal=sum(r['H']==r['H0'] for r in rs),harmful=sum(r['H']>r['H0'] for r in rs))
            assert g['movements']=={m:sum(r['movement']==m for r in rs) for m in ('no_move','x_only','y_only','both')}
            vector.append(means['H']<=1.01*means['H0'])
        vector.append(all(r['H']<=r['H0'] for r in rows if r['condition']=='clean'))
        assert vector==report['acceptance_vector']==list(report['clauses'].values()) and sum(vector)==report['passed'] and all(vector)==report['all_pass']
        passed[k]=all(vector)
    labels={(True,False):'necessity-dominant',(False,True):'direction-dominant',(True,True):'both individually sufficient / mixed',(False,False):'neither sufficient / interaction-or-representation-limited'}
    assert summary['interpretation']==labels[passed['A'],passed['B']] and summary['reference_only'] and not summary['deployable']
    assert completed['prediction_sha256_unchanged']==sha(raws['C_decisions'])
    result=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),rows=120,harmful_rows=20,all_counts_and_overlaps_exact=True,both_counterfactuals_exact=True,all_metrics_and_clauses_exact=True,interpretation=summary['interpretation'],prediction_sha256_unchanged=sha(raws['C_decisions']))
    (out/'independent_verification.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(a.output)
