"""Independent arithmetic/provenance replay; imports no TTIE or model library."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify(root,output,receipt):
    config=read(output/'config.json');lock=config['lock'];report=read(output/'report_receipt.json')
    assert lock['delta']==.01 and lock['reference_only'] and lock['training'] is False and lock['new_image_ids']==0
    assert config['reference_only'] and config['adaptation_rerun'] is False
    assert digest(root/'research_log/T020B_pipeline_lock.json')==config['pipeline_lock_sha256']
    origins=lock['origins'];inputs=root/'research_log/T020B_source_inputs'
    for name,o in origins.items():
        raw=subprocess.check_output(['git','show',o['commit']+':'+o['path']],cwd=root)
        assert hashlib.sha256(raw).hexdigest()==o['sha256']==digest(inputs/name)
    for path,value in config['source_code_sha256'].items():
        raw=subprocess.check_output(['git','show',config['source_sha']+':'+path],cwd=root)
        assert hashlib.sha256(raw).hexdigest()==value==digest(root/path)
    for path,value in lock['baseline_code_sha256'].items():assert config['source_code_sha256'][path]==value
    for name in ('candidate_table','table_frozen','evaluation','summary','config','cache_bindings'):
        assert digest(output/(name+'.json'))==report[name+'_sha256']
    freeze=read(output/'table_frozen.json')
    assert freeze['config_sha256']==digest(output/'config.json') and freeze['source_sha']==config['source_sha']
    assert freeze['candidate_table_sha256']==digest(output/'candidate_table.json') and freeze['cache_bindings_sha256']==digest(output/'cache_bindings.json')
    assert freeze['finalized_utc']<report['completed_utc'] and freeze['canonical_exact']==120
    manifest=read(inputs/'development_manifest.json');oldtable=read(inputs/'accepted_T018_development_table.json')
    ids=[r['image_id'] for r in manifest['images']]
    assert len(ids)==40 and ids==lock['development_ids']==sorted({r['image_id'] for r in oldtable})
    t016=read(inputs/'accepted_T016_config.json');cache_config=read(inputs/'accepted_cache_config.json')
    assert origins['development_manifest.json']['sha256']==t016['manifest_sha256']==cache_config['manifest_sha256']
    for key,name in [('config.json','accepted_cache_config.json'),('artifact_manifest.json','accepted_cache_manifest.json')]:assert t016['source_audit_hashes'][key]==origins[name]['sha256']
    rows=read(output/'evaluation.json');table=read(output/'candidate_table.json');bindings=read(output/'cache_bindings.json');summary=read(output/'summary.json')
    conditions=('clean','homogeneous_dark','homogeneous_bright');coords=[(x,y,0.) for x in (.4,.5,.6) for y in (.4,.5,.6)]
    assert lock['conditions']==list(conditions) and lock['candidate_coordinates']==[list(c) for c in coords] and lock['gains']==[1.,.45,1.55]
    assert len(rows)==len(table)==len(bindings)==120
    expected=[(i,c) for i in ids for c in conditions]
    assert [(r['image_id'],r['condition']) for r in table]==expected
    old={(r['image_id'],r['condition']):r for r in read(inputs/'accepted_cache_manifest.json')}
    images={r['image_id']:r for r in manifest['images']}
    for i,(r,t,b) in enumerate(zip(rows,table,bindings)):
        assert r['row_index']==t['row_index']==b['row_index']==i and all(r[k]==v for k,v in t.items())
        entry=old[expected[i]];assert r['source_directory']==b['source_directory']==entry['directory']
        assert r['source_image_sha256']==images[r['image_id']]['sha256']
        for path,h in b['files_sha256'].items():
            method,name=path.split('/');assert entry['files'][method][name]['sha256']==h
        assert b['canonical_pixels_exact'] and digest(output/f'{i:03d}'/'state.pt')==b['corners_file_sha256']
        v=t['candidate_mse'];assert len(v)==9 and all(x>=0 for x in v)
        cross=[v[j] for j in (4,1,7,3,5)];assert r['cross_mse']==cross
        chosen=[]
        for a,lo,hi in [('x',1,2),('y',3,4)]:
            values=[cross[0],cross[lo],cross[hi]]
            gains=[(values[0]-x)/values[0] for x in values[1:]] if values[0]!=0 else [None,None]
            j=0 if values[0]==0 or max(gains)<.01 else min(range(3),key=lambda k:(values[k],k))
            boundary=(.5,.4,.6)[j];chosen.append(boundary)
            assert r[a]['gains']==gains and r[a]['value']==boundary==r['b'+a]
            assert r[a+'_label']==('center','lower','upper')[j]
        index=coords.index((*chosen,0.));assert r['hard_index']==index
        assert (r['H0'],r['H_delta'],r['H_star'])==(v[4],v[index],min(v))
        assert r['center_in_oracle_tie_set']==(v[4]==min(v))
        move='no_move' if chosen==[.5,.5] else 'x_only' if chosen[1]==.5 else 'y_only' if chosen[0]==.5 else 'both'
        assert r['movement']==move
    clauses={}
    for name in ('nonspatial_pool',*conditions):
        group=rows if name=='nonspatial_pool' else [r for r in rows if r['condition']==name];g=summary['groups'][name]
        m={k:statistics.mean(r[k] for r in group) for k in ('H0','H_delta','H_star')}
        assert g['mse']==m and g['count']==len(group)
        assert g['ratios']=={k+'_over_H0':m[k]/m['H0'] if m['H0'] else None for k in ('H_delta','H_star')}
        assert g['outcomes']==dict(beneficial=sum(r['H_delta']<r['H0'] for r in group),equal=sum(r['H_delta']==r['H0'] for r in group),harmful=sum(r['H_delta']>r['H0'] for r in group))
        assert g['movements']=={k:sum(r['movement']==k for r in group) for k in ('no_move','x_only','y_only','both')}
        assert g['labels']=={a:{k:sum(r[a+'_label']==k for r in group) for k in ('center','lower','upper')} for a in ('x','y')}
        assert g['center_in_oracle_tie_count']==sum(r['H0']==r['H_star'] for r in group)
        assert g['center_in_oracle_tie_fraction']==g['center_in_oracle_tie_count']/len(group)
        assert g['harmful_rows']==[r['row_index'] for r in group if r['H_delta']>r['H0']]
        clauses[name+'_safety']=m['H_delta']<=1.01*m['H0']
    clauses['zero_harmful']=all(r['H_delta']<=r['H0'] for r in rows)
    assert summary['clauses']==clauses and summary['acceptance_vector']==list(clauses.values())
    assert summary['passed']==sum(clauses.values()) and summary['target_viable']==all(clauses.values())
    assert summary['reference_only'] and summary['fresh_qualified'] is False
    result=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),exact_rows=120,development_ids=ids,
        origin_count=len(origins),source_files=len(config['source_code_sha256']),clauses=clauses,reference_only=True,
        verifier='Independent fixed1% arithmetic, exact ties/zero denominators, combined lookup, provenance and summaries; no TTIE imports or rerendering')
    receipt.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print('PASS independent120-row target/provenance/metric verification')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--output',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    verify(a.root,a.output,a.receipt)
