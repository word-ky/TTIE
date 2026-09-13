"""T018-B: saved-energy decisions first; reference evaluation in a separate process."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
from .local_geometry import CANDIDATES,local_choice,now,sha,write,ratio
from .routing.provenance import verify_source

SCORE_COMMIT='4062e01cb93de731c394015c5ac741d6c08e04d8'
SCORE_BASE='research_log/remote_runs/20260913-013748-ttie-t016b-assets-ready/artifacts/scoring/'
TARGET_COMMIT='5ecf598763c499b2275994be53f9218a3757245c'
HARD=tuple((x,y,0.) for x in (.4,.5,.6) for y in (.4,.5,.6))
CROSS=(4,1,7,3,5)
SOURCES=('ttie/__init__.py','ttie/routing/__init__.py','ttie/routing/provenance.py','ttie/local_geometry.py','ttie/hard_local.py','ttie/energy_local.py')


def blob(commit,path):
    raw=subprocess.check_output(['git','show',commit+':'+path])
    return json.loads(raw),dict(commit=commit,path=path,sha256=hashlib.sha256(raw).hexdigest())


def load_scoring():
    data={};hashes={}
    for name in ('selection_receipt','config','selection'):
        data[name],hashes[name]=blob(SCORE_COMMIT,SCORE_BASE+name+'.json')
    assert hashes['selection']['sha256']==data['selection_receipt']['selection_sha256']
    assert hashes['config']['sha256']==data['selection_receipt']['config_sha256']
    assert data['selection_receipt']['reference_access'] is False
    assert data['config']['candidates']==data['selection']['candidates']==[list(c) for c in HARD]
    assert len(data['selection']['episodes'])==120 and all(len(r['energies'])==9 for r in data['selection']['episodes'])
    return data,hashes


def choose(episodes):
    # Only energies is accessed; original episode/path, gate, features and global choices are ignored.
    decisions=[]
    for i,row in enumerate(episodes):
        cross=[row['energies'][j] for j in CROSS];choice=local_choice(cross)
        decisions.append(dict(row_index=i,bx=choice['bx'],by=choice['by'],hard_index=choice['hard_index'],
            score_index=choice['hard_index']//3,cross_energies=cross))
    return decisions


def select(output,source):
    code=verify_source(source,SOURCES);data,hashes=load_scoring();decisions=choose(data['selection']['episodes'])
    output.mkdir(parents=True)
    write(output/'config.json',dict(task='T018-B',source_sha=source,source_code_sha256=code,score_inputs=hashes,
        candidates=HARD,cross_indices=CROSS,tie_order=[.5,.4,.6],runtime=dict(python=platform.python_version(),device='cpu')))
    write(output/'decisions.json',decisions)
    write(output/'decisions_frozen.json',dict(finalized_utc=now(),decisions_sha256=sha(output/'decisions.json'),config_sha256=sha(output/'config.json'),
        count=len(decisions),reference_access=False,family_or_image_id_access=False))
    print('Frozen 120 energy-only decisions: '+sha(output/'decisions.json'))


def load_references():
    data={};hashes={}
    for name in ('config','decisions','decisions_frozen','quantities','quantities_frozen','report_receipt'):
        data['target_'+name],hashes['target_'+name]=blob(TARGET_COMMIT,'research_log/T018A_run/'+name+'.json')
    for name,item in data['target_config']['input_artifact_hashes'].items():
        data[name],hashes[name]=blob(item['commit'],item['path']);assert hashes[name]==item
    assert hashes['target_decisions']['sha256']==data['target_decisions_frozen']['decisions_sha256']==data['target_report_receipt']['decisions_sha256']
    assert hashes['target_quantities']['sha256']==data['target_quantities_frozen']['quantities_sha256']==data['target_report_receipt']['quantities_sha256']
    return data,hashes


def join_and_measure(decisions,scored,data):
    table=data['candidate_metrics'];grid=data['config']['candidates']
    assert len(table)==len(scored)==len(decisions)==120 and grid==[list(c) for c in CANDIDATES]
    assert [grid[i] for i in range(0,27,3)]==[list(c) for c in HARD]
    reference={r['source_directory']:i for i,r in enumerate(table)}
    assert len(reference)==120 and len({r['episode'] for r in scored})==120
    assert set(reference)=={r['episode'] for r in scored}
    rows=[]
    for d,s in zip(decisions,scored):
        index=reference[s['episode']];t=table[index];target=data['target_decisions'][index];prior=data['target_quantities'][index]
        assert s['corners_sha256']==t['corners_sha256']
        mse=t['candidate_mse'];h0=mse[12];hs=mse[d['hard_index']];best=min(mse[::3])
        assert prior['H0']==h0 and prior['H_star']==best and prior['H1']==mse[target['hard_index']]
        assert target['row_index']==index
        x=d['bx']!=.5;y=d['by']!=.5
        rows.append(dict(row_index=d['row_index'],reference_row_index=index,bx=d['bx'],by=d['by'],hard_index=d['hard_index'],
            target_bx=target['bx'],target_by=target['by'],H0=h0,Hselected=hs,H_star=best,
            x_match=d['bx']==target['bx'],y_match=d['by']==target['by'],joint_match=(d['bx'],d['by'])==(target['bx'],target['by']),
            movement='both' if x and y else 'x_only' if x else 'y_only' if y else 'no_move',condition=t['condition']))
    return rows


def report_groups(rows):
    from .hard_local import qualify
    groups={}
    for name in ('spatial_pool','left_right','quadrants','offset_left_right_40'):
        group=rows if name=='spatial_pool' else [r for r in rows if r['condition']==name]
        m={k:statistics.mean(r[k] for r in group) for k in ('H0','Hselected','H_star')}
        groups[name]=dict(count=len(group),mse=m,ratios=dict(selected_over_H0=ratio(m['Hselected'],m['H0']),selected_over_H_star=ratio(m['Hselected'],m['H_star'])),
            target_agreement={k:dict(count=sum(r[k] for r in group),fraction=statistics.mean(r[k] for r in group)) for k in ('x_match','y_match','joint_match')},
            movements={k:sum(r['movement']==k for r in group) for k in ('no_move','x_only','y_only','both')},
            gains=dict(beneficial=sum(r['Hselected']<r['H0'] for r in group),equal=sum(r['Hselected']==r['H0'] for r in group),harmful=sum(r['Hselected']>r['H0'] for r in group)),
            harmful_examples=[{k:r[k] for k in ('row_index','bx','by','target_bx','target_by','H0','Hselected','H_star')} for r in group if r['Hselected']>r['H0']])
    result=qualify({k:dict(mse=dict(H0=v['mse']['H0'],H1=v['mse']['Hselected'],H_star=v['mse']['H_star'])) for k,v in groups.items()})
    result['interpretation']='frozen_energy_local_signal_viable' if result['passed']==5 else 'frozen_energy_local_signal_insufficient'
    return dict(groups=groups,**result,label_free_decision=True,development_only=True,fresh_qualified=False)


def evaluate(output):
    read=lambda name:json.loads((output/(name+'.json')).read_text(encoding='utf-8'))
    config=read('config');verify_source(config['source_sha'],SOURCES);frozen=read('decisions_frozen')
    assert sha(output/'decisions.json')==frozen['decisions_sha256'] and sha(output/'config.json')==frozen['config_sha256']
    decisions=read('decisions');opened=now()
    data,hashes=load_references();scoring,score_hashes=load_scoring();assert score_hashes==config['score_inputs']
    rows=join_and_measure(decisions,scoring['selection']['episodes'],data)
    write(output/'reference_precheck.json',dict(passed=True,completed_utc=now(),episode_identities_exact=120,corners_exact=120,candidates_exact=9,
        target_H0_H1_oracle_exact=120,input_artifact_hashes=hashes))
    report=report_groups(rows);write(output/'summary.json',report);write(output/'evaluation.json',rows)
    assert sha(output/'decisions.json')==frozen['decisions_sha256']
    write(output/'evaluation_receipt.json',dict(reference_opened_utc=opened,decisions_sha256=sha(output/'decisions.json'),
        decisions_frozen_sha256=sha(output/'decisions_frozen.json'),completed_utc=now()))
    print(json.dumps({k:report[k] for k in ('clauses','passed','interpretation')}))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','evaluate']);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-sha');a=p.parse_args()
    if a.stage=='select':select(a.output,a.source_sha)
    else:evaluate(a.output)


if __name__=='__main__':main()
