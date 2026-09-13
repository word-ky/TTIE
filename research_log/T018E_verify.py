"""Independent T018-E replay: reference-free CPU decisions, then table arithmetic."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,data):Path(path).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
def now():return datetime.now(timezone.utc).isoformat()


def replay(selected, heads, output):
    import torch
    from torch import nn
    torch.set_num_threads(1);opened=[]
    def get(path):opened.append(str(path));return read(path)
    freeze=get(selected/'decisions_frozen.json');features=get(selected/'features_frozen.json')
    assert digest(selected/'features_frozen.json')==freeze['features_frozen_sha256']
    assert digest(selected/'decisions.json')==freeze['decisions_sha256']
    assert digest(selected/'features.json')==features['features_sha256']==freeze['features_sha256']
    assert features['finalized_utc']<freeze['finalized_utc']
    rows=get(selected/'decisions.json');original=get(selected/'features.json');receipt=get(heads/'selector_frozen.json')
    assert digest(heads/'selector_frozen.json')==freeze['selector_receipt_sha256']=='db194f4caa897094655a36523fa074772d8cd7302ffda250a72e8b81cc2f9d94'
    assert len(rows)==len(original)==120 and [r['row_index'] for r in rows]==list(range(120))
    assert all(all(r[k]==v for k,v in o.items()) for r,o in zip(rows,original))
    f=torch.tensor([r['features'] for r in rows],dtype=torch.float32);assert f.shape==(120,5,28)
    result={}
    for axis,lo,hi in (('x',1,2),('y',3,4)):
        path=heads/('head_'+axis+'.pt');opened.append(str(path));raw=path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==receipt['files_sha256'][path.name]
        state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)['state_dict']
        assert receipt['normalization'][axis]==dict(mean=state['x_mean'].tolist(),scale=state['x_scale'].tolist())
        z=torch.cat((f[:,0],f[:,lo]-f[:,0],f[:,hi]-f[:,0]),1)
        net=nn.Sequential(nn.Linear(84,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,3)).eval()
        net.load_state_dict({k[4:]:v for k,v in state.items() if k.startswith('net.')})
        with torch.no_grad():logits=net((z-state['x_mean'])/state['x_scale'])
        result[axis]=(logits.tolist(),logits.argmax(1).tolist())
    classes=(.5,.4,.6);hard=[(x,y) for x in (.4,.5,.6) for y in (.4,.5,.6)]
    for i,r in enumerate(rows):
        for axis in ('x','y'):
            assert r[axis+'_logits']==result[axis][0][i] and r[axis+'_class']==result[axis][1][i]
        bx=classes[r['x_class']];by=classes[r['y_class']];index=hard.index((bx,by))
        assert (r['bx'],r['by'],r['score_index'],r['hard_index'])==(bx,by,index,3*index)
    save(output,dict(passed=True,completed_utc=now(),exact_logits_classes_decisions=120,reference_reads=False,
        ttie_imports=False,normalization_refit=False,opened_files=opened,decisions_sha256=freeze['decisions_sha256'],torch_version=torch.__version__))
    print('PASS independent reference-free exact120 decision replay')


def metrics(root, cohort, selected, evaluation, replay_receipt, output):
    # Independent check: the decision pins config, config pins preparation,
    # and preparation pins the original mapping and opaque input index.
    frozen=read(selected/'decisions_frozen.json')
    assert digest(selected/'config.json')==frozen['config_sha256']
    config=read(selected/'config.json')
    assert 'prepared_sha256' in config, 'Missing pre-inference prepared binding; historical receipts cannot be upgraded retroactively'
    assert digest(cohort/'prepared.json')==config['prepared_sha256']
    prepared=read(cohort/'prepared.json')
    assert digest(cohort/'mapping.json')==prepared['mapping_sha256']
    assert digest(cohort/'inputs/index.json')==prepared['inputs_sha256']==config['input_index_sha256']
    prior=read(root/'research_log/T018E_exclusions.json');manifest=read(cohort/'manifest.json');mf=read(cohort/'manifest_frozen.json')
    union=set()
    for entry in prior['prior_manifest_bindings']:
        raw=subprocess.check_output(['git','show','HEAD:'+entry['path']],cwd=root)
        assert hashlib.sha256(raw).hexdigest()==entry['sha256']
        found=set()
        def visit(v):
            if isinstance(v,dict):
                for k,x in v.items():
                    if k=='image_id' and isinstance(x,int):found.add(x)
                    elif k in ('image_ids','excluded_prior_ids') and isinstance(x,list):found.update(i for i in x if isinstance(i,int))
                    else:visit(x)
            elif isinstance(v,list):
                for x in v:visit(x)
        visit(json.loads(raw));assert found==set(entry['ids']);union.update(found)
    assert union==set(prior['excluded_ids']) and set(prior['development_T016_T018_ids'])<=union
    ids=[r['image_id'] for r in manifest['images']]
    assert len(ids)==len(set(ids))==40 and not set(ids)&union and ids==sorted(ids)
    assert ids==[r['image_id'] for r in manifest['inspected_prefix'] if r['eligible']][:40]
    assert all(min(r['width'],r['height'])>=320 for r in manifest['images'])
    assert digest(cohort/'manifest.json')==mf['manifest_sha256'] and digest(root/'research_log/T018E_exclusions.json')==mf['exclusions_sha256']
    freeze=read(selected/'decisions_frozen.json');er=read(evaluation/'evaluation_receipt.json');a=read(replay_receipt)
    assert mf['finalized_utc']<read(selected/'features_frozen.json')['started_utc']<freeze['finalized_utc']<a['completed_utc']<er['reference_opened_utc']
    assert a['passed'] and a['decisions_sha256']==freeze['decisions_sha256']==digest(selected/'decisions.json')
    assert digest(selected/'decisions_frozen.json')==er['decision_freeze_sha256']
    assert digest(evaluation/'evaluation.json')==er['evaluation_sha256'] and digest(evaluation/'summary.json')==er['summary_sha256']
    decisions=read(selected/'decisions.json');rows=read(evaluation/'evaluation.json');summary=read(evaluation/'summary.json');mapping=read(cohort/'mapping.json')
    assert len(rows)==len(mapping)==len(decisions)==120
    families=('left_right','quadrants','offset_left_right_40')
    assert {(r['image_id'],r['condition']) for r in rows}=={(i,c) for i in ids for c in families}
    for r,d,m in zip(rows,decisions,mapping):
        assert all(r[k]==v for k,v in m.items()) and r['row_index']==d['row_index']
        mse=r['candidate_mse'];assert len(mse)==9
        assert (r['bx'],r['by'],r['score_index'])==(d['bx'],d['by'],d['score_index'])
        assert (r['H0'],r['H1'],r['H_star'])==(mse[4],mse[d['score_index']],min(mse))
        assert r['delta']==r['H1']-r['H0'] and r['H1_over_H0']==r['H1']/r['H0'] and r['H1_over_H_star']==r['H1']/r['H_star']
    means={}
    for name in ('spatial_pool',*families):
        group=rows if name=='spatial_pool' else [r for r in rows if r['condition']==name]
        means[name]={k:statistics.mean(r[k] for r in group) for k in ('H0','H1','H_star')}
        g=summary['groups'][name];assert g['mse']==means[name] and g['count']==len(group)
        assert g['ratios']==dict(H1_over_H0=means[name]['H1']/means[name]['H0'],H1_over_H_star=means[name]['H1']/means[name]['H_star'])
        assert g['outcomes']==dict(beneficial=sum(r['H1']<r['H0'] for r in group),equal=sum(r['H1']==r['H0'] for r in group),harmful=sum(r['H1']>r['H0'] for r in group))
        assert g['harmful_rows']==[r['row_index'] for r in group if r['H1']>r['H0']]
    p=means['spatial_pool'];l=means['left_right'];q=means['quadrants'];o=means['offset_left_right_40']
    clauses=dict(pooled_gain=p['H1']<=.97*p['H0'],pooled_oracle=p['H1']<=1.03*p['H_star'],offset_gain=o['H1']<=.95*o['H0'],left_right_safety=l['H1']<=1.01*l['H0'],quadrants_safety=q['H1']<=1.01*q['H0'])
    assert summary['clauses']==clauses and summary['passed']==sum(clauses.values()) and summary['fresh_qualified']==all(clauses.values())
    save(output,dict(passed=True,completed_utc=now(),prior_manifests=len(prior['prior_manifest_bindings']),excluded_ids=len(union),fresh_ids=ids,
        disjoint=True,freeze_before_reference=True,exact_metric_rows=120,clauses=clauses,gate_count=sum(clauses.values()),
        metric_verification='Independent nine-candidate table indexing and aggregation; not a second pixel/feature run.'))
    print('PASS independent provenance,120 metric rows and5 literal clauses')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['replay','metrics'])
    for name in ('selected','heads','output','root','cohort','evaluation','replay-receipt'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.phase=='replay':replay(a.selected,a.heads,a.output)
    else:metrics(a.root,a.cohort,a.selected,a.evaluation,a.replay_receipt,a.output)
