"""Independent T020-A replay: reference-free CPU decisions, then table arithmetic."""
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
    assert digest(heads/'selector_frozen.json')==freeze['selector_receipt_sha256']=='0367456d7b4f235f987339baf07e343f86f870d1e109adbe461297c82b641c77'
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
    bindings=dict(exclusions_sha256=root/'research_log/T020A_exclusions.json',manifest_sha256=cohort/'manifest.json',
        manifest_frozen_sha256=cohort/'manifest_frozen.json',mapping_sha256=cohort/'mapping.json',
        input_index_sha256=cohort/'inputs/index.json',prepared_sha256=cohort/'prepared.json')
    for key,path in bindings.items():assert key in config and digest(path)==config[key], 'Missing or changed '+key
    lock=read(root/'research_log/T020A_pipeline_lock.json')
    assert digest(root/'research_log/T020A_pipeline_lock.json')==config['pipeline_lock_sha256']
    assert frozen['selector_receipt_sha256']==lock['selector_receipt_sha256']=='0367456d7b4f235f987339baf07e343f86f870d1e109adbe461297c82b641c77'
    for path,value in frozen['source_code_sha256'].items():
        raw=subprocess.check_output(['git','show',frozen['source_sha']+':'+path],cwd=root)
        assert hashlib.sha256(raw).hexdigest()==value==digest(root/path)
    for path,value in lock['baseline_code_sha256'].items():assert frozen['source_code_sha256'][path]==value
    prior=read(root/'research_log/T020A_exclusions.json');manifest=read(cohort/'manifest.json');mf=read(cohort/'manifest_frozen.json')
    union=set()
    for entry in prior['prior_manifest_bindings']:
        raw=subprocess.check_output(['git','show',entry['commit']+':'+entry['path']],cwd=root)
        assert hashlib.sha256(raw).hexdigest()==entry['sha256']
        found=set()
        def visit(v):
            if isinstance(v,dict):
                for k,x in v.items():
                    if k=='image_id' and isinstance(x,int):found.add(x)
                    elif k in ('image_ids','excluded_prior_ids','excluded_ids','used_ids','previously_inspected_extra_ids','development_T016_T018_ids','train_image_ids','heldout_image_ids','T018E_used_ids','T019D_used_ids') and isinstance(x,list):found.update(i for i in x if isinstance(i,int))
                    else:visit(x)
            elif isinstance(v,list):
                for x in v:visit(x)
        visit(json.loads(raw));assert found==set(entry['ids']);union.update(found)
    assert union==set(prior['excluded_ids']) and set(prior['development_T016_T018_ids'])<=union
    ids=[r['image_id'] for r in manifest['images']]
    assert len(ids)==len(set(ids))==40 and not set(ids)&union and ids==sorted(ids)
    assert ids==[r['image_id'] for r in manifest['inspected_prefix'] if r['eligible']][:40]
    assert all(min(r['width'],r['height'])>=320 for r in manifest['images'])
    assert digest(cohort/'manifest.json')==mf['manifest_sha256'] and digest(root/'research_log/T020A_exclusions.json')==mf['exclusions_sha256']
    assert set(prior['T018E_used_ids'])<=union and len(prior['T018E_used_ids'])==40
    assert set(prior['T019D_used_ids'])<=union and len(prior['T019D_used_ids'])==40
    assert set(prior['previously_inspected_extra_ids'])<=union
    assert manifest['excluded_prior_ids']==sorted(union)
    freeze=read(selected/'decisions_frozen.json');er=read(evaluation/'evaluation_receipt.json');a=read(replay_receipt)
    assert mf['finalized_utc']<read(selected/'features_frozen.json')['started_utc']<freeze['finalized_utc']<a['completed_utc']<er['reference_opened_utc']
    feature_freeze=read(selected/'features_frozen.json')
    assert prior['frozen_utc']<mf['finalized_utc']<prepared['completed_utc']<feature_freeze['started_utc']<=feature_freeze['finalized_utc']<freeze['finalized_utc']
    assert mf['source_sha']==freeze['source_sha']==config['source_sha']
    assert mf['source_code_sha256']==prepared['source_code_sha256']==freeze['source_code_sha256']==config['source_code_sha256']==er['source_code_sha256']
    assert digest(selected/'features_frozen.json')==freeze['features_frozen_sha256']
    assert digest(selected/'features.json')==feature_freeze['features_sha256']==freeze['features_sha256']
    assert digest(replay_receipt)==er['replay_receipt_sha256']
    assert a['passed'] and a['decisions_sha256']==freeze['decisions_sha256']==digest(selected/'decisions.json')
    assert digest(selected/'decisions_frozen.json')==er['decision_freeze_sha256']
    assert digest(evaluation/'evaluation.json')==er['evaluation_sha256'] and digest(evaluation/'summary.json')==er['summary_sha256']
    decisions=read(selected/'decisions.json');rows=read(evaluation/'evaluation.json');summary=read(evaluation/'summary.json');mapping=read(cohort/'mapping.json')
    assert len(rows)==len(mapping)==len(decisions)==120
    index=read(cohort/'inputs/index.json')
    assert index['manifest_frozen_sha256']==config['manifest_frozen_sha256']
    assert index['manifest_sha256']==prepared['manifest_sha256']==config['manifest_sha256']
    assert prepared['mapping_sha256']==config['mapping_sha256'] and prepared['source_code_sha256']==config['source_code_sha256']
    expected_mapping=[dict(row_index=i*3+j,image_id=e['image_id'],filename=e['filename'],condition=c,source_sha256=e['sha256'])
        for i,e in enumerate(manifest['images']) for j,c in enumerate(('clean','homogeneous_dark','homogeneous_bright'))]
    assert mapping==expected_mapping
    for d,item in zip(decisions,index['episodes']):
        assert d['row_index']==item['row_index'] and d['input_file_sha256']==item['sha256'] and d['pixels_sha256']==item['pixels_sha256']
        for filename,value in freeze['case_files_sha256'][str(d['row_index'])].items():assert digest(selected/f"{d['row_index']:03d}"/filename)==value
    families=('clean','homogeneous_dark','homogeneous_bright')
    assert manifest['conditions']==list(families)==lock['nonspatial_conditions']
    assert {(r['image_id'],r['condition']) for r in rows}=={(i,c) for i in ids for c in families}
    for r,d,m in zip(rows,decisions,mapping):
        assert all(r[k]==v for k,v in m.items()) and r['row_index']==d['row_index']
        assert (r['bx'],r['by'],r['score_index'])==(d['bx'],d['by'],d['score_index'])
        assert 'candidate_mse' not in r and 'H_star' not in r
        assert r['delta']==r['H1']-r['H0'] and r['H1_over_H0']==(r['H1']/r['H0'] if r['H0']!=0 else None)
        assert r['movement']==('no_move' if d['bx']==.5 and d['by']==.5 else 'x_only' if d['by']==.5 else 'y_only' if d['bx']==.5 else 'both')
    means={}
    for name in ('nonspatial_pool',*families):
        group=rows if name=='nonspatial_pool' else [r for r in rows if r['condition']==name]
        means[name]={k:statistics.mean(r[k] for r in group) for k in ('H0','H1')}
        g=summary['groups'][name];assert g['mse']==means[name] and g['count']==len(group)
        assert g['ratios']==dict(H1_over_H0=means[name]['H1']/means[name]['H0'] if means[name]['H0']!=0 else None)
        assert g['outcomes']==dict(beneficial=sum(r['H1']<r['H0'] for r in group),equal=sum(r['H1']==r['H0'] for r in group),harmful=sum(r['H1']>r['H0'] for r in group))
        assert g['harmful_rows']==[r['row_index'] for r in group if r['H1']>r['H0']]
        assert g['movements']=={key:sum(r['movement']==key for r in group) for key in ('no_move','x_only','y_only','both')}
    clauses={name+'_safety':m['H1']<=1.01*m['H0'] for name,m in means.items()}
    for k in ('H0','H1'):
        values=sorted(r[k] for r in rows if r['condition']=='clean')
        # Exactly40 rows: linear95th percentile lies0.05 of the way from index37 to38.
        position=.95*39;percentile=values[37]+(position-37)*(values[38]-values[37])
        assert summary['groups']['clean']['absolute_mse'][k]==dict(mean=means['clean'][k],p95=percentile)
    assert summary['clauses']==clauses and summary['passed']==sum(clauses.values()) and summary['fresh_qualified']==all(clauses.values())
    save(output,dict(passed=True,completed_utc=now(),prior_manifests=len(prior['prior_manifest_bindings']),excluded_ids=len(union),fresh_ids=ids,
        disjoint=True,freeze_before_reference=True,exact_metric_rows=120,clauses=clauses,gate_count=sum(clauses.values()),
        metric_verification='Independent two-output table aggregation and clean mean/p95; not a second pixel/feature run.'))
    print('PASS independent provenance,120 metric rows and4 literal clauses')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['replay','metrics'])
    for name in ('selected','heads','output','root','cohort','evaluation','replay-receipt'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.phase=='replay':replay(a.selected,a.heads,a.output)
    else:metrics(a.root,a.cohort,a.selected,a.evaluation,a.replay_receipt,a.output)
