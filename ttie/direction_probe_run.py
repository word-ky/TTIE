"""Fixed five-fold direct-direction probe and a separate post-freeze evaluator."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import torch
from .direction_probe import CLASSES,RECIPE,axis_features,fit_axis_fold
from .energy_local import blob,load_scoring,load_references,join_and_measure,report_groups,HARD,TARGET_COMMIT
from .local_geometry import now,sha,write
from .routing.provenance import verify_source

FOLD_COMMIT='433683ec'
FOLD_PATH='research_log/T016C_run/folds.json'
FOLD_HASH='8fdb03cdac63af5d6e58b557c96679bc15c1e2e921e525916eb9d21ad22cd2c1'
SOURCES=('ttie/__init__.py','ttie/routing/__init__.py','ttie/routing/provenance.py','ttie/local_geometry.py',
         'ttie/hard_local.py','ttie/energy_local.py','ttie/direction_probe.py','ttie/direction_probe_run.py')


def training_inputs():
    scores,hashes=load_scoring();hashes={'score_'+k:v for k,v in hashes.items()}
    full_commit=subprocess.check_output(['git','rev-parse',FOLD_COMMIT]).decode().strip()
    raw=subprocess.check_output(['git','show',full_commit+':'+FOLD_PATH]);assert hashlib.sha256(raw).hexdigest()==FOLD_HASH
    folds=json.loads(raw);hashes['folds']=dict(commit=full_commit,path=FOLD_PATH,sha256=FOLD_HASH)
    donor_config,hashes['fold_provenance']=blob(full_commit,'research_log/T016C_run/config.json')
    assert donor_config['input_artifact_hashes']['selection']['sha256']==hashes['score_selection']['sha256']
    targets,hashes['targets']=blob(TARGET_COMMIT,'research_log/T018A_run/decisions.json')
    target_freeze,hashes['target_freeze']=blob(TARGET_COMMIT,'research_log/T018A_run/decisions_frozen.json')
    assert hashes['targets']['sha256']==target_freeze['decisions_sha256']
    entries=scores['selection']['episodes'];assert len(entries)==len(targets)==120 and len({r['episode'] for r in entries})==120
    schema=scores['config']['schema'];assert schema==donor_config['base_feature_schema'] and schema['dimension']==28 and len(schema['names'])==28
    assert all(len(r['features'])==9 and all(len(v)==28 for v in r['features']) for r in entries)
    assert all(t['row_index']==i and t['hard_index']//3==HARD.index((t['bx'],t['by'],0.)) for i,t in enumerate(targets))
    seen=[]
    for f in folds:
        assert len(f['train'])==96 and len(f['heldout'])==24 and len(f['train_image_ids'])==32 and len(f['heldout_image_ids'])==8
        assert not set(f['train'])&set(f['heldout']) and sorted(f['train']+f['heldout'])==list(range(120))
        assert not set(f['train_image_ids'])&set(f['heldout_image_ids'])
        seen+=f['heldout']
    assert len(folds)==5 and sorted(seen)==list(range(120))
    return scores,targets,folds,raw,hashes


def train_fold(z,targets,fold,output):
    output.mkdir(parents=True);predictions={};receipts={}
    for axis,name in enumerate(('x','y')):
        head,history,logits,classes=fit_axis_fold(z,targets,fold,axis)
        folder=output/name;folder.mkdir();torch.save(dict(state_dict=head.state_dict(),recipe=RECIPE),folder/'head.pt')
        write(folder/'history.json',history)
        predictions[name]=dict(logits=logits,classes=classes)
        write(folder/'predictions.json',dict(row_indices=fold['heldout'],**predictions[name]))
        receipts[name]=dict(finalized_utc=now(),files_sha256={f:sha(folder/f) for f in ('head.pt','history.json','predictions.json')},
            train_rows=fold['train'],heldout_rows=fold['heldout'],train_image_ids=fold['train_image_ids'],heldout_image_ids=fold['heldout_image_ids'],
            epochs=len(history),train_cross_entropy_final=history[-1]['train_cross_entropy'],heldout_target_read=False,
            normalization=dict(mean=head.x_mean.tolist(),scale=head.x_scale.tolist()))
        write(folder/'receipt.json',receipts[name])
    decisions=[]
    for j,i in enumerate(fold['heldout']):
        cx=predictions['x']['classes'][j];cy=predictions['y']['classes'][j];bx=CLASSES[cx];by=CLASSES[cy]
        decisions.append(dict(row_index=i,fold=fold['fold'],x_logits=predictions['x']['logits'][j],y_logits=predictions['y']['logits'][j],
            x_class=cx,y_class=cy,bx=bx,by=by,score_index=HARD.index((bx,by,0.)),hard_index=3*HARD.index((bx,by,0.))))
    write(output/'decisions.json',decisions);write(output/'fold_frozen.json',dict(finalized_utc=now(),decisions_sha256=sha(output/'decisions.json'),heads=receipts))
    return decisions


def freeze_oof(output,decisions):
    decisions.sort(key=lambda d:d['row_index']);assert [d['row_index'] for d in decisions]==list(range(120))
    write(output/'decisions.json',decisions)
    write(output/'OOF_frozen.json',dict(finalized_utc=now(),decisions_sha256=sha(output/'decisions.json'),config_sha256=sha(output/'config.json'),
        folds_sha256=sha(output/'folds.json'),fold_frozen_hashes={str(i):sha(output/f'fold{i}'/'fold_frozen.json') for i in range(5)},
        head_count=10,heldout_reference_evaluated=False))


def train(output,source):
    code=verify_source(source,SOURCES);torch.set_num_threads(1);scores,targets,folds,raw,hashes=training_inputs()
    z=axis_features([r['features'] for r in scores['selection']['episodes']]);output.mkdir(parents=True)
    (output/'folds.json').write_bytes(raw)
    write(output/'config.json',dict(task='T018-C',source_sha=source,source_code_sha256=code,input_artifact_hashes=hashes,
        feature_schema=scores['config']['schema'],feature_construction='concat(f0,fminus-f0,fplus-f0)',recipe=RECIPE,
        runtime=dict(python=platform.python_version(),torch=torch.__version__,device='cpu'),development_only=True))
    write(output/'precheck.json',dict(passed=True,completed_utc=now(),episode_keys_unique=120,target_rows_aligned=120,feature_shape=list(z.shape),
        original_fold_bytes=True,heldout_rows=24,training_rows=96,heldout_ids=8,training_ids=32,reference_metrics_read=False))
    all_decisions=[]
    for fold in folds:
        all_decisions+=train_fold(z,targets,fold,output/f"fold{fold['fold']}")
        print('Fold',fold['fold'],'two final-epoch heads frozen',flush=True)
    freeze_oof(output,all_decisions);print('All OOF decisions frozen',sha(output/'decisions.json'),flush=True)


def confusion(rows):
    return {axis:[[sum(CLASSES.index(r['target_b'+axis])==truth and CLASSES.index(r['b'+axis])==pred for r in rows)
                  for pred in range(3)] for truth in range(3)] for axis in ('x','y')}


def evaluate(output):
    read=lambda name:json.loads((output/(name+'.json')).read_text(encoding='utf-8'))
    config=read('config');verify_source(config['source_sha'],SOURCES);frozen=read('OOF_frozen')
    assert sha(output/'decisions.json')==frozen['decisions_sha256'] and sha(output/'folds.json')==frozen['folds_sha256']
    decisions=read('decisions');opened=now();data,hashes=load_references();scored,score_hashes=load_scoring()
    for k,v in score_hashes.items():assert config['input_artifact_hashes']['score_'+k]==v
    rows=join_and_measure(decisions,scored['selection']['episodes'],data);assert all(r['row_index']==r['reference_row_index'] for r in rows)
    # Recheck actual image identities against the immutable fold assignment only now.
    table=data['candidate_metrics'];folds=read('folds')
    for f in folds:
        assert sorted({table[i]['image_id'] for i in f['train']})==f['train_image_ids']
        assert sorted({table[i]['image_id'] for i in f['heldout']})==f['heldout_image_ids']
    report=report_groups(rows)
    for name,g in report['groups'].items():g['confusion_matrices']=confusion(rows if name=='spatial_pool' else [r for r in rows if r['condition']==name])
    report['interpretation']='direct_direction_probe_viable' if report['passed']==5 else 'direct_direction_probe_insufficient'
    write(output/'summary.json',report);write(output/'evaluation.json',rows)
    write(output/'evaluation_receipt.json',dict(reference_opened_utc=opened,completed_utc=now(),input_artifact_hashes=hashes,
        decisions_sha256=sha(output/'decisions.json'),oof_frozen_sha256=sha(output/'OOF_frozen.json'),exact_episode_identity=120,
        reference_row_order_exact=True,actual_image_id_grouping_exact=True))
    assert sha(output/'decisions.json')==frozen['decisions_sha256'];print(json.dumps({k:report[k] for k in ('clauses','passed','interpretation')}))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','evaluate']);p.add_argument('--source-sha');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.stage=='train':train(a.output,a.source_sha)
    else:evaluate(a.output)


if __name__=='__main__':main()
