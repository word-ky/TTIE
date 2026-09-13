"""Independent two-stage verifier. Replay opens no reference/target artifact.

Run replay BEFORE evaluation; metrics is a separate invocation after evaluation.
No TTIE implementation is imported. Git origins must be available locally.
"""
import argparse
import hashlib
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

digest=lambda raw: hashlib.sha256(raw).hexdigest()
filehash=lambda path: digest(path.read_bytes())
now=lambda: datetime.now(timezone.utc).isoformat()
classes=[.5,.4,.6]
hard=[[x,y,0.] for x in (.4,.5,.6) for y in (.4,.5,.6)]


def origin(item):
    raw=subprocess.check_output(['git','show',item['commit']+':'+item['path']])
    assert digest(raw)==item['sha256']
    return json.loads(raw)


def save(path, value): path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def frozen(root):
    read=lambda name: json.loads((root/(name+'.json')).read_text(encoding='utf-8'))
    freeze=read('OOF_frozen')
    for name in ('decisions','config','folds'):
        assert filehash(root/(name+'.json'))==freeze[name+'_sha256']
    return read,read('config'),freeze


def replay(root):
    import torch
    from torch import nn
    torch.set_num_threads(1)
    read,cfg,freeze=frozen(root)
    # Explicit feature/fold-only allowlist. Never iterate all input origins:
    # that would open held-out labels in the reference-free replay process.
    origins=cfg['input_artifact_hashes']
    data={k:origin(origins[k]) for k in ('score_selection_receipt','score_config','score_selection','folds','fold_provenance','donor_config')}
    assert cfg['recipe']==data['donor_config']['recipe']
    assert cfg['feature_schema']==data['score_config']['schema']==data['fold_provenance']['base_feature_schema']
    assert data['score_selection_receipt']['reference_access'] is False
    assert data['score_selection_receipt']['selection_sha256']==origins['score_selection']['sha256']
    assert data['score_selection_receipt']['config_sha256']==origins['score_config']['sha256']
    for k in ('score_selection_receipt','score_config','score_selection','folds','fold_provenance'):
        assert origins[k]==data['donor_config']['input_artifact_hashes'][k]
    assert read('folds')==data['folds']
    for path, expected in cfg['source_code_sha256'].items():
        assert filehash(Path(path))==expected==digest(subprocess.check_output(['git','show',cfg['source_sha']+':'+path]))
    for path, expected in data['donor_config']['source_code_sha256'].items():
        assert cfg['source_code_sha256'][path]==expected
    f=torch.tensor([r['features'] for r in data['score_selection']['episodes']],dtype=torch.float32)
    assert f.shape==(120,9,28)
    z={'x':torch.cat([f[:,4],f[:,1]-f[:,4],f[:,7]-f[:,4]],1),
       'y':torch.cat([f[:,4],f[:,3]-f[:,4],f[:,5]-f[:,4]],1)}
    reconstructed={};seen=[];heads=0
    for fold in data['folds']:
        train=fold['train'];heldout=fold['heldout'];number=fold['fold'];seen+=heldout
        assert len(train)==96 and len(heldout)==24 and set(train).isdisjoint(heldout)
        assert sorted(train+heldout)==list(range(120))
        assert len(fold['train_image_ids'])==32 and len(fold['heldout_image_ids'])==8
        assert set(fold['train_image_ids']).isdisjoint(fold['heldout_image_ids'])
        folder=root/f'fold{number}'
        fr=json.loads((folder/'fold_frozen.json').read_text(encoding='utf-8'))
        assert filehash(folder/'fold_frozen.json')==freeze['fold_frozen_hashes'][str(number)]
        assert filehash(folder/'decisions.json')==fr['decisions_sha256']
        assert fr['finalized_utc']<=freeze['finalized_utc']
        predictions={}
        for axis in ('x','y'):
            sub=folder/axis;heads+=1;r=json.loads((sub/'receipt.json').read_text(encoding='utf-8'))
            assert r==fr['heads'][axis]
            for filename,expected in r['files_sha256'].items(): assert filehash(sub/filename)==expected
            assert r['target_artifact']==origins['targets']
            assert r['train_rows']==train and r['heldout_rows']==heldout
            assert r['train_image_ids']==fold['train_image_ids'] and r['heldout_image_ids']==fold['heldout_image_ids']
            assert r['heldout_target_read'] is False and r['epochs']==100
            history=json.loads((sub/'history.json').read_text(encoding='utf-8'))
            assert [h['epoch'] for h in history]==list(range(1,101))
            assert r['train_cross_entropy_final']==history[-1]['train_cross_entropy']
            checkpoint=torch.load(sub/'head.pt',map_location='cpu',weights_only=True)
            assert checkpoint['recipe']==cfg['recipe'];state=checkpoint['state_dict']
            mean=z[axis][train].double().mean(0).float()
            scale=z[axis][train].double().std(0,unbiased=False).clamp_min(1e-12).float()
            assert torch.equal(mean,state['x_mean']) and torch.equal(scale,state['x_scale'])
            norm=dict(mean=mean.tolist(),scale=scale.tolist())
            assert r['normalization']==norm==json.loads((sub/'normalization.json').read_text(encoding='utf-8'))
            model=nn.Sequential(nn.Linear(84,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,3)).eval()
            model.load_state_dict({k.removeprefix('net.'):v for k,v in state.items() if k.startswith('net.')})
            with torch.no_grad(): logits=model((z[axis][heldout]-mean)/scale)
            predictions[axis]=dict(row_indices=heldout,logits=logits.tolist(),classes=logits.argmax(1).tolist())
            assert predictions[axis]==json.loads((sub/'predictions.json').read_text(encoding='utf-8'))
        fd=[]
        for j,i in enumerate(heldout):
            cx=predictions['x']['classes'][j];cy=predictions['y']['classes'][j];x=classes[cx];y=classes[cy];index=hard.index([x,y,0.])
            d=dict(row_index=i,fold=number,x_logits=predictions['x']['logits'][j],y_logits=predictions['y']['logits'][j],
                x_class=cx,y_class=cy,bx=x,by=y,score_index=index,hard_index=index*3)
            reconstructed[i]=d;fd.append(d)
        assert fd==json.loads((folder/'decisions.json').read_text(encoding='utf-8'))
    assert heads==freeze['head_count']==10 and sorted(seen)==list(range(120))
    assert read('decisions')==[reconstructed[i] for i in range(120)]
    result=dict(passed=True,completed_utc=now(),heads=10,exact_oof_logits_classes_choices=120,
        reference_targets_decoded=False,reference_metrics_read=False,train_only_normalization_exact=True,
        donor_source_recipe_features_folds_unchanged=True,decisions_sha256=filehash(root/'decisions.json'),
        oof_frozen_sha256=filehash(root/'OOF_frozen.json'))
    save(root/'head_replay.json',result);print(json.dumps(result))


def metrics(root):
    read,cfg,freeze=frozen(root);receipt=read('evaluation_receipt');replayed=read('head_replay')
    assert freeze['finalized_utc']<replayed['completed_utc']<receipt['reference_opened_utc']
    assert receipt['head_replay_sha256']==filehash(root/'head_replay.json')
    assert receipt['oof_frozen_sha256']==filehash(root/'OOF_frozen.json')
    assert receipt['decisions_sha256']==replayed['decisions_sha256']==freeze['decisions_sha256']
    refs={k:origin(v) for k,v in receipt['input_artifact_hashes'].items()}
    targets=refs['deadband_targets'];table=refs['candidate_metrics'];decisions=read('decisions')
    scores=origin(cfg['input_artifact_hashes']['score_selection'])['episodes']
    assert receipt['input_artifact_hashes']['deadband_targets']==cfg['input_artifact_hashes']['targets']
    for fold in read('folds'):
        for part in ('train','heldout'):
            assert sorted({table[i]['image_id'] for i in fold[part]})==fold[part+'_image_ids']
        expected={str(i):{a:targets[i][a] for a in ('bx','by')} for i in fold['train']}
        assert expected==json.loads((root/f"fold{fold['fold']}"/'train_targets.json').read_text(encoding='utf-8'))
        for a in ('x','y'):
            r=json.loads((root/f"fold{fold['fold']}"/a/'receipt.json').read_text(encoding='utf-8'))
            assert r['train_targets_sha256']==filehash(root/f"fold{fold['fold']}"/'train_targets.json')
    evaluated=[]
    for i,(d,t,target,s) in enumerate(zip(decisions,table,targets,scores)):
        assert d['row_index']==target['row_index']==i
        assert s['episode']==t['source_directory'] and s['corners_sha256']==t['corners_sha256']
        x=d['bx'];y=d['by'];values=t['candidate_mse']
        evaluated.append(dict(row_index=i,reference_row_index=i,bx=x,by=y,hard_index=d['hard_index'],
            target_bx=target['bx'],target_by=target['by'],H0=values[12],Hselected=values[3*hard.index([x,y,0.])],H_star=min(values[::3]),
            x_match=x==target['bx'],y_match=y==target['by'],joint_match=(x,y)==(target['bx'],target['by']),
            movement='no_move' if x==.5 and y==.5 else 'x_only' if y==.5 else 'y_only' if x==.5 else 'both',condition=t['condition']))
    assert len(evaluated)==120 and evaluated==read('evaluation')
    summary=read('summary');detail=read('diagnostics')
    for name,g in summary['groups'].items():
        rows=[r for r in evaluated if name=='spatial_pool' or r['condition']==name]
        m={k:statistics.mean(r[k] for r in rows) for k in ('H0','Hselected','H_star')}
        assert m==g['mse'] and g['count']==len(rows)
        assert g['ratios']==dict(selected_over_H0=m['Hselected']/m['H0'],selected_over_H_star=m['Hselected']/m['H_star'])
        assert g['target_agreement']=={k:dict(count=sum(r[k] for r in rows),fraction=sum(r[k] for r in rows)/len(rows)) for k in ('x_match','y_match','joint_match')}
        assert g['movements']=={k:sum(r['movement']==k for r in rows) for k in ('no_move','x_only','y_only','both')}
        assert g['gains']==dict(beneficial=sum(r['Hselected']<r['H0'] for r in rows),equal=sum(r['Hselected']==r['H0'] for r in rows),harmful=sum(r['Hselected']>r['H0'] for r in rows))
        assert g['harmful_examples']==[{k:r[k] for k in ('row_index','bx','by','target_bx','target_by','H0','Hselected','H_star')} for r in rows if r['Hselected']>r['H0']]
        for a in ('x','y'):
            assert g['confusion_matrices'][a]==[[sum(r['target_b'+a]==t and r['b'+a]==p for r in rows) for p in classes] for t in classes]
            assert detail['groups'][name]['predicted_classes'][a]=={str(v):sum(r['b'+a]==v for r in rows) for v in classes}
            assert detail['groups'][name]['noncenter_on_center_target'][a]==sum(r['target_b'+a]==.5 and r['b'+a]!=.5 for r in rows)
        assert detail['groups'][name]['predicted_joint']=={f'{x},{y}':sum((r['bx'],r['by'])==(x,y) for r in rows) for x in classes for y in classes}
        for k in ('ratios','gains','movements'):
            assert detail['T018C_comparison'][name][k]==dict(T018C=refs['T018C_summary']['groups'][name][k],T019B=g[k])
    suppressed=[dict(row_index=r['row_index'],axis=a,prediction=r['b'+a],condition=r['condition'],original_target=targets[r['row_index']]['original_b'+a])
        for r in evaluated for a in ('x','y') if r['target_b'+a]==.5 and targets[r['row_index']]['original_b'+a]!=.5]
    assert len(suppressed)==26
    assert detail['suppressed_axes']==dict(total=26,predicts_center=sum(r['prediction']==.5 for r in suppressed),predicts_move=sum(r['prediction']!=.5 for r in suppressed),rows=suppressed)
    assert detail['quadrants_moves']==[r for r in evaluated if r['condition']=='quadrants' and r['movement']!='no_move']
    p=summary['groups']['spatial_pool'];o=summary['groups']['offset_left_right_40'];l=summary['groups']['left_right'];q=summary['groups']['quadrants']
    vector=[p['mse']['Hselected']<=.97*p['mse']['H0'],p['mse']['Hselected']<=1.03*p['mse']['H_star'],
        o['mse']['Hselected']<=.95*o['mse']['H0'],l['mse']['Hselected']<=1.01*l['mse']['H0'],q['mse']['Hselected']<=1.01*q['mse']['H0'],
        p['gains']['harmful']<=10,q['gains']['harmful']==0]
    assert list(summary['clauses'].values())==vector and summary['passed']==sum(vector)
    assert summary['interpretation']=='T019-B grouped-OOF deadband-direction '+('positive' if all(vector) else 'negative')
    assert receipt['T018E_data_used'] is False
    result=dict(passed=True,completed_utc=now(),exact_evaluation_rows=120,all_diagnostics_exact=True,
        train_target_subsets_match_accepted_origin=True,image_id_grouping_exact=True,
        replay_before_reference_open=True,clauses=vector,literal_passed=sum(vector),interpretation=summary['interpretation'])
    save(root/'metric_verification.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['replay','metrics']);p.add_argument('--output',type=Path,default=Path('research_log/T019B_run'))
    a=p.parse_args();{'replay':replay,'metrics':metrics}[a.stage](a.output)
