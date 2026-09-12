"""Audit frozen OOF evidence and saved-head inference; never train or change selections."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import torch
from ttie.stop_quality import QualityHead


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root=Path('research_log/T016C_run');config=read(root/'config.json')
    data={}
    for key,item in config['input_artifact_hashes'].items():
        blob=subprocess.check_output(['git','show',config['evidence_commit']+':'+item['path']])
        assert hashlib.sha256(blob).hexdigest()==item['sha256'];data[key]=json.loads(blob)
    for path,digest in config['source_code_sha256'].items():
        blob=subprocess.check_output(['git','show',config['source_sha']+':'+path])
        assert hashlib.sha256(blob).hexdigest()==digest==sha(Path(path))
    source=data['selection']['episodes'];reference={r['episode']:r for r in data['reference']}
    episodes=[r['episode'] for r in source];ids=sorted({r['image_id'] for r in reference.values()})
    assert len(ids)==40;assignment={image_id:i%5 for i,image_id in enumerate(ids)}
    folds=read(root/'folds.json');frozen=read(root/'OOF_frozen.json');evaluation_receipt=read(root/'evaluation_receipt.json')
    assert frozen['finalized_utc']<evaluation_receipt['evaluation_started_utc']
    assert sha(root/'OOF_frozen.json')==evaluation_receipt['oof_frozen_sha256']
    evaluated=read(root/'evaluation.json');summary=read(root/'summary.json')
    maximum_prediction_difference=0.;defined_correlations={}
    torch.set_num_threads(1)
    for dimension in (28,30):
        name=f'probe{dimension}';x=torch.tensor([r['features'] for r in source],dtype=torch.float32)
        if dimension==30:
            geometry=torch.tensor([((bx-.5)/.1,(by-.5)/.1) for bx in (.4,.5,.6) for by in (.4,.5,.6)])
            x=torch.cat((x,geometry.expand(120,9,2)),dim=-1)
        oof=read(root/name/'oof.json');oof_by_key={r['episode']:r for r in oof}
        assert len(oof)==len(oof_by_key)==120 and set(oof_by_key)==set(episodes)
        assert sha(root/name/'oof.json')==frozen['probes'][name]['oof_sha256']==evaluation_receipt['unchanged_oof_hashes'][name]
        seen=[]
        for fold in folds:
            f=fold['fold'];out=root/name/f'fold{f}';receipt=read(out/'receipt.json')
            assert receipt['finalized_utc']<=frozen['finalized_utc']
            expected_test=[i for i,e in enumerate(episodes) if assignment[reference[e]['image_id']]==f]
            expected_train=[i for i,e in enumerate(episodes) if assignment[reference[e]['image_id']]!=f]
            assert fold['train']==expected_train and fold['heldout']==expected_test
            assert len(expected_train)==96 and len(expected_test)==24
            assert len(receipt['train_image_ids'])==32 and len(receipt['heldout_image_ids'])==8
            assert not set(receipt['train_image_ids'])&set(receipt['heldout_image_ids'])
            assert receipt['train_rows']==864 and receipt['heldout_rows']==216 and receipt['epochs']==100
            for file,key in [('head.pt','head_sha256'),('predictions.json','predictions_sha256'),('history.json','history_sha256')]:
                assert sha(out/file)==receipt[key]
            saved=torch.load(out/'head.pt',weights_only=True);state=saved['state_dict']
            train_x=x[expected_train].reshape(-1,dimension)
            y=torch.tensor([reference[episodes[i]]['reference_mse'] for i in expected_train],dtype=torch.float64).flatten()
            log_y=(y+1e-6).log();scale=train_x.double().std(0,unbiased=False)
            expected=dict(x_mean=train_x.double().mean(0).float(),x_scale=torch.where(scale==0,torch.ones_like(scale),scale).float(),
                          y_mean=log_y.mean().float(),y_scale=log_y.std(unbiased=False).float())
            for key,value in expected.items():assert torch.equal(state[key],value)
            assert saved['recipe']==config['recipes'][name] and saved['dimension']==dimension
            head=QualityHead(dimension,torch.nn.SiLU);head.load_state_dict(state);head.eval().requires_grad_(False)
            with torch.no_grad():predicted=head(x[expected_test].reshape(-1,dimension)).reshape(-1,9)
            fold_predictions=read(out/'predictions.json')
            recorded=torch.tensor([r['predictions'] for r in fold_predictions],dtype=predicted.dtype)
            maximum_prediction_difference=max(maximum_prediction_difference,float((predicted-recorded).abs().max()))
            assert torch.equal(predicted,recorded)
            for i,r in zip(expected_test,fold_predictions):
                assert r['episode']==episodes[i] and r==oof_by_key[episodes[i]] and r['fold']==f
                assert r['selected_index']==min(range(9),key=lambda j:(r['predictions'][j],j))
                seen.append(i)
            history=read(out/'history.json');assert [r['epoch'] for r in history]==list(range(1,101))
            assert history[-1]['train_huber']==receipt['final_train_huber']==summary['probes'][name]['final_train_huber'][f]
        assert sorted(seen)==list(range(120));defined=0
        for r in evaluated[name]:
            original=reference[r['episode']];prediction=oof_by_key[r['episode']];mse=original['reference_mse']
            assert r['selected_mse']==mse[prediction['selected_index']]
            assert r['region2_mse']==original['region2_mse'] and r['frozen_t016b_mse']==original['selected_mse']
            assert r['hard_oracle_mse']==min(mse)
            if len(set(prediction['predictions']))>1 and len(set(mse))>1:
                assert r['spearman'] is not None;defined+=1
            else:assert r['spearman'] is None
        defined_correlations[name]=defined
        p=summary['probes'][name]
        for condition,g in p['groups'].items():
            group=evaluated[name] if condition=='spatial_pool' else [r for r in evaluated[name] if r['condition']==condition]
            for key in ('selected_mse','region2_mse','hard_oracle_mse','frozen_t016b_mse'):
                assert abs(statistics.mean(r[key] for r in group)-g[key])<1e-14
            for key in ('region2','hard_oracle','frozen_t016b'):
                assert abs(g['selected_mse']/g[key+'_mse']-g['ratios']['selected_over_'+key])<1e-14
            assert g['selected_counts']==[sum(r['selected_index']==i for r in group) for i in range(9)]
            assert g['oracle_counts']==[sum(r['oracle_index']==i for r in group) for i in range(9)]
            correlations=[r['spearman'] for r in group if r['spearman'] is not None]
            assert statistics.median(correlations)==g['spearman']['median']
        g=p['groups'];a=g['spatial_pool'];o=g['offset_left_right_40'];l=g['left_right'];q=g['quadrants']
        clauses=[a['selected_mse']<=.97*a['region2_mse'],a['selected_mse']<=1.05*a['hard_oracle_mse'],o['selected_mse']<=.95*o['region2_mse'],l['selected_mse']<=1.01*l['region2_mse'],q['selected_mse']<=1.01*q['region2_mse']]
        assert clauses==list(p['clauses'].values()) and sum(clauses)==p['passed']
    result=dict(passed=True,source_sha=config['source_sha'],scientific_git_blobs=len(config['source_code_sha256']),
        source_artifact_blobs=len(config['input_artifact_hashes']),folds=5,heads=10,
        train_only_normalizations_verified=10,group_isolation_verified=True,
        exact_saved_head_oof_prediction_values=2160,max_prediction_absolute_difference=maximum_prediction_difference,
        nonconstant_spearman_counts=defined_correlations,all_oof_finalized_before_reference_evaluation=True,
        no_training_or_rendering_in_verifier=True,clauses_passed={n:p['passed'] for n,p in summary['probes'].items()})
    Path('research_log/T016C_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
