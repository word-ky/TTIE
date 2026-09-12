"""One-off T016-D artifact audit; no fitting and no scientific code changes."""
import hashlib
import json
from pathlib import Path
import statistics as st
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from ttie.stop_quality import QualityHead

torch.set_num_threads(1)
root=Path('research_log/T016D_run')
read=lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda b: hashlib.sha256(b).hexdigest()
tensor_sha=lambda t: sha(t.contiguous().numpy().tobytes())
config=read(root/'config.json'); folds=read(root/'folds.json')
summary=read(root/'summary.json'); evaluated=read(root/'evaluation.json')
inputs={}
for name,item in config['input_artifact_hashes'].items():
    blob=subprocess.check_output(['git','show',item['commit']+':'+item['path']])
    assert sha(blob)==item['sha256']; inputs[name]=json.loads(blob)
for path,digest in config['source_code_sha256'].items():
    blob=subprocess.check_output(['git','show',config['source_sha']+':'+path])
    assert sha(blob)==digest==sha(Path(path).read_bytes())
assert folds==inputs['T016C/folds']
entries=inputs['T016B/selection']['episodes']; episodes=[r['episode'] for r in entries]
refs={r['episode']:r for r in inputs['T016B/reference']}
ids=sorted({r['image_id'] for r in refs.values()}); assert len(ids)==40 and len(episodes)==120
frozen=read(root/'OOF_frozen.json'); er=read(root/'evaluation_receipt.json')
assert frozen['finalized_utc']<er['evaluation_started_utc']
assert er['oof_frozen_sha256']==sha((root/'OOF_frozen.json').read_bytes())
pair_counts={}; predictions_checked=0
for name in ('rank28','rank30'):
    dim=int(name[-2:]); x=torch.tensor([r['features'] for r in entries],dtype=torch.float32)
    if dim==30:
        coords=torch.tensor([[(a-.5)/.1,(b-.5)/.1] for a,b,_ in config['candidates']],dtype=torch.float32)
        x=torch.cat((x,coords.expand(120,9,2)),dim=-1)
    oof=read(root/name/'oof.json'); assert [r['episode'] for r in oof]==episodes
    oof_by={r['episode']:r for r in oof}; pair_counts[name]=[]
    digest=sha((root/name/'oof.json').read_bytes())
    assert digest==frozen['probes'][name]['oof_sha256']==er['unchanged_oof_hashes'][name]
    for fold in folds:
        f=fold['fold']; p=root/name/f'fold{f}'; receipt=read(p/'receipt.json')
        assert receipt==frozen['probes'][name]['fold_receipts'][f]
        assert receipt['finalized_utc']<frozen['finalized_utc']
        heldids=ids[f::5]; trainids=[i for i in ids if i not in heldids]
        train=[i for i,e in enumerate(episodes) if refs[e]['image_id'] in trainids]
        held=[i for i,e in enumerate(episodes) if refs[e]['image_id'] in heldids]
        assert train==fold['train'] and held==fold['heldout']
        assert receipt['train_image_ids']==trainids and receipt['heldout_image_ids']==heldids
        assert len(train)==96 and len(held)==24 and receipt['heldout_reference_used'] is False
        for file,key in [('head.pt','head_sha256'),('history.json','history_sha256'),('predictions.json','predictions_sha256')]:
            assert sha((p/file).read_bytes())==receipt[key]
        saved=torch.load(p/'head.pt',weights_only=True,map_location='cpu')
        head=QualityHead(dim,torch.nn.SiLU); head.load_state_dict(saved['state_dict']);head.eval()
        assert head.normalization()==receipt['normalization']
        flat=x[train].reshape(-1,dim).double(); scale=flat.std(0,unbiased=False)
        assert torch.equal(head.x_mean,flat.mean(0).float())
        assert torch.equal(head.x_scale,torch.where(scale==0,torch.ones_like(scale),scale).float())
        assert head.y_mean.item()==0 and head.y_scale.item()==1
        pair=[]; signs=[]
        for e,index in enumerate(train):
            values=refs[episodes[index]]['reference_mse']
            for i in range(9):
                for j in range(i+1,9):
                    if values[i]!=values[j]:
                        pair.append([e,i,j]); signs.append(1. if values[i]<values[j] else -1.)
        n=len(pair); pair_counts[name].append(n)
        assert n==receipt['non_tied_pairs'] and receipt['exact_ties_skipped']==96*36-n
        assert tensor_sha(torch.tensor(pair,dtype=torch.int64))==receipt['pair_index_sha256']
        assert tensor_sha(torch.tensor(signs,dtype=torch.float32))==receipt['pair_sign_sha256']
        history=read(p/'history.json'); assert len(history)==100
        generator=torch.Generator().manual_seed(7)
        for epoch,row in enumerate(history,1):
            assert row['epoch']==epoch and row['pairs_seen']==n
            assert tensor_sha(torch.randperm(n,generator=generator))==row['permutation_sha256']
        assert history[-1]['train_pairwise_loss']==receipt['final_train_pairwise_loss']==summary['probes'][name]['final_train_pairwise_loss'][f]
        with torch.no_grad(): scores=head(x[held].reshape(-1,dim)).reshape(24,9).tolist()
        rows=read(p/'predictions.json')
        for index,values,row in zip(held,scores,rows):
            assert row==oof_by[episodes[index]] and values==row['predictions']
            assert row['selected_index']==min(range(9),key=lambda i:values[i])
            predictions_checked+=9
    rows=evaluated[name]; old={r['episode']:r for r in inputs['T016C/evaluation'][name.replace('rank','probe')]}
    for row in rows:
        pred=oof_by[row['episode']]; ref=refs[row['episode']]; mse=ref['reference_mse']; idx=pred['selected_index']
        assert row['predictions']==pred['predictions'] and row['selected_index']==idx
        assert row['selected_mse']==mse[idx] and row['hard_oracle_mse']==min(mse)
        assert row['region2_mse']==ref['region2_mse'] and row['frozen_t016b_mse']==ref['selected_mse']
        assert row['pointwise_t016c_mse']==old[row['episode']]['selected_mse']
        assert row['oracle_index']==mse.index(min(mse))
        assert row['disagreement']==(idx!=mse.index(min(mse)))
        assert row['outside_oracle_tie_set']==(mse[idx]!=min(mse))
        assert row['prediction_minimum_ties']==pred['predictions'].count(min(pred['predictions']))
        assert row['reference_minimum_ties']==mse.count(min(mse))
    report=summary['probes'][name]
    for condition,g in report['groups'].items():
        group=rows if condition=='spatial_pool' else [r for r in rows if r['condition']==condition]
        for key in ('selected_mse','region2_mse','hard_oracle_mse','frozen_t016b_mse','pointwise_t016c_mse'):
            assert g[key]==st.mean(r[key] for r in group)
        for key,ratio in g['ratios'].items(): assert ratio==g['selected_mse']/g[key.removeprefix('selected_over_')+'_mse']
        assert g['selected_counts']==[sum(r['selected_index']==i for r in group) for i in range(9)]
        assert g['oracle_counts']==[sum(r['oracle_index']==i for r in group) for i in range(9)]
        assert g['disagreement_rate']==st.mean(r['disagreement'] for r in group)
        assert g['outside_oracle_tie_set_rate']==st.mean(r['outside_oracle_tie_set'] for r in group)
    for fold in report['folds']:
        group=[r for r in rows if r['fold']==fold['fold']]
        assert fold['selected_mse']==st.mean(r['selected_mse'] for r in group)
        for c,v in fold['per_condition'].items(): assert v==st.mean(r['selected_mse'] for r in group if r['condition']==c)
    g=report['groups']; a=g['spatial_pool'];o=g['offset_left_right_40'];l=g['left_right'];q=g['quadrants']
    clauses=[a['selected_mse']<=.97*a['region2_mse'],a['selected_mse']<=1.05*a['hard_oracle_mse'],o['selected_mse']<=.95*o['region2_mse'],l['selected_mse']<=1.01*l['region2_mse'],q['selected_mse']<=1.01*q['region2_mse']]
    assert list(report['clauses'].values())==clauses and report['passed']==sum(clauses) and report['qualified']==all(clauses)
assert pair_counts['rank28']==pair_counts['rank30']
result=dict(passed=True,source_files=len(config['source_code_sha256']),source_artifacts=len(inputs),heads=10,
    exact_oof_scores=predictions_checked,fold_pairs=pair_counts,epoch_permutations_verified=1000,
    metrics_and_clauses_verified=True,freeze_before_evaluation=True,no_training=True)
Path('research_log/T016D_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
