"""T016-F independent artifact, exclusion and arithmetic audit; no fitting."""
import hashlib
import json
from pathlib import Path
import statistics as st
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from ttie.stop_quality import QualityHead

torch.set_num_threads(1);root=Path('research_log/T016F_run')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda b:hashlib.sha256(b).hexdigest()
tsha=lambda t:sha(t.contiguous().numpy().tobytes())
cfg=read(root/'config.json');data={}
for name,item in cfg['input_artifact_hashes'].items():
    blob=subprocess.check_output(['git','show',item['commit']+':'+item['path']]);assert sha(blob)==item['sha256'];data[name]=json.loads(blob)
for path,digest in cfg['source_code_sha256'].items():
    blob=subprocess.check_output(['git','show',cfg['source_sha']+':'+path]);assert sha(blob)==digest==sha(Path(path).read_bytes())
entries=data['T016B/selection']['episodes'];episodes=[r['episode'] for r in entries]
refs={r['episode']:r for r in data['T016B/reference']};ids=sorted({r['image_id'] for r in refs.values()})
x=torch.tensor([r['features'] for r in entries],dtype=torch.float32)
coords=torch.tensor([[(a-.5)/.1,(b-.5)/.1] for a,b,_ in cfg['candidates']],dtype=torch.float32)
x=torch.cat((x,coords.expand(120,9,2)),dim=-1)
folds=read(root/'folds.json');assert folds==data['T016C/folds'] and len(ids)==40
allf=read(root/'all_folds_frozen.json');er=read(root/'evaluation_receipt.json')
assert allf['finalized_utc']<=er['evaluation_started_utc']
assert sha((root/'all_folds_frozen.json').read_bytes())==er['all_folds_frozen_sha256']
assert sha((root/'decisions.json').read_bytes())==allf['decisions_sha256']==er['decisions_sha256']
decisions=read(root/'decisions.json');out={r['episode']:r for r in decisions}
assert len(out)==120 and [r['episode'] for r in decisions]==episodes
grid=[0.,.25,.5,.75,1.,1.5,2.,float('inf')];head_count=0;score_count=0;permutations=0;q_error=0.;pair_counts={}
def q_value(scores):
    s=np.asarray(scores,dtype=np.float64);j=min([0,1,2,3,5,6,7,8],key=lambda i:s[i]);scale=float(np.std(s,ddof=0))
    return j,0. if scale==0 else float((s[4]-s[j])/max(scale,1e-12))
for k,outer in enumerate(folds):
    path=root/f'fold{k}';frozen=read(path/'fold_frozen.json');assert frozen==allf['folds'][k]
    assert frozen['finalized_utc']<=allf['finalized_utc']
    assert sha((path/'fold_frozen.json').read_bytes())==er['fold_freeze_hashes'][str(k)]
    for file,digest in frozen['files_sha256'].items():assert sha((path/file).read_bytes())==digest
    heldids=ids[k::5];trainids=[i for i in ids if i not in heldids]
    assert outer['heldout_image_ids']==heldids and outer['train_image_ids']==trainids
    assert outer['train']==[i for i,e in enumerate(episodes) if refs[e]['image_id'] in trainids]
    assert outer['heldout']==[i for i,e in enumerate(episodes) if refs[e]['image_id'] in heldids]
    inner=read(path/'inner_folds.json');seen=[];pair_counts[str(k)]={}
    for j,f in enumerate(inner):
        ih=trainids[j::4];it=[i for i in trainids if i not in ih]
        assert f['heldout_image_ids']==ih and f['train_image_ids']==it
        assert f['heldout']==[i for i in outer['train'] if refs[episodes[i]]['image_id'] in ih]
        assert f['train']==[i for i in outer['train'] if refs[episodes[i]]['image_id'] in it];seen+=f['heldout']
    assert sorted(seen)==outer['train']
    for name,fold in [('outer_head',outer)]+[(f'inner{j}',f) for j,f in enumerate(inner)]:
        p=path/name;receipt=read(p/'receipt.json');assert receipt==frozen['head_receipts'][name]
        assert receipt['finalized_utc']<=frozen['finalized_utc'] and receipt['heldout_reference_used'] is False
        assert receipt['train_image_ids']==fold['train_image_ids'] and receipt['heldout_image_ids']==fold['heldout_image_ids']
        assert not set(receipt['train_image_ids'])&set(heldids)
        assert not set(receipt['train_image_ids'])&set(receipt['heldout_image_ids'])
        for file,key in [('head.pt','head_sha256'),('history.json','history_sha256'),('predictions.json','predictions_sha256')]:assert sha((p/file).read_bytes())==receipt[key]
        saved=torch.load(p/'head.pt',weights_only=True,map_location='cpu');assert saved['recipe']==cfg['recipe']
        head=QualityHead(30,torch.nn.SiLU);head.load_state_dict(saved['state_dict']);head.eval()
        flat=x[fold['train']].reshape(-1,30).double();scale=flat.std(0,unbiased=False)
        assert torch.equal(head.x_mean,flat.mean(0).float()) and torch.equal(head.x_scale,torch.where(scale==0,torch.ones_like(scale),scale).float())
        assert head.y_mean.item()==0 and head.y_scale.item()==1 and head.normalization()==receipt['normalization']
        pairs=[];signs=[]
        for e,index in enumerate(fold['train']):
            mse=refs[episodes[index]]['reference_mse']
            for i in range(9):
                for j in range(i+1,9):
                    if mse[i]!=mse[j]:pairs.append([e,i,j]);signs.append(1. if mse[i]<mse[j] else -1.)
        n=len(pairs);pair_counts[str(k)][name]=n
        assert n==receipt['non_tied_pairs'] and receipt['exact_ties_skipped']==len(fold['train'])*36-n
        assert tsha(torch.tensor(pairs,dtype=torch.int64))==receipt['pair_index_sha256']
        assert tsha(torch.tensor(signs,dtype=torch.float32))==receipt['pair_sign_sha256']
        generator=torch.Generator().manual_seed(7);history=read(p/'history.json');assert len(history)==100
        for epoch,row in enumerate(history,1):
            assert row['epoch']==epoch and row['pairs_seen']==n
            assert tsha(torch.randperm(n,generator=generator))==row['permutation_sha256'];permutations+=1
        assert receipt['final_train_pairwise_loss']==history[-1]['train_pairwise_loss']
        with torch.no_grad():values=head(x[fold['heldout']].reshape(-1,30)).reshape(-1,9).tolist()
        pred=read(p/'predictions.json')
        for index,v,r in zip(fold['heldout'],values,pred):
            assert r['episode']==episodes[index] and r['predictions']==v
            assert r['selected_index']==min(range(9),key=lambda i:v[i]);score_count+=9
        head_count+=1
    ioof=read(path/'inner_oof.json');oscores=read(path/'outer_scores.json');cal=read(path/'calibration.json')
    assert [r['episode'] for r in ioof]==[episodes[i] for i in outer['train']]
    for r in ioof:
        own=inner[r['fold']];assert refs[r['episode']]['image_id'] in own['heldout_image_ids']
        assert r in read(path/f"inner{r['fold']}"/'predictions.json')
    assert oscores==read(path/'outer_head'/'predictions.json')
    assert cal['inner_oof_sha256']==sha((path/'inner_oof.json').read_bytes())
    assert [r['episode'] for r in cal['calibration_episodes']]==[r['episode'] for r in ioof]
    means=[]
    for t,entry in zip(grid,cal['table']):
        mean=st.mean(refs[r['episode']]['reference_mse'][q_value(r['predictions'])[0] if q_value(r['predictions'])[1]>t else 4] for r in ioof)
        assert mean==entry['mean_selected_mse'];means.append(mean)
        assert entry['threshold']==('inf' if np.isinf(t) else t)
    threshold=grid[min(range(8),key=lambda i:(means[i],-grid[i]))]
    assert cal['threshold']==frozen['threshold']==('inf' if np.isinf(threshold) else threshold)
    for row in oscores:
        j,q=q_value(row['predictions']);d=out[row['episode']]
        assert d in read(path/'decisions.json') and d['fold']==k and d['predictions']==row['predictions']
        assert d['selected_index']==(j if q>threshold else 4) and d['noncanonical_index']==j
        assert d['threshold']==cal['threshold'];q_error=max(q_error,abs(q-d['q']));assert abs(q-d['q'])<1e-12
rows=read(root/'evaluation.json');report=read(root/'summary.json');old={r['episode']:r for r in data['T016E/evaluation']}
ungated=read(root/'ungated_evaluation.json');u_by={r['episode']:r for r in ungated}
for row in rows:
    e=row['episode'];mse=refs[e]['reference_mse'];d=out[e];u=min(range(9),key=lambda i:d['predictions'][i])
    assert row['selected_index']==d['selected_index'] and row['selected_mse']==mse[d['selected_index']]
    assert row['ungated_nested_mse']==mse[u]==u_by[e]['selected_mse'] and row['old_t016e_mse']==old[e]['selected_mse']
    assert row['canonical_candidate_mse']==mse[4]==row['region2_mse']==refs[e]['region2_mse']
    assert row['hard_oracle_mse']==min(mse) and row['oracle_index']==mse.index(min(mse))
    assert row['disagreement']==(d['selected_index']!=mse.index(min(mse)))
    assert row['outside_oracle_tie_set']==(row['selected_mse']!=min(mse))
def diagnostics(group,d):
    adapted=[r for r in group if r['selected_index']!=4];q=[r['q'] for r in group]
    assert d['adaptive']==len(adapted) and d['canonical']==len(group)-len(adapted)
    assert d['adapted_gains']==dict(beneficial=sum(r['selected_mse']<r['canonical_candidate_mse'] for r in adapted),harmful=sum(r['selected_mse']>r['canonical_candidate_mse'] for r in adapted),zero=sum(r['selected_mse']==r['canonical_candidate_mse'] for r in adapted))
    assert d['q']['mean']==st.mean(q) and d['q']['median']==st.median(q)
    for p,v in d['q']['quantiles'].items():assert abs(v-float(np.quantile(q,int(p)/100,method='linear')))<1e-12
for c,g in report['groups'].items():
    group=rows if c=='spatial_pool' else [r for r in rows if r['condition']==c]
    for key in ('selected_mse','region2_mse','hard_oracle_mse','frozen_t016b_mse','ungated_nested_mse','old_t016e_mse'):assert g[key]==st.mean(r[key] for r in group)
    for key,v in g['ratios'].items():assert v==g['selected_mse']/g[key.removeprefix('selected_over_')+'_mse']
    assert g['selected_counts']==[sum(r['selected_index']==i for r in group) for i in range(9)]
    assert g['oracle_counts']==[sum(r['oracle_index']==i for r in group) for i in range(9)]
    assert g['disagreement_rate']==st.mean(r['disagreement'] for r in group) and g['outside_oracle_tie_set_rate']==st.mean(r['outside_oracle_tie_set'] for r in group)
    diagnostics(group,g['confidence_diagnostics'])
for f in report['folds']:
    group=[r for r in rows if r['fold']==f['fold']];assert f['selected_mse']==st.mean(r['selected_mse'] for r in group)
    diagnostics(group,f['confidence_diagnostics'])
    for c,d in f['condition_diagnostics'].items():
        subset=[r for r in group if r['condition']==c];diagnostics(subset,d);assert f['per_condition'][c]==st.mean(r['selected_mse'] for r in subset)
g=report['groups'];a=g['spatial_pool'];o=g['offset_left_right_40'];l=g['left_right'];q=g['quadrants']
clauses=[a['selected_mse']<=.97*a['region2_mse'],a['selected_mse']<=1.05*a['hard_oracle_mse'],o['selected_mse']<=.95*o['region2_mse'],l['selected_mse']<=1.01*l['region2_mse'],q['selected_mse']<=1.01*q['region2_mse']]
assert clauses==list(report['clauses'].values()) and sum(clauses)==report['passed'] and all(clauses)==report['qualified']
result=dict(passed=True,source_files=len(cfg['source_code_sha256']),input_artifacts=len(data),heads=head_count,exact_saved_head_scores=score_count,
    epoch_permutations_verified=permutations,all_outer_reference_exclusions_verified=True,inner_oof_own_image_exclusions_verified=True,
    calibration_means_verified=40,thresholds=[f['threshold'] for f in allf['folds']],q_max_abs_numpy_difference=q_error,
    metrics_counts_gains_quantiles_clauses_verified=True,all_fold_freezes_before_evaluation=True,training_pairs=pair_counts,no_fitting_in_audit=True)
Path('research_log/T016F_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
