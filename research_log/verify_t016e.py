"""Independent post-run T016-E arithmetic/artifact audit; no calibration changes."""
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import subprocess
import numpy as np

root=Path('research_log/T016E_run')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda b:hashlib.sha256(b).hexdigest()
cfg=read(root/'config.json');inputs={}
for name,item in cfg['input_artifact_hashes'].items():
    data=subprocess.check_output(['git','show',item['commit']+':'+item['path']])
    assert sha(data)==item['sha256'];inputs[name]=json.loads(data)
for path,digest in cfg['source_code_sha256'].items():
    data=subprocess.check_output(['git','show',cfg['source_sha']+':'+path])
    assert sha(data)==digest==sha(Path(path).read_bytes())
folds=read(root/'folds.json');assert folds==inputs['folds']==inputs['c_folds']
oof=inputs['scores']; refs={r['episode']:r for r in inputs['reference']}
ids=sorted({r['image_id'] for r in refs.values()});assert len(ids)==40 and len(oof)==120
cal=read(root/'calibrations.json');decisions=read(root/'decisions.json');out={r['episode']:r for r in decisions}
frozen=read(root/'decisions_frozen.json');receipt=read(root/'evaluation_receipt.json')
assert frozen['finalized_utc']<=receipt['evaluation_started_utc']
assert sha((root/'decisions_frozen.json').read_bytes())==receipt['frozen_sha256']
for filename in ('calibrations','decisions'):
    assert sha((root/(filename+'.json')).read_bytes())==frozen[filename+'_sha256']==receipt[filename+'_sha256']
grid=[0.,.25,.5,.75,1.,1.5,2.,math.inf]
assert cfg['thresholds']==[0.,.25,.5,.75,1.,1.5,2.,'inf']
derived={};maxq=0.;dependency_count=0
for row in oof:
    s=np.asarray(row['predictions'],dtype=np.float64)
    j=min([0,1,2,3,5,6,7,8],key=lambda i:s[i])
    std=float(np.std(s,ddof=0));q=0. if std==0 else float((s[4]-s[j])/max(std,1e-12))
    derived[row['episode']]=(j,q)
    maxq=max(maxq,abs(q-out[row['episode']]['q']))
    assert abs(q-out[row['episode']]['q'])<1e-12
    assert j==out[row['episode']]['noncanonical_index']
    assert row['predictions']==out[row['episode']]['predictions']
for k,fold in enumerate(folds):
    heldids=ids[k::5];trainids=[i for i in ids if i not in heldids]
    train=[r for r in oof if refs[r['episode']]['image_id'] in trainids]
    held=[r for r in oof if refs[r['episode']]['image_id'] in heldids]
    assert len(train)==96 and len(held)==24
    assert trainids==fold['train_image_ids']==cal[k]['train_image_ids']
    assert heldids==fold['heldout_image_ids']==cal[k]['heldout_image_ids']
    assert [oof[i] for i in fold['train']]==train and [oof[i] for i in fold['heldout']]==held
    trace=cal[k]['calibration_episodes'];assert len(trace)==96
    for row,saved in zip(train,trace):
        image=refs[row['episode']]['image_id'];sourcefold=row['fold']
        assert image in folds[sourcefold]['heldout_image_ids'] and sourcefold!=k
        assert set(heldids).issubset(folds[sourcefold]['train_image_ids']);dependency_count+=1
        assert saved['episode']==row['episode'] and saved['source_fold']==sourcefold
        assert abs(saved['q']-derived[row['episode']][1])<1e-12
    means=[]
    for t,entry in zip(grid,cal[k]['table']):
        values=[]
        for row in train:
            j,q=derived[row['episode']];values.append(refs[row['episode']]['reference_mse'][j if q>t else 4])
        mean=st.mean(values);means.append(mean)
        assert mean==entry['mean_selected_mse']
        assert entry['threshold']==('inf' if math.isinf(t) else t)
    best=sorted(zip(means,grid),key=lambda pair:(pair[0],-pair[1]))[0][1]
    assert cal[k]['threshold']==('inf' if math.isinf(best) else best)==frozen['thresholds'][k]
    assert cal[k]['finalized_utc']<=frozen['finalized_utc']
    assert cal[k]['input_oof_sha256']==cfg['input_artifact_hashes']['scores']['sha256']
    for row in held:
        j,q=derived[row['episode']];d=out[row['episode']]
        assert d['fold']==k==d['source_fold'] and d['threshold']==cal[k]['threshold']
        assert d['selected_index']==(j if q>best else 4)
assert len(out)==120 and [d['episode'] for d in decisions]==[r['episode'] for r in oof]
rows=read(root/'evaluation.json');report=read(root/'summary.json')
dold={r['episode']:r for r in inputs['d_evaluation']['rank30']};cold={r['episode']:r for r in inputs['c_evaluation']['probe30']}
for row in rows:
    e=row['episode'];reference=refs[e];mse=reference['reference_mse'];j=out[e]['selected_index']
    assert row['selected_index']==j and row['selected_mse']==mse[j]
    assert row['canonical_candidate_mse']==mse[4]==reference['region2_mse']==row['region2_mse']
    assert row['ungated_rank30_mse']==dold[e]['selected_mse'] and row['pointwise_probe30_mse']==cold[e]['selected_mse']
    assert row['hard_oracle_mse']==min(mse) and row['oracle_index']==mse.index(min(mse))
    assert row['disagreement']==(j!=mse.index(min(mse))) and row['outside_oracle_tie_set']==(mse[j]!=min(mse))

def audit_diagnostics(group,diag):
    adapted=[r for r in group if r['selected_index']!=4]
    assert diag['count']==len(group) and diag['adaptive']==len(adapted) and diag['canonical']==len(group)-len(adapted)
    assert diag['adapted_gains']==dict(beneficial=sum(r['selected_mse']<r['canonical_candidate_mse'] for r in adapted),
        harmful=sum(r['selected_mse']>r['canonical_candidate_mse'] for r in adapted),zero=sum(r['selected_mse']==r['canonical_candidate_mse'] for r in adapted))
    q=[r['q'] for r in group]
    assert diag['q']['mean']==st.mean(q) and diag['q']['median']==st.median(q)
    for p,v in diag['q']['quantiles'].items():assert abs(v-float(np.quantile(q,int(p)/100,method='linear')))<1e-12

for condition,g in report['groups'].items():
    group=rows if condition=='spatial_pool' else [r for r in rows if r['condition']==condition]
    for key in ('selected_mse','region2_mse','hard_oracle_mse','frozen_t016b_mse','ungated_rank30_mse','pointwise_probe30_mse'):
        assert g[key]==st.mean(r[key] for r in group)
    for k,v in g['ratios'].items():assert v==g['selected_mse']/g[k.removeprefix('selected_over_')+'_mse']
    assert g['selected_counts']==[sum(r['selected_index']==i for r in group) for i in range(9)]
    assert g['oracle_counts']==[sum(r['oracle_index']==i for r in group) for i in range(9)]
    assert g['disagreement_rate']==st.mean(r['disagreement'] for r in group)
    assert g['outside_oracle_tie_set_rate']==st.mean(r['outside_oracle_tie_set'] for r in group)
    audit_diagnostics(group,g['confidence_diagnostics'])
for f in report['folds']:
    group=[r for r in rows if r['fold']==f['fold']]
    assert f['selected_mse']==st.mean(r['selected_mse'] for r in group)
    audit_diagnostics(group,f['confidence_diagnostics'])
    for c,diag in f['condition_diagnostics'].items():
        subset=[r for r in group if r['condition']==c];audit_diagnostics(subset,diag)
        assert f['per_condition'][c]==st.mean(r['selected_mse'] for r in subset)
g=report['groups'];a=g['spatial_pool'];o=g['offset_left_right_40'];l=g['left_right'];q=g['quadrants']
clauses=[a['selected_mse']<=.97*a['region2_mse'],a['selected_mse']<=1.05*a['hard_oracle_mse'],o['selected_mse']<=.95*o['region2_mse'],l['selected_mse']<=1.01*l['region2_mse'],q['selected_mse']<=1.01*q['region2_mse']]
assert clauses==list(report['clauses'].values()) and sum(clauses)==report['passed'] and all(clauses)==report['qualified']
result=dict(passed=True,source_files=len(cfg['source_code_sha256']),input_artifacts=len(inputs),q_decisions_verified=120,
    q_max_abs_numpy_difference=maxq,calibration_means_verified=40,thresholds=frozen['thresholds'],
    direct_calibration_heldout_separation=True,prior_oof_ranker_dependency_on_outer_heldout_ids=dependency_count,
    inherited_dependency_note='All480calibrationepisode appearances use a ranker trained on the current outer-heldout8IDs; prescribed OOF reuse is not fully nested ranker training.',
    metrics_counts_quantiles_and_clauses_verified=True,freeze_before_evaluation=True,no_models_or_training=True)
Path('research_log/T016E_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
