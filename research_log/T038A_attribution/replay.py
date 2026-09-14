"""Independent standard-library dot/cosine and complete aggregate replay."""
import json,math,statistics,sys,csv,hashlib
from pathlib import Path
from collections import Counter
root=Path(sys.argv[1]);rows=json.loads((root/'stage_b/states.json').read_bytes());summary=json.loads((root/'stage_b/summary.json').read_bytes())
freeze=json.loads((root/'stage_a/freeze.json').read_bytes());receipt=json.loads((root/'stage_b/receipt.json').read_bytes())
prior=list(csv.DictReader(Path(sys.argv[2]).open()));loss={r['low'] for r in prior if float(r['delta_psnr'])<0}
gates=json.loads((root/'stage_a_gates.json').read_bytes())
assert len(loss)==29 and len(rows)==501
assert hashlib.sha256((root/'stage_a/freeze.json').read_bytes()).hexdigest()==receipt['stage_a_freeze_sha256']
assert freeze['completed_utc']<receipt['started_utc']<min(r['utc'] for r in receipt['opened_images'] if r['kind']=='normal')
errors=[];sample_errors=[];samples=0
def flat(x):
    if isinstance(x,list):return [v for a in x for v in flat(a)]
    return [x]
def check(a,b,sample=False):
    if a is None or b is None:assert a is b;return
    err=abs(a-b);errors.append(err)
    if sample:sample_errors.append(err)
    assert err<=1e-6,(a,b)
def quantile(v,p):
    v=sorted(v);i=(len(v)-1)*p;lo=math.floor(i);hi=math.ceil(i)
    return v[lo]+(v[hi]-v[lo])*(i-lo)
for row in rows:
    sample=row['index']%10==0 and row['step'] in [0,20,40];samples+=sample
    assert row['prior_psnr_loss']==(row['low'] in loss)
    accepted=prior[row['index']];assert row['low']==accepted['low'] and row['selected']==(row['step']==int(accepted['common_step']))
    e=flat(row['g_e']);r=flat(row['g_r']);assert len(e)==len(r)==12 and all(math.isfinite(v) for v in e+r)
    masks={k:flat(v) for k,v in row['masks'].items()}
    gate=gates[row['index']];assert gate['stage_a_file_sha256']==freeze['rows'][row['index']]['sha256']
    assert masks['total']==flat(gate['gate']['active'])*3
    assert masks['legacy']==[v and i<8 for i,v in enumerate(masks['total'])]
    assert masks['gain']==[v and i>=8 for i,v in enumerate(masks['total'])]
    for group,mask in masks.items():
        x=[a for a,m in zip(e,mask) if m];y=[b for b,m in zip(r,mask) if m]
        dot=math.fsum(a*b for a,b in zip(x,y));en=math.sqrt(math.fsum(a*a for a in x));rn=math.sqrt(math.fsum(a*a for a in y));deg=en<=1e-12 or rn<=1e-12
        cosine=None if deg else dot/(en*rn)
        values=dict(active_count=len(x),energy_norm=en,reference_norm=rn,dot=dot,cosine=cosine,degenerate=deg,positive_dot=dot>0)
        for key,value in values.items():check(value,row['groups'][group][key],sample)
        row['groups'][group]=values
    check(row['groups']['legacy']['dot']+row['groups']['gain']['dot'],row['groups']['total']['dot'],sample)
selected=[r for r in rows if r['selected']]
sets=dict(all_selected=selected,loss_selected=[r for r in selected if r['prior_psnr_loss']],nonloss_selected=[r for r in selected if not r['prior_psnr_loss']],**{'step_'+str(s):[r for r in rows if r['step']==s] for s in [0,10,20,30,40]})
for name,data in sets.items():
    signs=dict(Counter(('legacy_valid' if r['groups']['legacy']['positive_dot'] else 'legacy_invalid')+' / '+('gain_valid' if r['groups']['gain']['positive_dot'] else 'gain_invalid') for r in data))
    assert signs==summary['sign_patterns'][name]
    for group,record in summary['aggregates'][name].items():
        items=[r['groups'][group] for r in data];valid=[r for r in items if not r['degenerate']];cos=[r['cosine'] for r in valid]
        result=dict(count=len(items),nondegenerate=len(valid),cosine_mean=statistics.mean(cos) if cos else None,cosine_median=statistics.median(cos) if cos else None,
            cosine_p10=quantile(cos,.1) if cos else None,cosine_p90=quantile(cos,.9) if cos else None,
            positive_dot_fraction=sum(r['positive_dot'] for r in valid)/len(valid) if valid else None,
            energy_zero_fraction=sum(r['energy_norm']<=1e-12 for r in items)/len(items),reference_zero_fraction=sum(r['reference_norm']<=1e-12 for r in items)/len(items),either_zero_fraction=sum(r['degenerate'] for r in items)/len(items),
            **{k+'_median':statistics.median(r[k] for r in items) for k in ['energy_norm','reference_norm','dot']})
        for k,v in result.items():check(v,record[k])
g=summary['aggregates']['loss_selected'];good=g['gain']['cosine_median'] is not None and g['gain']['cosine_median']<=-.25 and g['gain']['positive_dot_fraction']<=.35 and g['legacy']['positive_dot_fraction']-g['gain']['positive_dot_fraction']>=.2
verdict='gain-specific mismatch supported' if good else 'gain-specific mismatch not supported / shared-or-mixed field failure'
assert verdict==summary['classification'] and samples==30
assert freeze['max_feature_abs_error']<=1e-6 and freeze['max_energy_abs_error']<=1e-6 and freeze['max_historical_gradient_abs_error']<=1e-5
result=dict(status='PASS',all_states=501,group_vectors=1503,deterministic_subset='indices0,10,...90 at steps0,20,40',subset_states=samples,subset_max_abs_error=max(sample_errors),all_scalar_checks=len(errors),all_max_abs_error=max(errors),classification=verdict,stage_a_before_all_references=True,masks_rebuilt_from_frozen_stage_a_gates=True,exec_replay_thresholds_pass=True)
(root/'independent_replay.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
