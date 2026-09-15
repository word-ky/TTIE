"""Independent stdlib all200 scalar replay and matched-source comparison."""
import json,math,statistics,sys,hashlib
from pathlib import Path
root=Path(sys.argv[1]);rows=json.loads((root/'stage_b/states.json').read_bytes());summary=json.loads((root/'stage_b/summary.json').read_bytes())
freeze=json.loads((root/'stage_a/freeze.json').read_bytes());receipt=json.loads((root/'stage_b/receipt.json').read_bytes());gates=json.loads((root/'stage_a/gates.json').read_bytes())
assert len(rows)==freeze['audited_states']==receipt['output_hashes_exact']==200
assert hashlib.sha256((root/'stage_a/freeze.json').read_bytes()).hexdigest()==receipt['stage_a_freeze_sha256']
assert hashlib.sha256((root/'stage_a/gates.json').read_bytes()).hexdigest()==freeze['gates_sha256']
assert freeze['completed_utc']<receipt['started_utc']<min(r['utc'] for r in receipt['opened_images'] if r['kind']=='normal')
assert freeze['normal_decodes']==freeze['prior_metric_files_opened']==freeze['prior_loss_id_files_opened']==freeze['optimizer_updates']==freeze['selection_changes']==0
expected=[(i,g) for i in range(100) for g in [1.25,1.75]]
errors=[];sample_errors=[]
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
    idx=row['index'];gain=row['gain'];assert (idx,gain)==expected.pop(0)
    accepted=freeze['rows'][idx];assert row['low']==accepted['low'] and row['selected_step']==accepted['selected_step']
    e=flat(row['g_e']);r=flat(row['g_r']);assert len(e)==len(r)==12 and all(math.isfinite(v) for v in e+r)
    gate=gates[idx];assert gate['stage_a_file_sha256']==accepted['sha256']
    active=flat(gate['gate']['active']);masks=dict(total=active*3,legacy=active*2+[False]*4,gain=[False]*8+active)
    assert masks=={k:flat(v) for k,v in row['masks'].items()}
    for group,mask in masks.items():
        x=[a for a,m in zip(e,mask) if m];y=[b for b,m in zip(r,mask) if m]
        dot=math.fsum(a*b for a,b in zip(x,y));en=math.sqrt(math.fsum(a*a for a in x));rn=math.sqrt(math.fsum(a*a for a in y));deg=en<=1e-12 or rn<=1e-12
        values=dict(active_count=len(x),energy_norm=en,reference_norm=rn,dot=dot,cosine=None if deg else dot/(en*rn),degenerate=deg,positive_dot=dot>0)
        for key,value in values.items():check(value,row['groups'][group][key])
        row['groups'][group]=values
    check(row['groups']['legacy']['dot']+row['groups']['gain']['dot'],row['groups']['total']['dot'])
for gain in ['1.25','1.75']:
    data=[r for r in rows if str(r['gain'])==gain]
    for group,record in summary['by_gain'][gain].items():
        items=[r['groups'][group] for r in data];valid=[r for r in items if not r['degenerate']];cos=[r['cosine'] for r in valid]
        result=dict(count=len(items),nondegenerate=len(valid),cosine_mean=statistics.mean(cos) if cos else None,cosine_median=statistics.median(cos) if cos else None,
            cosine_p10=quantile(cos,.1) if cos else None,cosine_p90=quantile(cos,.9) if cos else None,
            positive_dot_fraction=sum(r['positive_dot'] for r in valid)/len(valid) if valid else None,
            energy_zero_fraction=sum(r['energy_norm']<=1e-12 for r in items)/len(items),reference_zero_fraction=sum(r['reference_norm']<=1e-12 for r in items)/len(items),either_zero_fraction=sum(r['degenerate'] for r in items)/len(items),
            **{k+'_median':statistics.median(r[k] for r in items) for k in ['energy_norm','reference_norm','dot']})
        for k,v in result.items():check(v,record[k])
for gain,b in summary['source_baselines'].items():
    raw=Path(b['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==b['sha256']
    assert json.loads(raw)['by_gain'][gain]==b['groups']
    for group in ['legacy','gain','total']:
        for k in ['positive_dot_fraction','cosine_median']:
            check(summary['by_gain'][gain][group][k]-b['groups'][group][k],summary['target_minus_source'][gain][group][k])
base=summary['source_baselines']['1.75']['groups']['total']
check(base['positive_dot_fraction'],.796633554084);check(base['cosine_median'],.763459378857)
t=summary['by_gain']['1.75']['total'];yes=.796633554084-t['positive_dot_fraction']>=.2 and .763459378857-t['cosine_median']>=.25
verdict='real selected-state field deficit beyond source high-gain supported' if yes else 'real selected-state field deficit beyond source high-gain not supported / mixed'
assert verdict==summary['classification'] and not expected
result=dict(status='PASS',all_states=200,group_vectors=600,scalar_checks=len(errors),max_abs_error=max(errors),classification=verdict,source_baseline_hashes_bound=True,masks_rebuilt_from_frozen_gates=True,stage_a_before_references=True)
(root/'independent_replay.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
