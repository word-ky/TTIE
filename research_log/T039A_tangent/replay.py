"""Independent stdlib masks/dot/norm/cosine/sign/aggregate replay; no model helpers."""
from pathlib import Path
import json,math,statistics,hashlib,sys
root=Path(sys.argv[1]);rows=json.loads((root/'stage_b/states.json').read_bytes());vectors=json.loads((root/'stage_b/vectors.json').read_bytes());summary=json.loads((root/'stage_b/summary.json').read_bytes())
f=json.loads((root/'stage_a/freeze.json').read_bytes());selection=json.loads((root/'stage_a/selection.json').read_bytes());receipt=json.loads((root/'stage_b/receipt.json').read_bytes())
assert len(rows)==len(vectors)==f['probes']==22038 and selection['states']==7346
assert hashlib.sha256((root/'stage_a/freeze.json').read_bytes()).hexdigest()==receipt['stage_a_freeze_sha256']
assert f['completed_utc']<receipt['started_utc']<min(r['utc'] for r in receipt['opened_source_targets']) and len(receipt['opened_source_targets'])==80
expected=[(i,j,g) for i,b in enumerate(selection['banks']) for j in range(b['states']) for g in [.75,1.,1.25]]
errors=[]
def check(a,b):
    if a is None or b is None:assert a is b;return
    err=abs(a-b);errors.append(err);assert err<=1e-6,(a,b)
for row,v,key in zip(rows,vectors,expected):
    assert (row['bank_index'],row['state_index'],row['gain'])==(v['bank_index'],v['state_index'],v['gain'])==key
    assert row['image_id']==selection['banks'][key[0]]['image_id'] and row['image_id'] in f['source_ids']
    active=v['active'];assert len(active)==4 and len(v['g_e'])==len(v['g_r'])==12
    assert all(math.isfinite(x) for x in v['g_e']+v['g_r'])
    masks=dict(legacy=active*2+[False]*4,gain=[False]*8+active,total=active*3)
    for group,mask in masks.items():
        e=[x for x,m in zip(v['g_e'],mask) if m];r=[x for x,m in zip(v['g_r'],mask) if m]
        en=math.sqrt(math.fsum(x*x for x in e));rn=math.sqrt(math.fsum(x*x for x in r));dot=math.fsum(a*b for a,b in zip(e,r));deg=en<=1e-12 or rn<=1e-12
        result=dict(active_count=len(e),energy_norm=en,reference_norm=rn,dot=dot,cosine=None if deg else dot/(en*rn),degenerate=deg,positive_dot=dot>0)
        for k,x in result.items():check(x,row['groups'][group][k])
        row['groups'][group]=result
    check(row['groups']['legacy']['dot']+row['groups']['gain']['dot'],row['groups']['total']['dot'])
def quantile(v,p):
    v=sorted(v);i=(len(v)-1)*p;lo=math.floor(i);hi=math.ceil(i);return v[lo]+(v[hi]-v[lo])*(i-lo)
def aggregates(data):
    result={}
    for group in ['legacy','gain','total']:
        items=[r['groups'][group] for r in data];valid=[r for r in items if not r['degenerate']];cs=[r['cosine'] for r in valid]
        result[group]=dict(count=len(items),nondegenerate=len(valid),cosine_mean=statistics.mean(cs) if cs else None,cosine_median=statistics.median(cs) if cs else None,cosine_p10=quantile(cs,.1) if cs else None,cosine_p90=quantile(cs,.9) if cs else None,
            positive_dot_fraction=sum(r['positive_dot'] for r in valid)/len(valid) if valid else None,energy_zero_fraction=sum(r['energy_norm']<=1e-12 for r in items)/len(items),reference_zero_fraction=sum(r['reference_norm']<=1e-12 for r in items)/len(items),either_zero_fraction=sum(r['degenerate'] for r in items)/len(items),
            **{k+'_median':statistics.median(r[k] for r in items) for k in ['energy_norm','reference_norm','dot']})
    return result
overall=aggregates(rows)
for actual,published in [(overall,summary['overall'])]+[(aggregates([r for r in rows if r['gain']==g]),summary['by_gain'][str(g)]) for g in [.75,1.,1.25]]:
    for group,values in actual.items():
        for k,v in values.items():check(v,published[group][k])
l,g=overall['legacy'],overall['gain'];yes=l['positive_dot_fraction']-g['positive_dot_fraction']>=.2 and l['cosine_median']-g['cosine_median']>=.25
verdict='source gain-tangent deficit supported' if yes else 'source gain-tangent deficit not supported / real-domain effect remains plausible'
assert verdict==summary['classification']
result=dict(status='PASS',canonical_states=7346,probes=22038,coordinate_groups=66114,scalar_checks=len(errors),max_abs_error=max(errors),classification=verdict,source_targets_opened_after_complete_freeze=True,independent_mask_rebuild=True)
(root/'independent_replay.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
