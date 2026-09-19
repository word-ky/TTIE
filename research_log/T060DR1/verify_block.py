"""Independent saved-score arithmetic and exact bindings; no model/reference reads."""
import hashlib,json,math
from pathlib import Path
p=Path('research_log/T060DR1');f=json.loads((p/'preflight.json').read_bytes())
s=json.loads(Path('research_log/T060B/selection.json').read_bytes())
assert f['selection_sha256']==hashlib.sha256(Path('research_log/T060B/selection.json').read_bytes()).hexdigest()
assert f['status']=='BLOCKED' and len(f['checks'])==27 and len(s['anchors'])==60
checked=[]
for i,r in enumerate(f['checks']):
    a=s['anchors'][i];assert r['order']==i and r['index']==a['index'] and r['bank_index']==a['bank_index']
    assert r['expected']==a['gate'] and r['low_hash']==a['state_before']['y0']
    assert r['actual']['active']==r['expected']['active'] and r['actual']['winner']==r['expected']['winner']
    for name in ['actual','expected']:
        for j,w in enumerate(r[name]['winner']):
            ev=(r[name]['scores'][j][w]-s['calibration']['tau'][w])/s['calibration']['scale'][w]
            assert math.isclose(ev,r[name]['evidence'][j],rel_tol=0,abs_tol=1e-14)
    errors={k:max(abs(x-y) for x,y in zip(sum(r['actual'][k],[]) if k=='scores' else r['actual'][k],sum(r['expected'][k],[]) if k=='scores' else r['expected'][k])) for k in ['scores','evidence']}
    assert errors==r['errors']
    for k in errors:assert r['equal'][k]==(errors[k]<=1e-5)
    assert all(r['equal'].values())==(i<26)
    checked.append(dict(order=i,bank_index=r['bank_index'],errors=errors,passed=all(r['equal'].values())))
assert checked[-1]['bank_index']==177 and checked[-1]['errors']['evidence']>1e-5
for k in ['source_clean_reads','reference_gradient_reads','metric_reads','target_domain_access','official_test_access','optimizer_steps']:assert f[k]==0
v=dict(status='PASS',preflight_status='BLOCKED',anchors_checked=27,anchors_passed=26,anchors_requested=60,first_failure_bank=177,max_score_difference=max(r['errors']['scores'] for r in checked),max_evidence_difference=max(r['errors']['evidence'] for r in checked),checks=checked,preflight_sha256=hashlib.sha256((p/'preflight.json').read_bytes()).hexdigest(),source_clean_reads=0,optimizer_steps=0)
(p/'block_verification.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({k:x for k,x in v.items() if k!='checks'}))
