"""Independent arithmetic check of the saved preflight block; no model/data reads."""
import hashlib,json,math
from pathlib import Path

p=Path('research_log/T060D');result=json.loads((p/'preflight.json').read_bytes())
selection=json.loads(Path('research_log/T060B/selection.json').read_bytes())
assert result['status']=='BLOCKED' and len(result['checks'])==1
assert result['selection_sha256']==hashlib.sha256(Path('research_log/T060B/selection.json').read_bytes()).hexdigest()
r=result['checks'][0];anchor=selection['anchors'][0]
assert r['expected']==anchor['gate'] and r['bank_index']==anchor['bank_index']==1
assert r['low_hash']==anchor['state_before']['y0']
assert r['actual']['active']==r['expected']['active']==[True]*4
assert r['actual']['winner']==r['expected']['winner']==[0]*4
for name in ['actual','expected']:
    for i,w in enumerate(r[name]['winner']):
        expected=(r[name]['scores'][i][w]-selection['calibration']['tau'][w])/selection['calibration']['scale'][w]
        assert math.isclose(expected,r[name]['evidence'][i],rel_tol=0,abs_tol=1e-14)
scores=max(abs(x-y) for a,b in zip(r['actual']['scores'],r['expected']['scores']) for x,y in zip(a,b))
evidence=max(abs(x-y) for x,y in zip(r['actual']['evidence'],r['expected']['evidence']))
assert scores>0 and evidence>0
for k in ['source_clean_reads','reference_gradient_reads','metric_reads','target_domain_access','official_test_access','optimizer_steps']:assert result[k]==0
receipt=dict(status='PASS',preflight_status='BLOCKED',anchors_checked=1,anchors_requested=60,max_score_difference=scores,max_evidence_difference=evidence,active_equal=True,winner_equal=True,source_clean_reads=0,optimizer_steps=0,preflight_sha256=hashlib.sha256((p/'preflight.json').read_bytes()).hexdigest())
(p/'block_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
