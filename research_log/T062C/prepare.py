"""Literal T062-C cohort freeze, then historical reference-use audit; no images."""
import hashlib,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
HERE=Path('research_log/T062C')
def sha(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def choose(records,excluded):
    rows=[dict(low=r['low'].replace(chr(92),'/'),normal=r['normal'].replace(chr(92),'/')) for r in records if r['low'].split('/')[-1] not in excluded]
    for r in rows:r['selection_hash']=sha(('T062C-v1:'+r['low']).encode('utf-8'))
    return sorted(rows,key=lambda r:r['selection_hash'])[:100]
def main():
    bindings={}
    def read(n):
        b=subprocess.check_output(['git','show','HEAD:'+n]);bindings[n]=sha(b);return json.loads(b)
    prior=read('research_log/T036A_cohort/manifest.json')
    ledger=read('research_log/T036A_cohort/exclusions.json')
    records=ledger['candidates'];assert len(records)==689
    assert all(r['low'].startswith('Train/Low/') and r['normal'].startswith('Train/Normal/') for r in records)
    excluded={r['low'].split('/')[-1] for r in prior['selected']}
    rows=choose(records,excluded);assert len(rows)==100
    assert not excluded & {r['low'].split('/')[-1] for r in rows}
    manifest=dict(task='T062-C',created_utc=utc(),relative_root='LOL-v2/Real_captured/',selection='SHA256(T062C-v1: + normalized_relative_low_path); ascending; exclude T036 filenames only',train_pairs=689,eligible=589,selected=rows,source_bindings=bindings,normal_image_opens=0)
    with (HERE/'manifest.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(manifest,f,indent=2);f.write('\n')
    manifest_hash=sha((HERE/'manifest.json').read_bytes())
    # Audit AFTER freezing the exact requested cohort; never change selection.
    by_low={r['low']:r for r in records}
    used=[dict(**r,prior_reference_use=by_low[r['low']]['excluded_by']) for r in rows if by_low[r['low']]['excluded_by']]
    # Actual accepted T032 evaluation access paths, without metric fields.
    receipt=read('research_log/T032A_result/audit/evaluation_receipt.json')
    paths=receipt.get('normal_opens',receipt.get('opened_normals'))
    normals={x[x.index('Train/Normal/'):] for x in paths}
    proven32=[r for r in rows if r['normal'] in normals]
    audit=dict(task='T062-C',status='BLOCKED' if used else 'COHORT_READY',manifest_sha256=manifest_hash,manifest_frozen_utc=manifest['created_utc'],audit_utc=utc(),original_development_overlap=0,prior_reference_used_count=len(used),prior_reference_used=used,actual_T032_reference_receipt_overlap=proven32,source_bindings=bindings,normal_image_opens=0,optimizer_runs=0,official_test_access=0,reason='Literal prescribed selection conflicts with previously unused requirement' if used else None)
    with (HERE/'freshness_audit.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(audit,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ['prior_reference_used','source_bindings','actual_T032_reference_receipt_overlap']}))
    print('Actual T032 receipt overlap',len(proven32))
if __name__=='__main__':main()
