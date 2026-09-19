"""Independent metadata-only replay; does not import cohort selector."""
import json,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone
p=Path('research_log/T062C')
def read(n):return json.loads(subprocess.check_output(['git','show','HEAD:'+n]))
def digest(b):return hashlib.sha256(b).hexdigest()
m=json.loads((p/'manifest.json').read_bytes());a=json.loads((p/'freshness_audit.json').read_bytes())
assert digest((p/'manifest.json').read_bytes())==a['manifest_sha256']
for n,h in a['source_bindings'].items():assert digest(subprocess.check_output(['git','show','HEAD:'+n]))==h
old=read('research_log/T036A_cohort/manifest.json');excluded={r['low'].split('/')[-1] for r in old['selected']}
# Older independent accepted train ledger, not prepare.py's T036 inventory.
ledger=read('research_log/T032A_cohort/ledger.json')
pairs=[(r['low'].replace(chr(92),'/'),r['normal'].replace(chr(92),'/')) for r in ledger['candidates']]
assert len(pairs)==689 and len(set(pairs))==689
ranked=sorted((hashlib.sha256(('T062C-v1:'+lo).encode()).digest(),lo,no) for lo,no in pairs if lo.split('/')[-1] not in excluded)
expected=ranked[:100]
assert [(r['selection_hash'],r['low'],r['normal']) for r in m['selected']]==[(h.hex(),lo,no) for h,lo,no in expected]
receipt=read('research_log/T032A_result/audit/evaluation_receipt.json')
paths=receipt.get('normal_opens',receipt.get('opened_normals'));normals={x[x.index('Train/Normal/'):] for x in paths}
overlap=[r for r in m['selected'] if r['normal'] in normals]
assert overlap==a['actual_T032_reference_receipt_overlap'] and len(overlap)==18
old_used={r['low'] for r in ledger['candidates'] if r['excluded_by']}
new_used={lo for lo,no in pairs if no in normals}
allused=[r['low'] for r in m['selected'] if r['low'] in old_used|new_used]
assert allused==[r['low'] for r in a['prior_reference_used']] and len(allused)==49
assert not excluded & {r['low'].split('/')[-1] for r in m['selected']}
assert m['created_utc']<a['audit_utc'] and a['normal_image_opens']==a['optimizer_runs']==0
result=dict(status='PASS',verified_classification='BLOCKED',selected=100,original_T036_overlap=0,historical_reference_use=49,actual_T032_receipt_overlap=18,selection_replay='separate hashlib binary digest sort from T032 ledger',manifest_sha256=a['manifest_sha256'],verified_utc=datetime.now(timezone.utc).isoformat(),sample_actual_reference_used=overlap[:3])
with (p/'verification.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
