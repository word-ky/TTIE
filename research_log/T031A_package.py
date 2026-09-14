"""Persist completed support and quarantined reference evidence."""
from pathlib import Path
import json,hashlib,tarfile,shutil
base=Path('/home/wenchang/asdasdsad/wjq/TTIE');home=base/'shared/t031a';backup=Path('/media/wenchang/F/wjq/TTIE/shared/t031a')
home.mkdir(parents=True,exist_ok=True);backup.mkdir(parents=True,exist_ok=True)
support_run='20260914-194203-ttie-t031a-support';ref_run='20260914-194331-ttie-t031a-reference';release='20260914-194158-ttie-t031a-support'
support=base/'runs'/support_run/'artifacts/support';reference=base/'runs'/ref_run/'artifacts/REFERENCE_GRADIENT_DIAGNOSTIC_ONLY'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
f=json.loads((support/'freeze.json').read_bytes());r=json.loads((reference/'receipt.json').read_bytes())
assert sha(support/'freeze.json')==r['support_freeze_sha256']
assert all(o['utc']>f['completed_utc'] for o in r['opened']) and sum(o['normal'] for o in r['opened'])==100
assert sha(support/'scores.json')==f['scores_sha256'] and sha(support/'bound_features.pt')==f['bound_features_sha256']
assert sha(reference/'validity.json')==r['validity_sha256'] and sha(reference/'gradients.pt')==r['gradients_sha256']
for run in [support_run,ref_run]:assert '[autodl] exit_code=0' in (base/'runs'/run/'train.log').read_text()
full=backup/'T031A_execution.tar'
with tarfile.open(full,'w') as t:
    for name in ['runs/'+support_run,'runs/'+ref_run,'releases/'+release,'shared/t030a/assets.json']:
        t.add(base/name,arcname=name,filter=lambda info:None if '__pycache__' in info.name or '.pytest_cache' in info.name else info)
compact=home/'T031A_compact.tar.gz'
with tarfile.open(compact,'w:gz') as t:
    t.add(support,arcname='support');t.add(reference,arcname='REFERENCE_GRADIENT_DIAGNOSTIC_ONLY')
    for run in [support_run,ref_run]:
        for n in ['train.log','run.sh','meta.json']:t.add(base/'runs'/run/n,arcname='runs/'+run+'/'+n)
    t.add(base/'shared/t030a/assets.json',arcname='assets.json')
shutil.copy2(compact,backup/compact.name)
receipt=dict(full=dict(path=str(full),bytes=full.stat().st_size,sha256=sha(full)),compact=dict(path=str(compact),bytes=compact.stat().st_size,sha256=sha(compact)),
    barrier_verified=True,source_files={str(p.relative_to(base/'releases'/release)):sha(p) for p in (base/'releases'/release).rglob('*') if p.is_file() and '__pycache__' not in str(p) and '.pytest_cache' not in str(p)})
for p in [home/'backup.json',backup/'backup.json']:p.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='source_files'},indent=2))
