"""Archive completed diagnostic evidence and both failed execution receipts."""
from pathlib import Path
import hashlib,json,tarfile,shutil
base=Path('/home/wenchang/asdasdsad/wjq/TTIE');backup=Path('/media/wenchang/F/wjq/TTIE/shared/t029a')
home=base/'shared/t029a';backup.mkdir(parents=True,exist_ok=True)
runs=['20260914-172718-ttie-t029a-preflight','20260914-172838-ttie-t029a-alignment',
      '20260914-173030-ttie-t029a-alignment','20260914-173750-ttie-t029a-alignment']
releases=['20260914-172659-ttie-t029a-alignment','20260914-173026-ttie-t029a-floatcheck','20260914-173746-ttie-t029a-diagnostic']
result=base/'runs'/runs[-1]/'artifacts/REFERENCE_GRADIENT_DIAGNOSTIC_ONLY'
assert json.loads((result/'independent_check.json').read_text())['status']=='PASS'
assert '[autodl] exit_code=0' in (base/'runs'/runs[-1]/'train.log').read_text()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
full=backup/'T029A_execution.tar'
with tarfile.open(full,'w') as t:
    for name in ['runs/'+r for r in runs]+['releases/'+r for r in releases]+['shared/t029a/reference_deployment.json']:
        t.add(base/name,arcname=name,filter=lambda info: None if '__pycache__' in info.name else info)
compact=home/'T029A_compact.tar.gz'
with tarfile.open(compact,'w:gz') as t:
    for r in runs:
        for name in ['train.log','run.sh','meta.json']:
            t.add(base/'runs'/r/name,arcname='runs/'+r+'/'+name)
    for name in ['states.json','gradients.pt','independent_samples.json','summary.json','receipt.json','independent_check.json']:
        t.add(result/name,arcname='evidence/'+name)
    pre=base/'runs'/runs[0]/'artifacts/preflight'
    for name in ['preflight.json','bound_states.pt']:t.add(pre/name,arcname='preflight/'+name)
    t.add(home/'reference_deployment.json',arcname='preflight/reference_deployment.json')
shutil.copy2(compact,backup/compact.name)
receipt=dict(full_backup=dict(path=str(full),bytes=full.stat().st_size,sha256=sha(full)),
    compact=dict(path=str(compact),bytes=compact.stat().st_size,sha256=sha(compact)),
    source_files={str(p.relative_to(base/'releases'/releases[-1])):sha(p)
        for p in (base/'releases'/releases[-1]).rglob('*') if p.is_file() and '__pycache__' not in str(p)},
    run_ids=runs,release_ids=releases)
for p in [home/'backup.json',backup/'backup.json']:p.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ['full_backup','compact']},indent=2))
