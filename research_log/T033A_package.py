"""Preserve full outputs and bound baseline assets; export compact scalar evidence."""
from pathlib import Path
import json,hashlib,tarfile,shutil
base=Path('/home/wenchang/asdasdsad/wjq/TTIE');home=base/'shared/t033a';backup=Path('/media/wenchang/F/wjq/TTIE/shared/t033a');backup.mkdir(parents=True,exist_ok=True)
run='20260915-005309-ttie-t033a-retinex';evaluation='20260915-005425-ttie-t033a-eval';release='20260915-005241-ttie-t033a-retinex'
audit=base/'runs'/run/'artifacts/audit';src=base/'releases'/release
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for r in [run,evaluation]:assert '[autodl] exit_code=0' in (base/'runs'/r/'train.log').read_text()
f=json.loads((audit/'freeze.json').read_bytes());dep=json.loads((home/'evaluation_deployment.json').read_bytes());ev=json.loads((audit/'evaluation_receipt.json').read_bytes())
assert sha(audit/'freeze.json')==dep['freeze_sha256']==ev['freeze_sha256'] and f['completed_utc']<dep['created_utc']<ev['normal_opens'][0]['utc']
for r in f['rows']:assert sha(audit/r['output'])==r['file_sha256']
b=json.loads((src/'research_log/T033A/binding.json').read_bytes())
for path,h in b['files'].items():assert sha(Path(path) if Path(path).is_absolute() else src/path)==h
full=backup/'T033A_execution.tar'
with tarfile.open(full,'w') as t:
    for name in ['runs/'+run,'runs/'+evaluation,'releases/'+release,'shared/t033a/evaluation_deployment.json']:
        t.add(base/name,arcname=name,filter=lambda i:None if '__pycache__' in i.name or '.pytest_cache' in i.name else i)
    for path in b['files']:
        p=Path(path)
        if p.is_absolute():t.add(p,arcname=str(p.relative_to(base)))
compact=home/'T033A_compact.tar.gz'
with tarfile.open(compact,'w:gz') as t:
    for r in [run,evaluation]:
        for n in ['train.log','run.sh','meta.json']:t.add(base/'runs'/r/n,arcname='runs/'+r+'/'+n)
    for n in ['config.json','freeze.json','metrics.csv','summary.json','evaluation_receipt.json','outputs/receipt.json']:t.add(audit/n,arcname='audit/'+n)
    t.add(home/'evaluation_deployment.json',arcname='evaluation_deployment.json')
shutil.copy2(compact,backup/compact.name)
r=dict(full=dict(path=str(full),bytes=full.stat().st_size,sha256=sha(full)),compact=dict(path=str(compact),bytes=compact.stat().st_size,sha256=sha(compact)),
    output_hashes_verified=100,source_checkpoint_postrun_verified=True,barrier_verified=True,
    deployed_source={str(p.relative_to(src)):sha(p) for p in src.rglob('*') if p.is_file() and '__pycache__' not in str(p) and '.pytest_cache' not in str(p)})
for p in [home/'backup.json',backup/'backup.json']:p.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='deployed_source'},indent=2))
