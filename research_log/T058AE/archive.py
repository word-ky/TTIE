from pathlib import Path
import json,hashlib,tarfile,shutil
root=Path('/home/wenchang/asdasdsad/wjq/TTIE');run=root/'runs/20260917-135400-ttie-t058ae-stagea';out=root/'shared/t058ae';backup=Path('/media/wenchang/F/wjq/TTIE/shared/t058ae')
out.mkdir(parents=True,exist_ok=True);backup.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt=json.loads((run/'artifacts/T058AE/receipt.json').read_bytes())
for n,h in receipt['source_bindings'].items():assert sha(root/'releases/20260917-135320-ttie-t058ae-stagea'/n)==h
for n,h in receipt['historical_files_unchanged'].items():assert sha(root/'runs/20260917-023631-ttie-t058a-stage-a'/n)==h
for n,h in receipt['prior_verifier_files_unchanged'].items():assert sha(root/'runs/20260917-131739-ttie-t058ad-shard0'/n)==h
marker=json.loads((run/'artifacts/T058AE/complete.json').read_bytes())
assert marker['combined_rows']==7346 and marker['combined_reopen_sha256']==sha(run/'artifacts/T058AE/combined_reopen.json')
path=out/'T058AE_evidence.tar.gz'
with tarfile.open(path,'w:gz') as t:
    for p in run.rglob('*'):
        if p.is_file():t.add(p,arcname=str(p.relative_to(run)))
shutil.copy2(path,backup/path.name);assert sha(path)==sha(backup/path.name)
r=dict(path=str(path),backup=str(backup/path.name),bytes=path.stat().st_size,sha256=sha(path),source197_unchanged=True,historical8_unchanged=True,prior_shard27_unchanged=True,files={str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file()})
(out/'T058AE_archives.json').write_text(json.dumps(r,indent=2)+'\n');shutil.copy2(out/'T058AE_archives.json',backup/'T058AE_archives.json');print(json.dumps({k:v for k,v in r.items() if k!='files'}))
