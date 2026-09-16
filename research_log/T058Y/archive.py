from pathlib import Path
import json,hashlib,tarfile,shutil
root=Path('/home/wenchang/asdasdsad/wjq/TTIE');run=root/'runs/20260917-062907-ttie-t058y-shadow';out=root/'shared/t058y';backup=Path('/media/wenchang/F/wjq/TTIE/shared/t058y')
out.mkdir(parents=True,exist_ok=True);backup.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt=json.loads((run/'artifacts/T058Y/receipt.json').read_bytes())
for n,h in receipt['source_bindings'].items():assert sha(root/'releases/20260917-062841-ttie-t058y-shadow'/n)==h
for n,h in receipt['historical_files_unchanged'].items():assert sha(root/'runs/20260917-023631-ttie-t058a-stage-a'/n)==h
for n,h in receipt['prior_verifier_files_unchanged'].items():assert sha(root/'runs/20260917-054710-ttie-t058x-backend'/n)==h
path=out/'T058Y_evidence.tar.gz'
with tarfile.open(path,'w:gz') as t:
    for p in run.rglob('*'):
        if p.is_file():t.add(p,arcname=str(p.relative_to(run)))
shutil.copy2(path,backup/path.name);assert sha(path)==sha(backup/path.name)
r=dict(path=str(path),backup=str(backup/path.name),bytes=path.stat().st_size,sha256=sha(path),source156_unchanged=True,historical8_unchanged=True,prior_verifier7_unchanged=True,files={str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file()})
(out/'T058Y_archives.json').write_text(json.dumps(r,indent=2)+'\n');shutil.copy2(out/'T058Y_archives.json',backup/'T058Y_archives.json');print(json.dumps(r))
