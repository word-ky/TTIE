from pathlib import Path
import json,hashlib,tarfile,shutil
root=Path('/home/wenchang/asdasdsad/wjq/TTIE');run=root/'runs/20260917-092633-ttie-t058ab-full';out=root/'shared/t058ab';backup=Path('/media/wenchang/F/wjq/TTIE/shared/t058ab')
out.mkdir(parents=True,exist_ok=True);backup.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt=json.loads((run/'artifacts/T058AB/receipt.json').read_bytes())
for n,h in receipt['source_bindings'].items():assert sha(root/'releases/20260917-092610-ttie-t058ab-full'/n)==h
for n,h in receipt['historical_files_unchanged'].items():assert sha(root/'runs/20260917-023631-ttie-t058a-stage-a'/n)==h
for n,h in receipt['prior_verifier_files_unchanged'].items():assert sha(root/'runs/20260917-082811-ttie-t058aa-clamp'/n)==h
path=out/'T058AB_evidence.tar.gz'
with tarfile.open(path,'w:gz') as t:
    for p in run.rglob('*'):
        if p.is_file():t.add(p,arcname=str(p.relative_to(run)))
shutil.copy2(path,backup/path.name);assert sha(path)==sha(backup/path.name)
r=dict(path=str(path),backup=str(backup/path.name),bytes=path.stat().st_size,sha256=sha(path),source175_unchanged=True,historical8_unchanged=True,prior_verifier6_unchanged=True,files={str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file()})
(out/'T058AB_archives.json').write_text(json.dumps(r,indent=2)+'\n');shutil.copy2(out/'T058AB_archives.json',backup/'T058AB_archives.json');print(json.dumps(r))
