from pathlib import Path
import json,tarfile,hashlib,shutil
root=Path('/home/wenchang/asdasdsad/wjq/TTIE');release=root/'releases/20260917-135320-ttie-t058ae-stagea';run=root/'runs/20260917-135400-ttie-t058ae-stagea';out=root/'shared/t058ae';backup=Path('/media/wenchang/F/wjq/TTIE/shared/t058ae')
binding=json.loads((release/'research_log/T058AE_source_binding.json').read_bytes());files={n:release/n for n in binding};files['research_log/T058AE_source_binding.json']=release/'research_log/T058AE_source_binding.json'
for n,h in binding.items():assert hashlib.sha256(files[n].read_bytes()).hexdigest()==h
for p in run.rglob('*'):
 if p.is_file():files['research_log/T058AE_result/'+str(p.relative_to(run))]=p
for p in (out/'metadata').glob('*'):
 if p.is_file():files['research_log/'+p.name]=p
old_run=root/'runs/20260917-131739-ttie-t058ad-shard0'
old_files=json.loads((run/'artifacts/T058AE/receipt.json').read_bytes())['shard0_before']['files']
for n,h in old_files.items():
 p=old_run/n;assert hashlib.sha256(p.read_bytes()).hexdigest()==h;files['research_log/T058AD_result/'+n]=p
archive=out/'T058AE_recovery.tar.gz' 
with tarfile.open(archive,'w:gz') as t:
 for name,p in sorted(files.items()):t.add(p,arcname=name,recursive=False)
shutil.copy2(archive,backup/archive.name)
sha=hashlib.sha256(archive.read_bytes()).hexdigest();assert hashlib.sha256((backup/archive.name).read_bytes()).hexdigest()==sha
receipt=dict(includes_immutable_AD_shard0=True,combined_gradient_rows=7346,files=len(files),bytes=archive.stat().st_size,sha256=sha,home=str(archive),backup=str(backup/archive.name),home_verified=True,F_backup_verified=True,local_archive_copy=False)
(out/'T058AE_recovery.json').write_text(json.dumps(receipt,indent=2)+'\n');shutil.copy2(out/'T058AE_recovery.json',backup/'T058AE_recovery.json');print(json.dumps(receipt))
