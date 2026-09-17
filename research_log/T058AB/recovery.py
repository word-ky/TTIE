from pathlib import Path
import json,tarfile,hashlib,shutil
root=Path('/home/wenchang/asdasdsad/wjq/TTIE');release=root/'releases/20260917-092610-ttie-t058ab-full';run=root/'runs/20260917-092633-ttie-t058ab-full';out=root/'shared/t058ab';backup=Path('/media/wenchang/F/wjq/TTIE/shared/t058ab')
binding=json.loads((release/'research_log/T058AB_source_binding.json').read_bytes());files={n:release/n for n in binding};files['research_log/T058AB_source_binding.json']=release/'research_log/T058AB_source_binding.json'
for n,h in binding.items():assert hashlib.sha256(files[n].read_bytes()).hexdigest()==h
for p in run.rglob('*'):
 if p.is_file():files['research_log/T058AB_result/'+str(p.relative_to(run))]=p
for p in (out/'metadata').glob('*'):
 if p.is_file():files['research_log/'+p.name]=p
archive=out/'T058AB_recovery.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for name,p in sorted(files.items()):t.add(p,arcname=name,recursive=False)
shutil.copy2(archive,backup/archive.name)
sha=hashlib.sha256(archive.read_bytes()).hexdigest();assert hashlib.sha256((backup/archive.name).read_bytes()).hexdigest()==sha
receipt=dict(files=len(files),bytes=archive.stat().st_size,sha256=sha,home=str(archive),backup=str(backup/archive.name),home_verified=True,F_backup_verified=True,local_archive_copy=False)
(out/'T058AB_recovery.json').write_text(json.dumps(receipt,indent=2)+'\n');shutil.copy2(out/'T058AB_recovery.json',backup/'T058AB_recovery.json');print(json.dumps(receipt))
