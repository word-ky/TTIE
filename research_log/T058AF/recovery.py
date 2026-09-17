from pathlib import Path
import json,tarfile,hashlib,shutil
root=Path('/home/wenchang/asdasdsad/wjq/TTIE');release=root/'releases/20260917-153229-ttie-t058af-stageb';run=root/'runs/20260917-153308-ttie-t058af-stageb';out=root/'shared/t058af';backup=Path('/media/wenchang/F/wjq/TTIE/shared/t058af')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
binding=json.loads((release/'research_log/T058AF_source_binding.json').read_bytes());files={n:release/n for n in binding};files['research_log/T058AF_source_binding.json']=release/'research_log/T058AF_source_binding.json'
for n,h in binding.items():assert sha(files[n])==h
for p in run.rglob('*'):
 if p.is_file():files['research_log/T058AF_result/'+str(p.relative_to(run))]=p
for folder in ['metadata','export']:
 for p in (out/folder).glob('*'):
  if p.is_file():files['research_log/'+p.name]=p
prior=root/'shared/t058ae/T058AE_recovery.tar.gz';assert sha(prior)=='850b7a557345d5416265fc91e50c73c09eba224aa3ca3b9a64159e54da457bf2';files['research_log/T058AE_recovery.tar.gz']=prior
archive=out/'T058AF_recovery.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for name,p in sorted(files.items()):t.add(p,arcname=name,recursive=False)
shutil.copy2(archive,backup/archive.name);assert sha(archive)==sha(backup/archive.name)
receipt=dict(includes_immutable_stage_a_recovery=True,source_reference_rows=7346,files=len(files),bytes=archive.stat().st_size,sha256=sha(archive),home=str(archive),backup=str(backup/archive.name),home_verified=True,F_backup_verified=True,local_archive_copy=False)
(out/'T058AF_recovery.json').write_text(json.dumps(receipt,indent=2)+'\n');shutil.copy2(out/'T058AF_recovery.json',backup/'T058AF_recovery.json');print(json.dumps(receipt))
