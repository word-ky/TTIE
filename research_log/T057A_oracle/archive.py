import argparse,json,hashlib,tarfile,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--evaluation-run',required=True);a=p.parse_args();base=Path('/home/wenchang/asdasdsad/wjq/TTIE');run=base/'runs/20260917-010959-ttie-t057a-oracle';pre=base/'runs/20260917-010902-ttie-t057a-preflight';ev=base/'runs'/a.evaluation_run;out=base/'shared/t057a';out.mkdir(exist_ok=True);backup=Path('/media/wenchang/F/wjq/TTIE/shared/t057a');backup.mkdir(exist_ok=True);audit=run/'artifacts/REFERENCE_ORACLE_ONLY'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
f=json.loads((audit/'freeze.json').read_bytes());pf=json.loads((pre/'artifacts/preflight/preflight.json').read_bytes())
for row in f['rows']:
 for name,h in row['files'].items():assert sha(audit/f'{row["index"]:03d}'/name)==h
for row in pf['rows']:assert sha(pre/'artifacts/preflight'/row['start_file'])==row['start_file_sha256']
config=json.loads((audit/'config.json').read_bytes());release=base/'releases/20260917-010837-ttie-t057a-chroma'
for name,h in config['source_binding'].items():assert sha(release/name)==h
full=backup/'T057A_full.tar'
with tarfile.open(full,'w') as t:
 for path,name in [(run,'oracle_run'),(pre,'preflight_run'),(ev,'evaluation_run')]:t.add(path,arcname=name)
compact=out/'T057A_compact.tar.gz'
with tarfile.open(compact,'w:gz') as t:
 for file in audit.glob('*.json'):t.add(file,arcname='audit/'+file.name)
 for row in f['rows']:t.add(audit/f'{row["index"]:03d}'/'history.pt',arcname=f'audit/{row["index"]:03d}/history.pt')
 for file in (pre/'artifacts/preflight').iterdir():
  if not file.name.startswith('start_'):t.add(file,arcname='preflight/'+file.name)
 for path,name in [(run,'oracle_run'),(pre,'preflight_run'),(ev,'evaluation_run')]:
  for file in ['meta.json','run.sh','train.log']:t.add(path/file,arcname=name+'/'+file)
shutil.copy2(compact,backup/compact.name);assert sha(compact)==sha(backup/compact.name)
metadata=out/'T057A_metadata.tar.gz'
with tarfile.open(compact) as src,tarfile.open(metadata,'w:gz') as dst:
 for m in src.getmembers():
  if m.isfile() and m.name.endswith(('.json','.log','.sh')):dst.addfile(m,src.extractfile(m))
r=dict(all_output_history_start_hashes_unchanged=True,source_bindings_unchanged=True,full=dict(path=str(full),bytes=full.stat().st_size,sha256=sha(full)),compact=dict(path=str(compact),bytes=compact.stat().st_size,sha256=sha(compact)),metadata=dict(path=str(metadata),bytes=metadata.stat().st_size,sha256=sha(metadata)),compact_backup=str(backup/compact.name));(out/'T057A_archives.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
