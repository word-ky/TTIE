import argparse,json,hashlib,tarfile,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--evaluation-run',required=True);a=p.parse_args()
base=Path('/home/wenchang/asdasdsad/wjq/TTIE');run=base/'runs/20260916-200429-ttie-t054a-oracle';pre=base/'runs/20260916-200318-ttie-t054a-preflight';ev=base/'runs'/a.evaluation_run;out=base/'shared/t054a';out.mkdir(exist_ok=True);backup=Path('/media/wenchang/F/wjq/TTIE/shared/t054a');backup.mkdir(exist_ok=True)
audit=run/'artifacts/REFERENCE_ORACLE_ONLY'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
f=json.loads((audit/'freeze.json').read_bytes())
for row in f['rows']:
    for name,h in row['files'].items():assert sha(audit/f'{row["index"]:03d}'/name)==h
config=json.loads((audit/'config.json').read_bytes());release=base/'releases/20260916-200256-ttie-t054a-detail'
for name,h in config['source_binding'].items():assert sha(release/name)==h,name
full=backup/'T054A_full.tar'
with tarfile.open(full,'w') as t:
    t.add(run,arcname='oracle_run');t.add(pre,arcname='preflight_run');t.add(ev,arcname='evaluation_run')
compact=out/'T054A_compact.tar.gz'
with tarfile.open(compact,'w:gz') as t:
    for file in audit.glob('*.json'):t.add(file,arcname='audit/'+file.name)
    for row in f['rows']:t.add(audit/f'{row["index"]:03d}'/'history.pt',arcname=f'audit/{row["index"]:03d}/history.pt')
    t.add(pre/'artifacts/preflight',arcname='preflight')
    for path,name in [(run,'oracle_run'),(pre,'preflight_run'),(ev,'evaluation_run')]:
        for file in ['meta.json','run.sh','train.log']:t.add(path/file,arcname=name+'/'+file)
shutil.copy2(compact,backup/compact.name);assert sha(compact)==sha(backup/compact.name)
r=dict(all_output_history_hashes_unchanged=True,source_bindings_unchanged=True,full=dict(path=str(full),bytes=full.stat().st_size,sha256=sha(full)),compact=dict(path=str(compact),bytes=compact.stat().st_size,sha256=sha(compact)),compact_backup=str(backup/compact.name))
(out/'T054A_archives.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
