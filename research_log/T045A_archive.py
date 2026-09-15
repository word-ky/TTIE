from pathlib import Path
import json,hashlib,tarfile,shutil
base=Path('/home/wenchang/asdasdsad/wjq/TTIE');inference=base/'runs/20260915-161153-ttie-t045a-snr';evaluation=base/'runs/20260915-161312-ttie-t045a-eval';audit=inference/'artifacts/audit';out=base/'shared/t045a';backup=Path('/media/wenchang/F/wjq/TTIE/shared/t045a');backup.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
f=json.loads((audit/'freeze.json').read_bytes());b=json.loads((audit/'config.json').read_bytes())['binding']
for row in f['rows']:assert sha(audit/row['output'])==row['file_sha256']
release=base/'releases/20260915-161131-ttie-t045a-snr'
for name,h in b['files'].items():assert sha(Path(name) if name.startswith('/') else release/name)==h,name
full=backup/'T045A_full.tar'
with tarfile.open(full,'w') as t:
    t.add(inference,arcname='inference');t.add(evaluation,arcname='evaluation');t.add(out/'evaluation_deployment.json',arcname='evaluation_deployment.json')
compact=out/'T045A_compact.tar.gz'
with tarfile.open(compact,'w:gz') as t:
    for file in audit.glob('*.json'):t.add(file,arcname='audit/'+file.name)
    t.add(audit/'metrics.csv',arcname='audit/metrics.csv');t.add(audit/'outputs/receipt.json',arcname='audit/outputs/receipt.json')
    for run,name in [(inference,'inference'),(evaluation,'evaluation')]:
        for file in ['run.sh','train.log','meta.json']:t.add(run/file,arcname=name+'/'+file)
    t.add(out/'evaluation_deployment.json',arcname='evaluation_deployment.json')
shutil.copy2(compact,backup/compact.name);assert sha(compact)==sha(backup/compact.name)
r=dict(all100_output_hashes_unchanged=True,source_checkpoint_unchanged=True,full=dict(path=str(full),bytes=full.stat().st_size,sha256=sha(full)),compact=dict(path=str(compact),bytes=compact.stat().st_size,sha256=sha(compact)),compact_backup=str(backup/compact.name))
(out/'T045A_archives.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
