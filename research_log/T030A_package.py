"""Package final T030-A artifacts; retain full outputs remotely and compact evidence locally."""
import argparse,json,hashlib,tarfile,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--evaluation-run',required=True);a=p.parse_args()
base=Path('/home/wenchang/asdasdsad/wjq/TTIE');home=base/'shared/t030a';backup=Path('/media/wenchang/F/wjq/TTIE/shared/t030a');backup.mkdir(parents=True,exist_ok=True)
run='20260914-183308-ttie-t030a-guard';release='20260914-183209-ttie-t030a-guard';audit=base/'runs'/run/'artifacts/audit'
assert json.loads((audit/'independent_replay.json').read_bytes())['status']=='PASS'
assert json.loads((audit/'evaluation_receipt.json').read_bytes())['pairs']==100
assert '[autodl] exit_code=0' in (base/'runs'/run/'train.log').read_text()
assert '[autodl] exit_code=0' in (base/'runs'/a.evaluation_run/'train.log').read_text()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
freeze=json.loads((audit/'freeze.json').read_bytes())
for r in freeze['rows']:
    for n,h in r['files'].items():assert sha(audit/f'{r["index"]:03d}'/n)==h['sha256']
full=backup/'T030A_execution.tar'
with tarfile.open(full,'w') as t:
    for name in ['runs/'+run,'runs/'+a.evaluation_run,'releases/'+release,'shared/t030a/assets.json','shared/t030a/low_deployment.json','shared/t030a/reference_deployment.json']:
        t.add(base/name,arcname=name,filter=lambda info:None if '__pycache__' in info.name else info)
compact=home/'T030A_compact.tar.gz'
with tarfile.open(compact,'w:gz') as t:
    for r in [run,a.evaluation_run]:
        for n in ['train.log','run.sh','meta.json']:t.add(base/'runs'/r/n,arcname='runs/'+r+'/'+n)
    for n in ['config.json','freeze.json','independent_replay.json','metrics.csv','summary.json','evaluation_receipt.json']:
        t.add(audit/n,arcname='audit/'+n)
    for i in range(100):
        for n in ['decision.json','trajectory.pt']:t.add(audit/f'{i:03d}'/n,arcname=f'audit/{i:03d}/'+n)
    for n in ['assets.json','low_deployment.json','reference_deployment.json']:t.add(home/n,arcname='preparation/'+n)
shutil.copy2(compact,backup/compact.name)
record=dict(full_backup=dict(path=str(full),bytes=full.stat().st_size,sha256=sha(full)),
    compact=dict(path=str(compact),bytes=compact.stat().st_size,sha256=sha(compact)),
    original_and_guarded_outputs=200,artifact_files_verified=400,
    deployed_source={str(f.relative_to(base/'releases'/release)):sha(f) for f in (base/'releases'/release).rglob('*') if f.is_file() and '__pycache__' not in str(f)})
for dest in [home/'backup.json',backup/'backup.json']:dest.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='deployed_source'},indent=2))
