"""Preserve full GPU outputs on F and compact reproducible evidence on both roots."""
import argparse,json,hashlib,tarfile,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--evaluation-run',required=True);a=p.parse_args()
base=Path('/home/wenchang/asdasdsad/wjq/TTIE');home=base/'shared/t032a';backup=Path('/media/wenchang/F/wjq/TTIE/shared/t032a');backup.mkdir(parents=True,exist_ok=True)
run='20260914-232831-ttie-t032a-trust';radius_run='20260914-232650-ttie-t032a-radius';release='20260914-232618-ttie-t032a-support'
failed_run='20260914-233346-ttie-t032a-eval'
audit=base/'runs'/run/'artifacts/audit';radius=base/'runs'/radius_run/'artifacts/radius'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((audit/'independent_replay.json').read_bytes())['status']=='PASS'
assert json.loads((audit/'evaluation_receipt.json').read_bytes())['pairs']==100
for r in [run,radius_run,a.evaluation_run]:assert '[autodl] exit_code=0' in (base/'runs'/r/'train.log').read_text()
freeze=json.loads((audit/'freeze.json').read_bytes());dep=json.loads((home/'reference_deployment.json').read_bytes())
assert freeze['completed_utc']<dep['started_utc'] and dep['freeze_sha256']==sha(audit/'freeze.json')
for r in freeze['rows']:
    for n,h in r['files'].items():assert sha(audit/f'{r["index"]:03d}'/n)==h['sha256']
full=backup/'T032A_execution.tar'
with tarfile.open(full,'w') as t:
    for name in ['runs/'+run,'runs/'+radius_run,'runs/'+a.evaluation_run,'runs/'+failed_run,'releases/'+release,'shared/t032a/assets.json','shared/t032a/low_deployment.json','shared/t032a/reference_deployment.json']:
        t.add(base/name,arcname=name,filter=lambda i:None if '__pycache__' in i.name or '.pytest_cache' in i.name else i)
compact=home/'T032A_compact.tar.gz'
with tarfile.open(compact,'w:gz') as t:
    for r in [run,radius_run,a.evaluation_run,failed_run]:
        for n in ['train.log','run.sh','meta.json']:t.add(base/'runs'/r/n,arcname='runs/'+r+'/'+n)
    for n in ['config.json','freeze.json','independent_replay.json','metrics.csv','summary.json','evaluation_receipt.json']:t.add(audit/n,arcname='audit/'+n)
    for i in range(100):
        for n in ['decision.json','trajectory.pt']:t.add(audit/f'{i:03d}'/n,arcname=f'audit/{i:03d}/'+n)
    for n in ['assets.json','low_deployment.json','reference_deployment.json']:t.add(home/n,arcname='preparation/'+n)
    for n in ['source.pt','freeze.json','spec.json','cross_distances.json']:t.add(radius/n,arcname='radius/'+n)
shutil.copy2(compact,backup/compact.name)
record=dict(full_backup=dict(path=str(full),bytes=full.stat().st_size,sha256=sha(full)),compact=dict(path=str(compact),bytes=compact.stat().st_size,sha256=sha(compact)),
    artifact_files_verified=400,barrier_verified=True,deployed_source={str(f.relative_to(base/'releases'/release)):sha(f) for f in (base/'releases'/release).rglob('*') if f.is_file() and '__pycache__' not in str(f) and '.pytest_cache' not in str(f)})
for dest in [home/'backup.json',backup/'backup.json']:dest.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='deployed_source'},indent=2))
