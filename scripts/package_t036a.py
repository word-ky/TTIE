"""Archive the completed qualification without rerunning either method."""
from pathlib import Path
import argparse,hashlib,json,tarfile,shutil,datetime
p=argparse.ArgumentParser();p.add_argument('--eval-run',required=True);a=p.parse_args()
base=Path('/home/wenchang/asdasdsad/wjq/TTIE');backup=Path('/media/wenchang/F/wjq/TTIE/shared/t036a')
run=base/'runs/20260915-035341-ttie-t036a-common';pre=base/'runs/20260915-035138-ttie-t036a-preflight';evaluation=base/'runs'/a.eval_run
out=run/'artifacts/audit';dest=base/'shared/t036a';backup.mkdir(parents=True,exist_ok=True)
assert '[autodl] exit_code=0' in (run/'train.log').read_text() and '[autodl] exit_code=0' in (evaluation/'train.log').read_text()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
full=backup/'T036A_execution.tar'
with tarfile.open(full,'w') as t:
    for d in [run,pre,evaluation]:t.add(d,arcname=d.name)
    for n in ['low_deployment.json','reference_deployment.json']:t.add(dest/n,arcname=n)
compact=dest/'T036A_compact.tar.gz'
with tarfile.open(compact,'w:gz') as t:
    for f in sorted(out.rglob('*')):
        if f.is_file() and f.name!='output.pt':t.add(f,arcname='audit/'+str(f.relative_to(out)))
    for d,label in [(run,'run'),(pre,'preflight'),(evaluation,'evaluation')]:
        for n in ['meta.json','run.sh','train.log']:t.add(d/n,arcname=label+'/'+n)
    for n in ['low_deployment.json','reference_deployment.json']:t.add(dest/n,arcname=n)
shutil.copy2(compact,backup/compact.name)
r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run=run.name,evaluation_run=evaluation.name,
       full=dict(path=str(full),bytes=full.stat().st_size,sha256=sha(full)),compact=dict(path=str(compact),bytes=compact.stat().st_size,sha256=sha(compact)),compact_backup_sha256=sha(backup/compact.name))
assert r['compact']['sha256']==r['compact_backup_sha256']
(dest/'T036A_archives.json').write_text(json.dumps(r,indent=2)+'\n');shutil.copy2(dest/'T036A_archives.json',backup/'T036A_archives.json')
print(json.dumps(r),flush=True)
