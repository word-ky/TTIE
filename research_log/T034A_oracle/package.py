"""Archive the completed T034 diagnostic; never rerun optimization or metrics."""
from pathlib import Path
import hashlib, json, tarfile, shutil, datetime

base=Path('/home/wenchang/asdasdsad/wjq/TTIE')
backup=Path('/media/wenchang/F/wjq/TTIE/shared/t034a')
run=base/'runs/20260915-014747-ttie-t034a-oracle'
pre=base/'runs/20260915-014157-ttie-t034a-preflight'
out=run/'artifacts/REFERENCE_ORACLE_ONLY'
dest=base/'shared/t034a';dest.mkdir(parents=True,exist_ok=True);backup.mkdir(parents=True,exist_ok=True)
assert '[autodl] exit_code=0' in (run/'train.log').read_text()
assert (out/'evaluation_receipt.json').is_file()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
full=backup/'T034A_execution.tar'
with tarfile.open(full,'w') as t:
    t.add(run,arcname=run.name);t.add(pre,arcname=pre.name)
compact=dest/'T034A_compact.tar.gz'
with tarfile.open(compact,'w:gz') as t:
    for p in sorted(out.iterdir()):
        if p.is_file():t.add(p,arcname='evidence/'+p.name)
    for p in sorted(out.glob('*/*.json')):t.add(p,arcname='evidence/'+str(p.relative_to(out)))
    for p in sorted(out.glob('*/histories.pt')):t.add(p,arcname='evidence/'+str(p.relative_to(out)))
    for p in ['meta.json','run.sh','train.log']:t.add(run/p,arcname=p)
    t.add(pre/'train.log',arcname='preflight_train.log')
shutil.copy2(compact,backup/compact.name)
receipt=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run=run.name,
    full=dict(path=str(full),bytes=full.stat().st_size,sha256=sha(full)),
    compact=dict(path=str(compact),bytes=compact.stat().st_size,sha256=sha(compact)),
    compact_backup_sha256=sha(backup/compact.name))
assert receipt['compact']['sha256']==receipt['compact_backup_sha256']
(dest/'T034A_archives.json').write_text(json.dumps(receipt,indent=2)+'\n')
shutil.copy2(dest/'T034A_archives.json',backup/'T034A_archives.json')
print(json.dumps(receipt),flush=True)
