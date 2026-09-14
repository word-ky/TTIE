"""Preserve full reconstructed frames on F and compact scalar evidence on both roots."""
import hashlib,json,shutil,tarfile
from pathlib import Path
from datetime import datetime,timezone

home=Path('/home/wenchang/asdasdsad/wjq/TTIE')
backup=Path('/media/wenchang/F/wjq/TTIE/shared/t037a');backup.mkdir(parents=True,exist_ok=True)
shared=home/'shared/t037a';shared.mkdir(parents=True,exist_ok=True)
run=home/'runs/20260915-055148-ttie-t037a-reconstruct'
evaluation=home/'runs/20260915-055515-ttie-t037a-evaluate'
out=run/'artifacts/REFERENCE_DIAGNOSTIC_ONLY'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
assert '[autodl] exit_code=0' in (evaluation/'train.log').read_text()
freeze=json.loads((out/'freeze.json').read_bytes())
for row in freeze['rows']:
    for r in row['methods'].values():assert sha(out/r['file'])==r['sha256']
print('200 complete trajectory hashes unchanged after reference scoring',flush=True)
archives={}
for name,dest,full in [('full',backup/'T037A_full.tar',True),('compact',shared/'T037A_compact.tar.gz',False)]:
    with tarfile.open(dest,'w' if full else 'w:gz') as tar:
        for p in sorted(out.rglob('*')):
            if p.is_file() and (full or p.suffix in ['.json','.csv']):tar.add(p,arcname='REFERENCE_DIAGNOSTIC_ONLY/'+p.relative_to(out).as_posix())
        for directory,label in [(run,'reconstruction'),(evaluation,'evaluation')]:
            for p in sorted(directory.iterdir()):
                if p.is_file():tar.add(p,arcname=label+'/'+p.name)
    archives[name]=dict(path=str(dest),bytes=dest.stat().st_size,sha256=sha(dest))
    print(name,json.dumps(archives[name]),flush=True)
copy=backup/'T037A_compact.tar.gz';shutil.copyfile(shared/'T037A_compact.tar.gz',copy)
assert sha(copy)==archives['compact']['sha256']
receipt=dict(completed_utc=datetime.now(timezone.utc).isoformat(),post_evaluation_trajectory_hashes=200,archives=archives,compact_backup=str(copy))
for root in [shared,backup]:(root/'T037A_archives.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
