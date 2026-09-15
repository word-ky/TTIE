"""Archive evaluation evidence; original tensors remain in accepted T041/T042 archives."""
from pathlib import Path
import json,hashlib,tarfile,shutil,sys
from datetime import datetime,timezone
home=Path('/home/wenchang/asdasdsad/wjq/TTIE');run=home/'runs'/sys.argv[1];shared=home/'shared/t043a';backup=Path('/media/wenchang/F/wjq/TTIE/shared/t043a')
for p in [shared,backup]:p.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert '[autodl] exit_code=0' in (run/'train.log').read_text()
for row in json.loads((run/'artifacts/quality/output_binding.json').read_bytes())['pairs']:
    for name in ['baseline','candidate']:assert sha(row[name]['path'])==row[name]['sha256']
path=shared/'T043A_evidence.tar.gz'
with tarfile.open(path,'w:gz') as t:
    for p in sorted(run.rglob('*')):
        if p.is_file():
            rel=p.relative_to(run).as_posix();arc=rel[len('artifacts/quality/'):] if rel.startswith('artifacts/quality/') else 'run/'+rel;t.add(p,arcname=arc)
shutil.copyfile(path,backup/path.name);assert sha(path)==sha(backup/path.name)
d=dict(completed_utc=datetime.now(timezone.utc).isoformat(),archive=str(path),backup=str(backup/path.name),bytes=path.stat().st_size,sha256=sha(path),original_200_tensor_files_unchanged=True,original_tensor_archives=['/media/wenchang/F/wjq/TTIE/shared/t041a/T041A_full.tar','/media/wenchang/F/wjq/TTIE/shared/t042a/T042A_full.tar'])
for p in [shared,backup]:(p/'T043A_archives.json').write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(d))
