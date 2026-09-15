"""Preserve complete T042 source diagnostic runs and compact reviewer evidence."""
from pathlib import Path
import json,hashlib,tarfile,shutil,sys
from datetime import datetime,timezone
home=Path('/home/wenchang/asdasdsad/wjq/TTIE');backup=Path('/media/wenchang/F/wjq/TTIE/shared/t042a');shared=home/'shared/t042a'
for p in [backup,shared]:p.mkdir(parents=True,exist_ok=True)
runs={k:home/'runs'/v for k,v in zip(['stage_a','stage_b'],sys.argv[1:3])}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
for run in runs.values():assert '[autodl] exit_code=0' in (run/'train.log').read_text()
source=runs['stage_a']/'artifacts/stage_a';freeze=json.loads((source/'freeze.json').read_bytes())
for row in freeze['rows']:assert sha(source/row['file'])==row['sha256']
archives={}
for name,path,full in [('full',backup/'T042A_full.tar',True),('compact',shared/'T042A_compact.tar.gz',False)]:
    with tarfile.open(path,'w' if full else 'w:gz') as tar:
        for kind,run in runs.items():
            for p in sorted(run.rglob('*')):
                if not p.is_file() or (not full and p.suffix=='.pt'):continue
                relative=p.relative_to(run).as_posix()
                arc=kind+'/'+relative.split('/',2)[2] if relative.startswith('artifacts/'+kind+'/') else kind+'_run/'+relative
                tar.add(p,arcname=arc)
    archives[name]=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path));print(name,json.dumps(archives[name]),flush=True)
copy=backup/'T042A_compact.tar.gz';shutil.copyfile(shared/'T042A_compact.tar.gz',copy);assert sha(copy)==archives['compact']['sha256']
receipt=dict(completed_utc=datetime.now(timezone.utc).isoformat(),stage_a_100_file_hashes_unchanged=True,archives=archives,compact_backup=str(copy))
for folder in [shared,backup]:(folder/'T042A_archives.json').write_text(json.dumps(receipt,indent=2)+'\n')
