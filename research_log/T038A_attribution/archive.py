"""Preserve Stage-A tensors and Stage-B attribution with exact checksums."""
from pathlib import Path
import json,hashlib,tarfile,shutil
from datetime import datetime,timezone
home=Path('/home/wenchang/asdasdsad/wjq/TTIE');backup=Path('/media/wenchang/F/wjq/TTIE/shared/t038a');shared=home/'shared/t038a'
for p in [backup,shared]:p.mkdir(parents=True,exist_ok=True)
runs=dict(stage_a=home/'runs/20260915-064200-ttie-t038a-stage-a',stage_b=home/'runs/20260915-064355-ttie-t038a-stage-b')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
for run in runs.values():assert '[autodl] exit_code=0' in (run/'train.log').read_text()
stage_a=runs['stage_a']/'artifacts/stage_a';f=json.loads((stage_a/'freeze.json').read_bytes())
for row in f['rows']:assert sha(stage_a/row['file'])==row['sha256']
archives={}
for label,path,full in [('full',backup/'T038A_full.tar',True),('compact',shared/'T038A_compact.tar.gz',False)]:
    with tarfile.open(path,'w' if full else 'w:gz') as tar:
        for kind,run in runs.items():
            folder=run/'artifacts'/kind
            for p in sorted(folder.iterdir()):
                if p.is_file() and (full or p.suffix=='.json'):tar.add(p,arcname=kind+'/'+p.name)
            for p in sorted(run.iterdir()):
                if p.is_file():tar.add(p,arcname=kind+'_run/'+p.name)
    archives[label]=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path));print(label,json.dumps(archives[label]),flush=True)
copy=backup/'T038A_compact.tar.gz';shutil.copyfile(shared/'T038A_compact.tar.gz',copy);assert sha(copy)==archives['compact']['sha256']
receipt=dict(completed_utc=datetime.now(timezone.utc).isoformat(),stage_a_100_file_hashes_unchanged=True,archives=archives,compact_backup=str(copy))
for folder in [shared,backup]:(folder/'T038A_archives.json').write_text(json.dumps(receipt,indent=2)+'\n')
