"""Archive the stopped run without executing any scientific code."""
from pathlib import Path
import json,hashlib,tarfile,shutil
root=Path('/home/wenchang/asdasdsad/wjq/TTIE');run=root/'runs/20260917-023631-ttie-t058a-stage-a';out=root/'shared/t058a';backup=Path('/media/wenchang/F/wjq/TTIE/shared/t058a')
out.mkdir(parents=True,exist_ok=True);backup.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stage=run/'artifacts/stage_a';assert not (stage/'freeze.json').exists()
pre=json.loads((stage/'preflight.json').read_bytes());fd=json.loads((stage/'finite_difference_energy.json').read_bytes())
receipt=dict(status='PARTIAL',run=run.name,exit_code=1,all7346_preflight_pass=len(pre['rows'])==7346,max_reconstruction_abs=pre['max_abs'],preflight_completed_utc=pre['completed_utc'],preflight_sha256=sha(stage/'preflight.json'),selection_sha256=sha(stage/'selection.json'),persisted_bank_files=[p.name for p in stage.glob('*.pt')],finite_difference=fd,stage_a_complete_freeze=False,stage_b_started=False,source_jpg_opens=0,optimizer_updates=0,reruns=0,files={str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file()})
path=out/'T058A_failure.tar.gz'
with tarfile.open(path,'w:gz') as t:
    for p in run.rglob('*'):
        if p.is_file():t.add(p,arcname=str(p.relative_to(run)))
shutil.copy2(path,backup/path.name);assert sha(path)==sha(backup/path.name)
receipt['archive']=dict(path=str(path),backup=str(backup/path.name),bytes=path.stat().st_size,sha256=sha(path))
(out/'T058A_failure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');shutil.copy2(out/'T058A_failure_receipt.json',backup/'T058A_failure_receipt.json')
print(json.dumps(receipt),flush=True)
