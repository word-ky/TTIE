from pathlib import Path
import tarfile,hashlib,json,shutil
base=Path('/home/wenchang/asdasdsad/wjq/TTIE');a=base/'runs/20260915-152458-ttie-t044a-stage-a';b=base/'runs/20260915-152656-ttie-t044a-stage-b';out=base/'shared/t044a';out.mkdir(exist_ok=True)
f=json.loads((a/'artifacts/stage_a/freeze.json').read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(a/'artifacts/stage_a/scores.json')==f['scores_sha256'] and sha(a/'artifacts/stage_a/states.pt')==f['states_sha256']
archive=out/'T044A_evidence.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for run,stage in [(a,'stage_a'),(b,'stage_b')]:
        t.add(run/'artifacts'/stage,arcname=stage)
        for name in ['run.sh','train.log','meta.json']:t.add(run/name,arcname=stage+'_run/'+name)
backup=Path('/media/wenchang/F/wjq/TTIE/shared/t044a');backup.mkdir(exist_ok=True);shutil.copy2(archive,backup/archive.name)
assert sha(archive)==sha(backup/archive.name)
receipt=dict(path=str(archive),backup=str(backup/archive.name),bytes=archive.stat().st_size,sha256=sha(archive),scores_and_states_unchanged=True)
(out/'T044A_archives.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
