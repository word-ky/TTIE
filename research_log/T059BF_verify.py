"""Read-only saved-evidence verification; never calls training or model forward."""
from pathlib import Path
import json,hashlib,gzip,tarfile,shutil,datetime,torch
from ttie.energy_model import load_energy
from research_log.T058A_tangent.core import thash
root=Path('/home/wenchang/asdasdsad/wjq/TTIE')
run=root/'runs/20260917-211326-ttie-t059bf-fixedfit';out=run/'artifacts/T059BF'
dest=root/'shared/t059bf';dest.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda n:json.loads((out/n).read_bytes())
m=read('complete.json');r=read('receipt.json');s=read('summary.json');p=read('preflight.json');history=read('history.json')
for name,key in [('receipt.json','receipt_sha256'),('summary.json','summary_sha256'),('dual_tangent.pt','head_sha256')]:assert sha(out/name)==m[key]
assert len(history)==100 and r['training_runs']==1 and r['optimizer_steps']==2900 and p['passed']
for before,after in [('asset_hashes_before','asset_hashes_after'),('tensor_hashes_before','tensor_hashes_after'),('frozen_head_before','frozen_head_after')]:assert r[before]==r[after]
for name,h in r['asset_hashes_after'].items():assert sha(Path(name))==h,name
head=load_energy(out/'dual_tangent.pt');assert all(torch.isfinite(v).all() for v in head.state_dict().values())
pre={}
for name,c in p['comparisons'].items():
 error=abs(c['actual']-c['expected']);tol=max(2e-6,2e-5*max(abs(c['actual']),abs(c['expected'])))
 pre[name]=dict(**c,absolute_error=error,allowed_tolerance=tol,margin=tol-error)
values=dict(detail_positive=(s['new']['detail']['positive_fraction'],.75),detail_cosine=(s['new']['detail']['median_cosine'],.5),legacy_positive=(s['new']['legacy']['positive_fraction'],.95),legacy_cosine=(s['new']['legacy']['median_cosine'],.9),value_huber=(s['new']['legacy']['value_huber'],1.5*.051005665212869644))
gates={k:dict(actual=v,threshold=t,margin=t-v if k=='value_huber' else v-t,passed=s['gates'][k]) for k,(v,t) in values.items()}
verification=dict(source='524f6436a2282492d69e32cfb3ab67b37b750979',classification=s['classification'],preflight=pre,gates=gates,head_sha256=sha(out/'dual_tangent.pt'),final_head_hashes={k:thash(v) for k,v in head.state_dict().items()},history_sha256=sha(out/'history.json'),epochs=len(history),optimizer_steps=r['optimizer_steps'],training_runs=1,new_feature_forwards=0,counter_basis='Reviewed run uses immutable cached features; no feature extractor forward. Postrun verifier performs no model forward or optimizer calls.',immutable_asset_hashes_reopened=True,checkpoint_reopened=True,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(dest/'T059BF_verification.json').write_text(json.dumps(verification,indent=2))
files={str(f.relative_to(run)):sha(f) for f in run.rglob('*') if f.is_file()}
archive=dest/'T059BF_evidence.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for n in files:t.add(run/n,arcname=n)
backup=Path('/media/wenchang/F/wjq/TTIE/shared/t059bf');backup.mkdir(parents=True,exist_ok=True);shutil.copy2(archive,backup/archive.name);assert sha(archive)==sha(backup/archive.name)
info=dict(files=files,home=str(archive),backup=str(backup/archive.name),sha256=sha(archive),bytes=archive.stat().st_size,home_verified=True,F_backup_verified=True)
(dest/'T059BF_archives.json').write_text(json.dumps(info,indent=2))
for name in ['summary.json','preflight.json','history.json','complete.json']:shutil.copy2(out/name,dest/('T059BF_'+name))
(dest/'T059BF_receipt.json.gz').write_bytes(gzip.compress((out/'receipt.json').read_bytes()))
for f in dest.glob('*.json'):shutil.copy2(f,backup/f.name)
shutil.copy2(dest/'T059BF_receipt.json.gz',backup/'T059BF_receipt.json.gz')
print(json.dumps(dict(summary=s,verification=verification,runtime=r['seconds'],archive=info)))
