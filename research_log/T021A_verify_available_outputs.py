"""Read-only byte verification of accepted T014 outputs; no tensor loading."""
from pathlib import Path
import hashlib,json,datetime
root=Path('/media/wenchang/F/wjq/TTIE/runs/20260912-191947-ttie-t014-stage-b/artifacts/audit')
out=Path('/home/wenchang/asdasdsad/wjq/TTIE/research_log/T021A_preflight');out.mkdir(parents=True,exist_ok=True)
expected={'artifact_manifest.json': {'commit': 'cebecffbd1335fade336df17d653eb4e5fb65ba3', 'path': 'research_log/remote_runs/20260912-191947-ttie-t014-stage-b/artifacts/audit/artifact_manifest.json', 'sha256': 'dfb6d1a6fe18357e52cfc67921e531cc05a0421e6c685d55a6a37571ace605cb'}, 'config.json': {'commit': 'cebecffbd1335fade336df17d653eb4e5fb65ba3', 'path': 'research_log/remote_runs/20260912-191947-ttie-t014-stage-b/artifacts/audit/config.json', 'sha256': 'c73887eb95df09a87af681a4aa072f5c160786affe81919206be4684ddc19b11'}, 'output_verification.json': {'commit': 'cebecffbd1335fade336df17d653eb4e5fb65ba3', 'path': 'research_log/remote_runs/20260912-191947-ttie-t014-stage-b/artifacts/audit/output_verification.json', 'sha256': '25fac5c54faba826080ce4f425f19c963d192c379552e2ef109bd6d1146da394'}, 'final_checks.json': {'commit': 'cebecffbd1335fade336df17d653eb4e5fb65ba3', 'path': 'research_log/remote_runs/20260912-191947-ttie-t014-stage-b/artifacts/audit/final_checks.json', 'sha256': '5b23ae076d34703beec6710e084086e6a6a9e789ff5ee6e7f024327588e1cd65'}, 'sobolev_pilot.py': {'commit': 'f861b2c6ffde6d017cb174ef8e00cb75701bf5e1', 'path': 'ttie/sobolev_pilot.py', 'sha256': '7dfdba0e0cabff5f46aff6864994e578f0e110d7dcce019f9b7c58d9ace995d3'}, 'energy_io.py': {'commit': 'f861b2c6ffde6d017cb174ef8e00cb75701bf5e1', 'path': 'ttie/energy_io.py', 'sha256': '3f59c23392ae1dec390ef26302e7fd1adbe9afffd45f0c6f168edd9a5d16ffa9'}, 'natural.py': {'commit': 'f861b2c6ffde6d017cb174ef8e00cb75701bf5e1', 'path': 'ttie/natural.py', 'sha256': 'bfc8cbf8d5c2f92e7e4dfdb49f5c0fc217cb25167be2a8b7a16a0322996fb94e'}, 'residual_metrics.py': {'commit': 'f861b2c6ffde6d017cb174ef8e00cb75701bf5e1', 'path': 'ttie/residual_metrics.py', 'sha256': 'de0fc3b36e431b097253ff1084073c717967a623009c8681748bb789e69df523'}}
for name in ('artifact_manifest.json','config.json','output_verification.json','final_checks.json'):
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected[name]['sha256']
manifest=json.loads((root/'artifact_manifest.json').read_bytes())
clean={e['image_id']:e['directory'] for e in manifest if e['condition']=='clean'}
records=[];total=0
for i,e in enumerate(manifest):
 files={}
 for name,receipt in e['files']['episode'].items():
  p=root/e['directory']/name;digest=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
  assert digest.hexdigest()==receipt['sha256'] and p.stat().st_size==receipt['bytes']
  files[name]=dict(path=str(p),**receipt);total+=receipt['bytes']
 records.append(dict(row_index=i,image_id=e['image_id'],condition=e['condition'],directory=e['directory'],files=files,clean_identity_directory=clean[e['image_id']]))
result=dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted_origins=expected,verified_rows=len(records),source_images=len(clean),verified_files=2*len(records),verified_bytes=total,tensors_loaded=False,clean_reference_read=False,all_output_hashes_exact=True,baseline_pending=True,pool_pending=True,rows=records)
(out/'outputs_verified.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('rows','accepted_origins')}))
