"""Read only selected training members; external freeze binding before normals."""
import argparse,json,zipfile
from pathlib import Path
from scripts.run_t036a import sha,write,utc
p=argparse.ArgumentParser()
for k in ['archive','manifest','out']:p.add_argument('--'+k,type=Path,required=True)
p.add_argument('--kind',choices=['low','normal'],required=True);p.add_argument('--audit',type=Path);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True);m=json.loads(a.manifest.read_bytes());assert len(m['selected'])==100
receipt=dict(started_utc=utc(),manifest_sha256=sha(a.manifest),kind=a.kind,files=[])
if a.kind=='normal':
    f=json.loads((a.audit/'freeze.json').read_bytes());assert f['normal_decodes']==0 and f['manifest_sha256']==sha(a.manifest) and len(f['rows'])==100
    check=json.loads((a.audit/'independent_replay.json').read_bytes());assert check['status']=='PASS' and check['methods']==200
    for r in f['rows']:
        for method,record in r['methods'].items():
            for name,h in record['files'].items():assert sha(a.audit/f'{r["index"]:03d}'/method/name)==h['sha256']
    receipt['freeze_sha256']=sha(a.audit/'freeze.json');write(a.out/'reference_deployment.json',receipt)
with zipfile.ZipFile(a.archive) as z:
    for r in m['selected']:
        dest=a.out/a.kind/r[a.kind];dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(z.read('LOL-v2/Real_captured/'+r[a.kind]));assert sha(dest)==r[a.kind+'_sha256']
        receipt['files'].append(dict(path=str(dest),sha256=r[a.kind+'_sha256']))
receipt.update(completed_utc=utc(),normal_members_read=100 if a.kind=='normal' else 0,pixel_decodes=0)
write(a.out/('reference_deployment.json' if a.kind=='normal' else 'low_deployment.json'),receipt)
print(a.kind,'deployed100 selected training members',flush=True)
