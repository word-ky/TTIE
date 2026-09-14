"""Deploy exact named training members; normals only after full output freeze."""
import argparse,json,hashlib,zipfile
from pathlib import Path
from datetime import datetime,timezone

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()

def main():
    p=argparse.ArgumentParser()
    for k in ['archive','manifest','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--kind',choices=['low','normal'],required=True);p.add_argument('--audit',type=Path);a=p.parse_args()
    started=utc();manifest=json.loads(a.manifest.read_bytes());rows=manifest['selected'];assert len(rows)==100
    receipt=dict(started_utc=started,manifest_sha256=sha(a.manifest),kind=a.kind,files=[])
    if a.kind=='normal':
        freeze=json.loads((a.audit/'freeze.json').read_bytes())
        assert freeze['completed_utc']<started and freeze['normal_decodes']==0 and len(freeze['rows'])==100
        assert freeze['manifest_sha256']==sha(a.manifest)
        replay=json.loads((a.audit/'independent_replay.json').read_bytes());assert replay['status']=='PASS' and replay['count']==100
        for r in freeze['rows']:
            for n,h in r['files'].items():assert sha(a.audit/f'{r["index"]:03d}'/n)==h['sha256']
        receipt['freeze_sha256']=sha(a.audit/'freeze.json')
        # Persist the independently bound expected freeze before any normal member is read.
        (a.out/'reference_deployment.json').write_text(json.dumps(receipt,indent=2)+'\n')
    with zipfile.ZipFile(a.archive) as z:
        for r in rows:
            data=z.read('LOL-v2/Real_captured/'+r[a.kind]);assert hashlib.sha256(data).hexdigest()==r[a.kind+'_sha256']
            dest=a.out/a.kind/r[a.kind];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
            receipt['files'].append(dict(path=str(dest),sha256=r[a.kind+'_sha256']))
    receipt.update(completed_utc=utc(),normal_members_read=100 if a.kind=='normal' else 0,pixel_decodes=0)
    name='reference_deployment.json' if a.kind=='normal' else 'low_deployment.json'
    (a.out/name).write_text(json.dumps(receipt,indent=2)+'\n');print(a.kind,'deployed100',flush=True)

if __name__=='__main__':main()
