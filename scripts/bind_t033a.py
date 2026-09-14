"""Persist an external evaluation authorization bound to the completed output freeze."""
import argparse,json,hashlib
from pathlib import Path
from datetime import datetime,timezone

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify(audit, receipt):
    assert sha(audit/'freeze.json')==receipt['freeze_sha256'],'output freeze mismatch'
    f=json.loads((audit/'freeze.json').read_bytes());assert f['completed_utc']<receipt['created_utc']
    return f

def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    f=json.loads((a.audit/'freeze.json').read_bytes());assert len(f['rows'])==100 and f['normal_decodes']==0
    for r in f['rows']:assert sha(a.audit/r['output'])==r['file_sha256']
    d=dict(created_utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=sha(a.audit/'freeze.json'))
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(d,indent=2)+'\n');print(d)

if __name__=='__main__':main()
