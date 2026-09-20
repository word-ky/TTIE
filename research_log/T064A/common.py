import hashlib,json,os
from pathlib import Path
from datetime import datetime,timezone
RAW=Path('/media/wenchang/F/wjq/TTIE/runs/T063D-fresh-progress')
ARCHIVE=Path('/media/wenchang/F/wjq/TTIE/shared/t022a/LOL-v2.zip')
MANIFEST=Path('research_log/T063D/manifest.json')
COHORT='3206ea57061f4b45164a81f105f818de6d6d15b342a77797ce9f1eaaccc52554'
FREEZE='7cdb658d380720f51c2519817953a9e0b6dc0d6f4f2b4bca1deb0bb5cdb3cdd2'
HERE=Path('research_log/T064A')
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def thash(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
def write(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())

PAIRED='6efbc78f14bd844d6427f1f633858889fedae730b6f07255f3d56afc71b5d6a4'

LOWROOT=Path('/media/wenchang/F/wjq/TTIE/shared/t063d/low')
