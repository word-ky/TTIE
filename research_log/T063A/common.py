import hashlib,json,os
from pathlib import Path
from datetime import datetime,timezone
RAW=Path('/media/wenchang/F/wjq/TTIE/runs/T062CR2-fresh-step27')
ARCHIVE=Path('/media/wenchang/F/wjq/TTIE/shared/t022a/LOL-v2.zip')
MANIFEST=Path('research_log/T062CR1/manifest.json')
COHORT='e8a66a350bcce5282df2e8114355afc434bcd0f4ea69e54957ebb7618f9183c2'
FREEZE='700ae2612234eb139a20c3560b58c85dca1eef3849347c24ffc09575620c07de'
HERE=Path('research_log/T063A')
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def thash(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
def write(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())

PAIRED='a3058b898279f6e4094c642c66d1e5d9013349be83827768d4200bcb374baaad'
