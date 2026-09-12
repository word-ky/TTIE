"""Obtain official image-only COCO validation archive, then select by ID."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile
from PIL import Image

project=Path('/home/wenchang/asdasdsad/wjq/TTIE')
shared=project/'shared/t008'
shared.mkdir(parents=True,exist_ok=True)
archive=shared/'val2017.zip'
url='http://images.cocodataset.org/zips/val2017.zip'
with urllib.request.urlopen(url,timeout=60) as response,archive.open('wb') as output:
    total=0
    while chunk:=response.read(8*1024*1024):
        output.write(chunk);total+=len(chunk)
        if total%(64*1024*1024)==0:print('Downloaded bytes',total,flush=True)
with zipfile.ZipFile(archive) as bundle:
    names=[name for name in bundle.namelist() if name.endswith('.jpg')]
    assert len(names)==5000
    bundle.extractall(shared)
prior=set()
for task in ('T004','T005','T006','T007'):
    prior.update(r['image_id'] for r in json.loads((project/f'current/research_log/{task}_manifest.json').read_text())['images'])
selected=[]
inspected=[]
for path in sorted((shared/'val2017').glob('*.jpg'),key=lambda p:int(p.stem)):
    if int(path.stem) in prior:continue
    with Image.open(path) as image:w,h=image.size
    inspected.append(dict(image_id=int(path.stem),width=w,height=h,eligible=min(w,h)>=320))
    if min(w,h)<320:continue
    selected.append(dict(image_id=int(path.stem),filename=path.name,width=w,height=h,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest(),split='evaluation_t008',
                         source_url='http://images.cocodataset.org/val2017/'+path.name))
    if len(selected)==40:break
assert len(selected)==40
manifest=dict(source='Official complete COCO val2017 image-only archive',source_url=url,
              archive_bytes=archive.stat().st_size,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
              image_count=5000,selection='Numeric ascending IDs; exclude all T004-T007 IDs; original min-side>=320; first40',
              excluded_prior_ids=sorted(prior),inspected_prefix=inspected,images=selected)
(shared/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('Selected40 images:',[r['image_id'] for r in selected],flush=True)
