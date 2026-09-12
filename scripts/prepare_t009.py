"""Metadata-only development split from the complete official T008 download."""
import hashlib
import json
from pathlib import Path
from PIL import Image

root=Path('/home/wenchang/asdasdsad/wjq/TTIE')
cache=root/'shared/t008/val2017'
output=root/'shared/t009';output.mkdir(parents=True,exist_ok=True)
prior=set()
for task in ('T004','T005','T006','T007','T008'):
    prior.update(r['image_id'] for r in json.loads((root/f'current/research_log/{task}_manifest.json').read_text())['images'])
images=[];inspected=[]
for path in sorted(cache.glob('*.jpg'),key=lambda p:int(p.stem)):
    if int(path.stem) in prior:continue
    with Image.open(path) as image:w,h=image.size
    inspected.append(dict(image_id=int(path.stem),width=w,height=h,eligible=min(w,h)>=320))
    if min(w,h)<320:continue
    images.append(dict(image_id=int(path.stem),filename=path.name,width=w,height=h,split='development_t009',
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_url='http://images.cocodataset.org/val2017/'+path.name))
    if len(images)==40:break
assert len(images)==40
manifest=dict(source='Complete official COCO val2017 image-only archive downloaded for T008',
  archive_sha256='4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05',image_count=5000,
  selection='Numeric ascending; exclude all T004-T008 IDs; original min-side>=320; first40; development only',
  excluded_prior_ids=sorted(prior),inspected_prefix=inspected,images=images)
(output/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('Development IDs:',[r['image_id'] for r in images],flush=True)
