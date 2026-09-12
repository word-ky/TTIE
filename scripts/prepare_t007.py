"""Metadata-only deterministic unused image selection; no image scoring."""
import hashlib
import json
import shutil
from pathlib import Path
from PIL import Image

project = Path('/home/wenchang/asdasdsad/wjq/TTIE')
cache = Path('/home/liujianhua/wjq/TAISP/shared/coco200/val2017')
output = project/'shared/t007'
(output/'images').mkdir(parents=True, exist_ok=True)
prior = set()
for task in ('T004','T005','T006'):
    manifest = json.loads((project/f'current/research_log/{task}_manifest.json').read_text())
    prior.update(r['image_id'] for r in manifest['images'])
eligible = []
for path in sorted(cache.glob('*.jpg'), key=lambda p:int(p.stem)):
    if int(path.stem) in prior: continue
    with Image.open(path) as image: w,h = image.size
    if min(w,h)<320: continue
    eligible.append(dict(image_id=int(path.stem), filename=path.name, width=w, height=h,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         url='http://images.cocodataset.org/val2017/'+path.name,
                         split='evaluation_t007'))
assert len(eligible)>=20, 'Insufficient unused cache; stop before scoring'
selected = eligible[:40]
manifest = dict(source='Existing COCO2017 image-only 200-image cache; original selection provenance incomplete',
                selection='Exclude all T004/T005/T006 IDs; original min-side>=320; ascending numeric ID; first40',
                excluded_prior_ids=sorted(prior), eligible_unused_count=len(eligible), images=selected,
                deviation=None if len(selected)==40 else 'Only20-39eligible: use all')
(output/'manifest.json').write_text(json.dumps(manifest,indent=2))
for row in selected: shutil.copy2(cache/row['filename'],output/'images'/row['filename'])
print(json.dumps(dict(eligible=len(eligible),selected=len(selected),excluded=len(prior),manifest=str(output/'manifest.json'))))
