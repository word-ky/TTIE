"""Metadata-only fresh split, called only after source calibration succeeds."""
import hashlib
from pathlib import Path
from PIL import Image


def evaluation_manifest(pool,excluded,*,count=40,split='evaluation_t010'):
    images=[];inspected=[]
    for path in sorted(Path(pool).glob('*.jpg'),key=lambda p:int(p.stem)):
        if int(path.stem) in excluded:continue
        with Image.open(path) as image:w,h=image.size
        inspected.append(dict(image_id=int(path.stem),width=w,height=h,eligible=min(w,h)>=320))
        if min(w,h)<320:continue
        images.append(dict(image_id=int(path.stem),filename=path.name,width=w,height=h,split=split,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_url='http://images.cocodataset.org/val2017/'+path.name))
        if len(images)==count:break
    assert len(images)==count
    return dict(source='Complete official COCO val2017 image-only archive downloaded for T008',
        archive_sha256='4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05',
        selection='numeric ascending; exclude T004-T009; original min-side>=320; first40',
        excluded_prior_ids=sorted(excluded),inspected_prefix=inspected,images=images)
