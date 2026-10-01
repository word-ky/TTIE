import re
from collections import Counter
from pathlib import PurePosixPath
import numpy as np
from ttie.ssim_transfer import rgb_ssim,METRIC
PREFIX='LOL-v2/Real_captured/Test/'
def pairs(infos):
    lows={};refs={}
    for i in infos:
        if i.is_dir() or not i.filename.startswith(PREFIX):continue
        m=re.fullmatch(re.escape(PREFIX)+r'(Low/low|Normal/normal)([0-9]{5})\.png',i.filename)
        assert m is not None,i.filename
        target=lows if m[1]=='Low/low' else refs;key=int(m[2]);assert key not in target;target[key]=i
    assert set(lows)==set(refs)==set(range(690,790))
    return [dict(index=j,image_number=k,low=lows[k].filename,normal=refs[k].filename,low_crc32=lows[k].CRC,normal_crc32=refs[k].CRC,low_bytes=lows[k].file_size,normal_bytes=refs[k].file_size) for j,k in enumerate(sorted(lows))]
def metrics(output,reference):
    x=np.asarray(output,dtype=np.float64);y=np.asarray(reference,dtype=np.float64)
    assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
    return dict(psnr=float(-10*np.log10(np.mean((x-y)**2))),rgb_ssim=rgb_ssim(x,y))
def aggregate(rows):
    ps=[r['psnr'] for r in rows];ss=[r['rgb_ssim'] for r in rows];steps=[r['selected_step'] for r in rows]
    return dict(images=len(rows),mean_psnr=float(np.mean(ps)),median_psnr=float(np.median(ps)),mean_rgb_ssim=float(np.mean(ss)),selected_steps=dict(min=min(steps),median=float(np.median(steps)),max=max(steps),histogram={str(k):v for k,v in sorted(Counter(steps).items())}))
