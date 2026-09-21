import zipfile
import numpy as np
import pytest
from research_log.T071A.core import pairs,metrics,aggregate,PREFIX

def infos():
    rows=[zipfile.ZipInfo(PREFIX+kind+str(i).zfill(5)+'.png') for kind in ['Low/low','Normal/normal'] for i in range(690,790)]
    for r in rows:r.CRC=0
    return rows
def test_complete_official_pairing():
    rows=pairs(infos());assert len(rows)==100 and rows[0]['image_number']==690 and rows[-1]['image_number']==789
    with pytest.raises(AssertionError):pairs(infos()[:-1])
    with pytest.raises(AssertionError):pairs(infos()+[infos()[0]])
def test_metric_convention():
    x=np.full((16,16,3),.2);y=np.full_like(x,.3);m=metrics(x,y)
    assert abs(m['psnr']-20)<1e-12 and abs(m['rgb_ssim']-(2*.2*.3+.0001)/(.2**2+.3**2+.0001))<1e-11

def test_posthoc_aggregation():
    rows=[dict(psnr=x,rgb_ssim=.5,selected_step=k) for x,k in [(10,3),(20,4),(30,4)]];a=aggregate(rows)
    assert a['mean_psnr']==a['median_psnr']==20 and a['mean_rgb_ssim']==.5 and a['selected_steps']==dict(min=3,median=4.,max=4,histogram={'3':1,'4':2})
