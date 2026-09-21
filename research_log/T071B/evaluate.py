"""Shared T071-A metric arithmetic; no network inference in this process."""
import argparse,io,zipfile,hashlib,zlib
from pathlib import Path
import numpy as np
from PIL import Image
from research_log.T071A.core import metrics,METRIC
from research_log.T071B.common import *

def main(out):
    cohort=read(PAIRS);freezes={m:read(out/m/'freeze.json') for m in METHODS}
    for m,f in freezes.items():
        assert len(f['rows'])==100 and f['reference_reads']==0
        assert f['pairs_manifest_sha256']==sha(PAIRS) and sha(out/m/'config.json')==f['config_sha256']
        cfg=read(out/m/'config.json')
        for p,h in cfg['source_binding'].items():assert sha(p)==h,p
        for r in f['rows']:assert sha(out/m/r['output'])==r['file_sha256']
    freeze_hashes={m:sha(out/m/'freeze.json') for m in METHODS}
    write(out/'reference_open.json',dict(first_reference_access_utc=utc(),freeze_hashes=freeze_hashes))
    ours=read(OURS);assert len(ours)==100
    tables={m:[] for m in METHODS};reads=[]
    with zipfile.ZipFile(ARCHIVE) as z:
        for i,(pair,old) in enumerate(zip(cohort['rows'],ours)):
            assert pair['index']==old['index']==i and pair['low']==old['low'] and pair['normal']==old['normal']
            timestamp=utc();raw=z.read(pair['normal']);h=hashlib.sha256(raw).hexdigest()
            assert h==old['normal_sha256'] and len(raw)==pair['normal_bytes'] and zlib.crc32(raw)==pair['normal_crc32']
            reads.append(dict(index=i,member=pair['normal'],opened_utc=timestamp,sha256=h))
            with Image.open(io.BytesIO(raw)) as im:y=(np.asarray(im.convert('RGB'),dtype=np.float64)/255).astype(np.float32).astype(np.float64)
            for m in METHODS:
                row=freezes[m]['rows'][i];assert row['index']==i and row['low']==pair['low'] and row['normal']==pair['normal']
                x=np.load(out/m/row['output']).astype(np.float64);v=metrics(x,y)
                tables[m].append(dict(index=i,low=pair['low'],normal=pair['normal'],normal_sha256=h,**v,ours_minus_baseline_psnr=old['psnr']-v['psnr']))
    write(out/'reference_reads.json',reads)
    for m in METHODS:write(out/m/'per_image_metrics.json',tables[m])
    results={m:dict(**summary(tables[m]),inference_seconds=freezes[m]['inference_seconds'],mean_image_seconds=freezes[m]['inference_seconds']/100,freeze_sha256=freeze_hashes[m],training=read(HERE/'baseline_bindings.json')[m]['training']) for m in METHODS}
    result=dict(classification='OFFICIAL_LOLV2_REAL_BASELINES_FROZEN',baselines=results,ours=read('research_log/T071A/evidence/result.json'),ours_per_image_sha256=sha(OURS),pairs_manifest_sha256=sha(PAIRS),ssim=METRIC,reference_member_reads=100,inference_reference_reads=0,model_fits=0,optimizer_runs=0,completed_utc=utc())
    write(out/'result.json',result);print(__import__('json').dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
