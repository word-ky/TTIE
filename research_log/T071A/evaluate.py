import argparse,json,io,zipfile,hashlib,zlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from research_log.T063A.common import sha,thash,write,utc
from research_log.T071A.run import ARCHIVE,MANIFEST,MANIFEST_SHA,HERE
from research_log.T071A.core import metrics,aggregate,METRIC

def main(out):
    get=lambda n:json.loads((out/n).read_bytes());cfg=get('config.json');frozen=get('output_freeze.json');cohort=get('pairs_manifest.json')
    for path,h in cfg['source_binding'].items():assert sha(path)==h,path
    assert sha(MANIFEST)==MANIFEST_SHA==frozen['final_manifest_sha256']
    assert len(frozen['rows'])==len(cohort['rows'])==100 and frozen['reference_reads']==cohort['reference_reads']==0
    assert sha(out/'pairs_manifest.json')==frozen['pairs_manifest_sha256'] and sha(out/'config.json')==frozen['config_sha256']
    for row in frozen['rows']:
        directory=out/'outputs'/f"{row['index']:03d}"
        assert sha(directory/'output.pt')==row['output_file_sha256'] and sha(directory/'decision.json')==row['decision_file_sha256']
    write(out/'reference_open.json',dict(first_reference_access_utc=utc(),output_freeze_sha256=sha(out/'output_freeze.json')))
    table=[];reads=[]
    with zipfile.ZipFile(ARCHIVE) as z:
        for row,pair in zip(frozen['rows'],cohort['rows']):
            assert row['index']==pair['index'] and row['low']==pair['low'] and row['normal']==pair['normal']
            timestamp=utc();raw=z.read(pair['normal']);reads.append(dict(index=row['index'],member=pair['normal'],opened_utc=timestamp,sha256=hashlib.sha256(raw).hexdigest(),crc32=zlib.crc32(raw)))
            assert len(raw)==pair['normal_bytes'] and reads[-1]['crc32']==pair['normal_crc32']
            with Image.open(io.BytesIO(raw)) as im:reference=(np.asarray(im.convert('RGB'),dtype=np.float64)/255).astype(np.float32).astype(np.float64)
            saved=torch.load(out/'outputs'/f"{row['index']:03d}"/'output.pt',weights_only=True,map_location='cpu')
            assert thash(saved['image'])==row['output_hash'] and thash(saved['state'])==row['selected_state_hash']
            image=saved['image'].squeeze(0).permute(1,2,0).numpy().astype(np.float64)
            table.append(dict(index=row['index'],image_number=pair['image_number'],low=row['low'],normal=pair['normal'],normal_sha256=reads[-1]['sha256'],selected_step=row['selected_step'],seconds=row['seconds'],**metrics(image,reference)))
    write(out/'reference_reads.json',dict(rows=reads,reference_member_reads=len(reads),output_freeze_sha256=sha(out/'output_freeze.json')));write(out/'per_image_metrics.json',table)
    result=dict(classification='OFFICIAL_LOLV2_REAL_TEST_RESULT_FROZEN',**aggregate(table),ssim=METRIC,psnr_convention='per-image -10log10 full RGBMSE, float64 over float32[0,1]pixels, no crop/resize/outputquantization',inference_seconds=frozen['inference_seconds'],mean_image_seconds=frozen['inference_seconds']/100,output_freeze_sha256=sha(out/'output_freeze.json'),final_manifest_sha256=MANIFEST_SHA,inference_reference_reads=0,reference_member_reads=100,optimizer_runs=100,optimizer_updates=2700,model_fits=0,completed_utc=utc())
    write(out/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
