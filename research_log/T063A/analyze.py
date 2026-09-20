import argparse,json,zipfile,io,time,math,statistics
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from PIL import Image
from ttie.ssim_transfer import rgb_ssim
from scripts.evaluate_t026a import independent_ssim
from research_log.T063A.common import *

def oracle(psnr,baseline):
 reachable=[k for k,p in enumerate(psnr) if p-baseline>=-5.614]
 chosen=max(reachable or list(range(28)),key=lambda k:(psnr[k],-k))
 return dict(reachable_steps=reachable,oracle_step=chosen,safety_unreachable=not reachable)

def metric(image,normal,independent):
 x=image[0].permute(1,2,0).numpy().astype(np.float64);d=x-normal
 if independent:return dict(psnr=-10*math.log10(float(d.reshape(-1)@d.reshape(-1))/d.size),ssim=independent_ssim(x,normal))
 return dict(psnr=float(-10*np.log10(np.mean(d*d))),ssim=rgb_ssim(x,normal))

def score(job):
 out,rec,item,reference,prior,independent=job;torch.set_num_threads(1);d=Path(out)/f"{rec['index']:03d}"
 assert sha(d/'images.pt')==rec['image_file_sha256'];images=torch.load(d/'images.pt',weights_only=True,map_location='cpu');assert len(images)==28
 for k,image in enumerate(images):assert thash(image)==rec['rendered_hashes'][k]
 old=RAW/f"{rec['index']:03d}";assert sha(old/'outputs.pt')==rec['input_outputs_sha256'];saved=torch.load(old/'outputs.pt',weights_only=True,map_location='cpu');assert torch.equal(images[27],saved['T062'])
 stamp=utc()
 with zipfile.ZipFile(ARCHIVE) as z:raw=z.read('LOL-v2/Real_captured/'+item['normal'])
 assert hashlib.sha256(raw).hexdigest()==reference['normal_sha256']
 with Image.open(io.BytesIO(raw)) as im:normal=(np.asarray(im.convert('RGB'),dtype=np.float64)/255).astype(np.float32).astype(np.float64)
 metrics=[metric(image,normal,independent) for image in images];controls={n:metric(saved[n],normal,independent) for n in ['T026','T036','identity']}
 for n,values in controls.items():
  for key,value in values.items():assert abs(value-prior['metrics'][n][key])<1e-10
 for key in ['psnr','ssim']:assert abs(metrics[27][key]-prior['metrics']['T062'][key])<1e-10
 return dict(index=rec['index'],low=item['low'],normal=item['normal'],reference_read_utc=stamp,normal_sha256=reference['normal_sha256'],states=metrics,controls=controls)

def main(out):
 start=time.perf_counter();freeze=json.loads((out/'freeze.json').read_bytes());assert freeze['outputs']==2800 and freeze['step27_exact']==100 and sha(MANIFEST)==COHORT
 assert sha(out/'config.json')==freeze['config_sha256'] and sha(RAW/'freeze.json')==FREEZE
 first=utc();assert first>freeze['completed_utc'];write(out/'reference_open.json',dict(first_reference_or_quality_read_utc=first,freeze_sha256=sha(out/'freeze.json')))
 rows=json.loads(MANIFEST.read_bytes())['selected'];refs=json.loads(Path('research_log/T062CR2/reference_inputs.json').read_bytes());assert sha(RAW/'paired.json')==PAIRED;prior=json.loads((RAW/'paired.json').read_bytes())
 jobs=[(str(out),f,r,n,b,False) for f,r,n,b in zip(freeze['rows'],rows,refs,prior)]
 with ProcessPoolExecutor(max_workers=8) as pool:scored=list(pool.map(score,jobs))
 for r in scored:r.update(oracle([s['psnr'] for s in r['states']],r['controls']['T026']['psnr']))
 selected=[r['states'][r['oracle_step']] for r in scored];d=np.array([s['psnr']-r['controls']['T036']['psnr'] for s,r in zip(selected,scored)]);b=np.array([s['psnr']-r['controls']['T026']['psnr'] for s,r in zip(selected,scored)]);ds=np.array([s['ssim']-r['controls']['T036']['ssim'] for s,r in zip(selected,scored)])
 gates=dict(mean_psnr=float(d.mean())>=2,median_psnr=float(np.median(d))>0,regressions=int((b<0).sum())<=29,worst=float(b.min())>=-5.614,mean_ssim=float(ds.mean())>=-.001)
 failures=[r for r in scored if r['states'][27]['psnr']-r['controls']['T026']['psnr']< -5.614]
 result=dict(label='REFERENCE_ORACLE_ONLY',classification='SELECTION_HEADROOM_PRESENT' if not any(r['safety_unreachable'] for r in scored) and all(gates.values()) else 'TRAJECTORY_LIMITED',reachable_images=sum(not r['safety_unreachable'] for r in scored),unreachable_indices=[r['index'] for r in scored if r['safety_unreachable']],gates=gates,mean_delta_psnr=float(d.mean()),median_delta_psnr=float(np.median(d)),regressions_t026=int((b<0).sum()),worst_delta_t026=float(b.min()),mean_delta_ssim=float(ds.mean()),mean_psnr=float(np.mean([s['psnr'] for s in selected])),mean_ssim=float(np.mean([s['ssim'] for s in selected])),step27_failure_indices=[r['index'] for r in failures],freeze_sha256=sha(out/'freeze.json'),freeze_utc=freeze['completed_utc'],first_reference_read_utc=first,seconds=time.perf_counter()-start,completed_utc=utc(),optimizer_runs=0)
 write(out/'per_image.json',scored);write(out/'step27_failures.json',failures);write(out/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
