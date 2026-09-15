"""Separate full metric replay, no main evaluator or aggregation imports."""
import argparse,json,csv,hashlib,statistics,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import convolve1d
p=argparse.ArgumentParser()
for k in ['audit','split','normal-root','deployment','ours']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();torch.set_num_threads(1);start=time.perf_counter()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
f=json.loads((a.audit/'freeze.json').read_bytes());d=json.loads(a.deployment.read_bytes());assert sha(a.audit/'freeze.json')==d['freeze_sha256'] and f['completed_utc']<d['created_utc']
assert sha(a.split)==f['split_sha256']=='b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b'
assert sha(a.ours)=='ad704fca9dbc393a0e30737eae60d212d41bb797e630e6599d39059380e5a397'
split=json.loads(a.split.read_bytes())['selected'];ours=list(csv.DictReader(a.ours.open()));metrics=list(csv.DictReader((a.audit/'metrics.csv').open()));assert len(split)==len(ours)==len(metrics)==len(f['rows'])==100
w=np.exp(-np.arange(-5,6,dtype=np.float64)**2/4.5);w/=w.sum()
def smooth(x):return convolve1d(convolve1d(x,w,axis=0,mode='reflect'),w,axis=1,mode='reflect')
values={k:[] for k in ['snr_psnr','snr_ssim','ours_psnr','ours_ssim','delta_psnr','delta_ssim']};errors=[]
for row,item,old,main in zip(f['rows'],split,ours,metrics):
    assert row['low']==item['low']==old['low']==main['low'] and item['normal']==old['normal']
    assert sha(a.audit/row['output'])==row['file_sha256'] and sha(a.normal_root/item['normal'])==item['normal_sha256']
    raw=np.load(a.audit/row['output']);assert hashlib.sha256(raw.tobytes()).hexdigest()==row['output_sha256']
    x=raw.astype(np.float64)
    with Image.open(a.normal_root/item['normal']) as im:y=(np.asarray(im.convert('RGB'),dtype=np.float64)/255).astype(np.float32).astype(np.float64)
    mse=torch.from_numpy(x-y).square().mean();psnr=float(-10*torch.log10(mse))
    mx,my=smooth(x),smooth(y);ssim=float(np.mean(((2*mx*my+.0001)*(2*(smooth(x*y)-mx*my)+.0009))/((mx*mx+my*my+.0001)*(smooth(x*x)-mx*mx+smooth(y*y)-my*my+.0009))))
    v=dict(snr_psnr=psnr,snr_ssim=ssim,ours_psnr=float(old['ours_psnr']),ours_ssim=float(old['ours_ssim']),delta_psnr=psnr-float(old['ours_psnr']),delta_ssim=ssim-float(old['ours_ssim']))
    for key,val in v.items():values[key].append(val);errors.append(abs(val-float(main[key])))
summary=json.loads((a.audit/'summary.json').read_bytes())
for key,v in values.items():
    for name,val in [('mean',statistics.mean(v)),('median',statistics.median(v))]:errors.append(abs(val-summary['metrics'][key][name]))
assert max(errors)<=1e-10
out=dict(status='PASS',images=100,independent_output_metrics=200,scalar_checks=len(errors),max_abs_error=max(errors),freeze_before_authorization=True,all_output_hashes_unchanged=True,seconds=time.perf_counter()-start)
(a.audit/'independent_replay.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
