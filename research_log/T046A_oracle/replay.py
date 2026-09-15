"""Independent raw-history winner and complete metric/statistic replay."""
import argparse,json,csv,math,hashlib,statistics,collections,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import convolve1d
p=argparse.ArgumentParser()
for k in ['split','normal-root','accepted','preflight','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();torch.set_num_threads(1);start=time.perf_counter()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def thash(x):return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
f=json.loads((a.out/'freeze.json').read_bytes());pre=json.loads((a.preflight/'preflight.json').read_bytes());receipt=json.loads((a.out/'evaluation_receipt.json').read_bytes());summary=json.loads((a.out/'summary.json').read_bytes());pairs=json.loads((a.out/'pairs.json').read_bytes())
assert sha(a.out/'freeze.json')==receipt['freeze_sha256'] and sha(a.preflight/'preflight.json')==f['preflight_sha256']
assert all(pre['completed_utc']<v['utc'] for v in f['opened']) and all(f['completed_utc']<v['utc'] for v in receipt['normal_opens'])
assert sha(a.accepted/'freeze.json')==pre['prior_freeze_sha256'] and sha(a.accepted/'per_image.csv')==receipt['prior_csv_sha256']=='d531c45ce670f5b84c2b74059e931350e21e49c7d1c70d15229f4fbb26a24d72'
assert sha(a.split)==pre['split_sha256'];split=json.loads(a.split.read_bytes())['selected'];baseline=list(csv.DictReader((a.accepted/'per_image.csv').open()));starts=torch.load(a.preflight/'starts.pt',weights_only=True,map_location='cpu');assert sha(a.preflight/'starts.pt')==pre['starts_sha256']
weights=np.exp(-np.arange(-5,6,dtype=np.float64)**2/4.5);weights/=weights.sum()
def smooth(v):return convolve1d(convolve1d(v,weights,axis=0,mode='reflect'),weights,axis=1,mode='reflect')
errors=[];values={k:[] for k in ['psnr','ssim','t035_psnr','t035_ssim','delta_psnr','delta_ssim']};steps=[]
for i,(row,item,old,main) in enumerate(zip(f['rows'],split,baseline,pairs)):
    assert row['low']==item['low']==old['low']==main['low'];root=a.out/f'{i:03d}'
    for name,h in row['files'].items():assert sha(root/name)==h
    h=torch.load(root/'history.pt',weights_only=True,map_location='cpu');v=torch.load(root/'output.pt',weights_only=True,map_location='cpu')
    assert h['raw'].shape==(1001,1,3,2,2) and h['mse'].shape==(1001,) and torch.isfinite(h['raw']).all() and torch.isfinite(h['mse']).all()
    step=min(range(1001),key=lambda j:float(h['mse'][j]));steps.append(step)
    assert step==row['best_step']==main['best_step'] and torch.equal(v['raw'],h['raw'][step]) and torch.equal(starts[i],h['raw'][0])
    assert thash(v['raw'])==row['raw_sha256'] and thash(v['image'])==row['output_sha256'] and thash(starts[i])==pre['rows'][i]['raw_sha256']
    raw=h['raw'][:,0].numpy().astype(np.float64);ev=2*np.tanh(raw[:,0]);gamma=np.exp(np.log(2)*np.tanh(raw[:,1]));gain=np.exp(np.log(2)*np.tanh(raw[:,2]));box=pre['rows'][i]['box'];lo=np.asarray(box['lower'])[0];hi=np.asarray(box['upper'])[0]
    for channel,grid in enumerate([ev,gamma]):assert (grid>=lo[channel]-1e-6).all() and (grid<=hi[channel]+1e-6).all()
    active=np.asarray(pre['rows'][i]['gate']['active']).reshape(2,2);assert (gain>=.5).all() and (gain<=2).all() and np.all(gain[:,~active]==1)
    assert sha(a.normal_root/item['normal'])==item['normal_sha256']
    with Image.open(a.normal_root/item['normal']) as image:y=(np.asarray(image.convert('RGB'),dtype=np.float64)/255).astype(np.float32).astype(np.float64)
    x=v['image'][0].permute(1,2,0).numpy().astype(np.float64);mse=float(torch.from_numpy(x-y).square().mean());psnr=float(-10*np.log10(mse));mx,my=smooth(x),smooth(y)
    ssim=float(np.mean(((2*mx*my+.0001)*(2*(smooth(x*y)-mx*my)+.0009))/((mx*mx+my*my+.0001)*(smooth(x*x)-mx*mx+smooth(y*y)-my*my+.0009))))
    errors.append(abs(mse-float(h['mse'][step])))
    val=dict(psnr=psnr,ssim=ssim,t035_psnr=float(old['oracle_psnr']),t035_ssim=float(old['oracle_ssim']),delta_psnr=psnr-float(old['oracle_psnr']),delta_ssim=ssim-float(old['oracle_ssim']))
    for k,n in val.items():values[k].append(n);errors.append(abs(n-main[k]))
assert len(steps)==100 and collections.Counter(map(str,steps))==summary['best_step_histogram'] and sum(s==1000 for s in steps)==summary['at_step1000']
for k,v in values.items():
    for name,n in [('mean',statistics.mean(v)),('median',statistics.median(v))]:errors.append(abs(n-summary['metrics'][k][name]))
for k in ['psnr','ssim']:
    v=values['delta_'+k];assert dict(win=sum(x>0 for x in v),equal=sum(x==0 for x in v),loss=sum(x<0 for x in v))==summary['win_equal_loss'][k]
classification='T035 common-gain oracle materially underconverged' if statistics.mean(values['delta_psnr'])>=1 and statistics.median(values['delta_psnr'])>=.75 else 'T035 common-gain oracle material underconvergence not supported under fixed extension'
assert classification==summary['classification'] and max(errors)<=1e-10
result=dict(status='PASS',images=100,history_states=100100,selected_output_metrics=200,scalar_checks=len(errors),max_abs_error=max(errors),classification=classification,all_states_finite_bounded=True,all_earliest_selected_raws_exact=True,preflight_before_normals=True,freeze_before_metrics=True,seconds=time.perf_counter()-start)
(a.out/'independent_replay.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
