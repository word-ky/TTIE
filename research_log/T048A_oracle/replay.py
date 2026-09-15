"""Independent raw-history winner and complete metric/statistic replay."""
import argparse,json,csv,math,hashlib,statistics,collections,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import convolve1d
p=argparse.ArgumentParser()
for k in ['split','low-root','normal-root','accepted','preflight','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();torch.set_num_threads(1);start=time.perf_counter()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def thash(x):return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
f=json.loads((a.out/'freeze.json').read_bytes());pre=json.loads((a.preflight/'preflight.json').read_bytes());receipt=json.loads((a.out/'evaluation_receipt.json').read_bytes());summary=json.loads((a.out/'summary.json').read_bytes());pairs=json.loads((a.out/'pairs.json').read_bytes())
assert sha(a.out/'freeze.json')==receipt['freeze_sha256'] and sha(a.preflight/'preflight.json')==f['preflight_sha256']
assert all(pre['completed_utc']<v['utc'] for v in f['opened']) and all(f['completed_utc']<v['utc'] for v in receipt['normal_opens'])
assert sha(a.accepted/'freeze.json')==pre['prior_freeze_sha256'] and sha(a.accepted/'pairs.json')==receipt['prior_csv_sha256']=='d4b12f46734c05369efea777b5e699e659d82a16d1e969d4bd93a9cdecd37790'
assert sha(a.split)==pre['split_sha256'];split=json.loads(a.split.read_bytes())['selected'];baseline=json.loads((a.accepted/'pairs.json').read_bytes());starts=torch.load(a.preflight/'starts.pt',weights_only=True,map_location='cpu');assert sha(a.preflight/'starts.pt')==pre['starts_sha256']
weights=np.exp(-np.arange(-5,6,dtype=np.float64)**2/4.5);weights/=weights.sum()
def smooth(v):return convolve1d(convolve1d(v,weights,axis=0,mode='reflect'),weights,axis=1,mode='reflect')
errors=[];values={k:[] for k in ['psnr','ssim','t047_psnr','t047_ssim','delta_psnr','delta_ssim','total_vs_t046_psnr','total_vs_t046_ssim']};steps=[];lifts=[];gains=[];render_errors=[];initial_lifts=torch.load(a.preflight/'lifts.pt',weights_only=True,map_location='cpu');assert sha(a.preflight/'lifts.pt')==pre['lifts_sha256']
for i,(row,item,old,main) in enumerate(zip(f['rows'],split,baseline,pairs)):
    assert row['low']==item['low']==old['low']==main['low'];root=a.out/f'{i:03d}'
    for name,h in row['files'].items():assert sha(root/name)==h
    h=torch.load(root/'history.pt',weights_only=True,map_location='cpu');v=torch.load(root/'output.pt',weights_only=True,map_location='cpu')
    assert h['lift'].shape==(501,2,2) and h['mse'].shape==(501,) and torch.isfinite(h['lift']).all() and torch.isfinite(h['mse']).all()
    step=min(range(501),key=lambda j:float(h['mse'][j]));steps.append(step)
    assert step==row['best_step']==main['best_step'] and torch.equal(v['lift'],h['lift'][step]) and torch.equal(starts[i,:,:2],v['raw'][:,:2]) and torch.equal(starts[i,:,2:3],h['gain_raw'][0]) and torch.equal(initial_lifts[i],h['lift'][0]) and torch.equal(v['raw'][:,2:3],h['gain_raw'][step])
    assert thash(v['raw'])==row['raw_sha256'] and thash(v['image'])==row['output_sha256'] and thash(starts[i])==pre['rows'][i]['raw_sha256']
    assert h['gain_raw'].shape==(501,1,1,2,2) and torch.isfinite(h['gain_raw']).all() and thash(v['raw'][:,2:3])==row['gain_raw_sha256']
    mapped=(math.log(2)*h['gain_raw'].tanh()).exp();assert (mapped>=.5).all() and (mapped<=2).all()
    active=np.asarray(pre['rows'][i]['gate']['active']).reshape(2,2);lift=h['lift'].numpy();assert np.isfinite(lift).all() and np.all(np.abs(lift)<=np.float32(.2)) and np.all(lift[:,~active]==0)
    assert torch.count_nonzero(h['gain_raw'][:,:,:,~torch.from_numpy(active)])==0
    gains.extend(mapped[step,0,0][torch.from_numpy(active)].tolist())
    assert thash(v['lift'])==row['lift_sha256'];lifts.extend(v['lift'].numpy()[active].tolist())
    assert sha(a.low_root/item['low'])==item['low_sha256']
    with Image.open(a.low_root/item['low']) as image:low=torch.from_numpy((np.asarray(image.convert('RGB'),dtype=np.float64)/255).astype(np.float32)).permute(2,0,1).unsqueeze(0)
    raw=v['raw'];bounded=raw.tanh();ev=2*bounded[:,0:1];gamma=(math.log(2)*bounded[:,1:2]).exp();gain=(math.log(2)*bounded[:,2:3]).exp()
    height,width=low.shape[-2:];yy=(torch.arange(height)>=height//2).long();xx=(torch.arange(width)>=width//2).long()
    ev=ev[:,:,yy[:,None],xx[None,:]];gamma=gamma[:,:,yy[:,None],xx[None,:]];gain=gain[:,:,yy[:,None],xx[None,:]]
    expected=low*torch.exp2(ev);expected=(expected+1e-6).pow(gamma)-torch.pow(1e-6,gamma);expected=expected*gain
    expected=(.5+(expected+v['lift'][yy[:,None],xx[None,:]]-.5)).clamp(0,1)
    expected=torch.where(torch.from_numpy(active)[yy[:,None],xx[None,:]],expected,low)
    render_errors.append(float((expected-v['image']).abs().max()));assert render_errors[-1]<=1e-6
    assert sha(a.normal_root/item['normal'])==item['normal_sha256']
    with Image.open(a.normal_root/item['normal']) as image:y=(np.asarray(image.convert('RGB'),dtype=np.float64)/255).astype(np.float32).astype(np.float64)
    x=v['image'][0].permute(1,2,0).numpy().astype(np.float64);mse=float(torch.from_numpy(x-y).square().mean());psnr=float(-10*np.log10(mse));mx,my=smooth(x),smooth(y)
    ssim=float(np.mean(((2*mx*my+.0001)*(2*(smooth(x*y)-mx*my)+.0009))/((mx*mx+my*my+.0001)*(smooth(x*x)-mx*mx+smooth(y*y)-my*my+.0009))))
    errors.append(abs(mse-float(h['mse'][step])))
    val=dict(psnr=psnr,ssim=ssim,t047_psnr=float(old['psnr']),t047_ssim=float(old['ssim']),delta_psnr=psnr-float(old['psnr']),delta_ssim=ssim-float(old['ssim']),total_vs_t046_psnr=psnr-float(old['t046_psnr']),total_vs_t046_ssim=ssim-float(old['t046_ssim']))
    for k,n in val.items():values[k].append(n);errors.append(abs(n-main[k]))
assert len(steps)==100 and collections.Counter(map(str,steps))==summary['best_step_histogram'] and sum(s==500 for s in steps)==summary['at_step500']
for k,v in values.items():
    for name,n in [('mean',statistics.mean(v)),('median',statistics.median(v))]:errors.append(abs(n-summary['metrics'][k][name]))
for k in ['psnr','ssim']:
    v=values['delta_'+k];assert dict(win=sum(x>0 for x in v),equal=sum(x==0 for x in v),loss=sum(x<0 for x in v))==summary['win_equal_loss'][k]
classification='post-gamma affine coupling materially supported' if statistics.mean(values['delta_psnr'])>=.5 and statistics.median(values['delta_psnr'])>=.25 else 'post-gamma affine coupling material increment not supported under fixed probe'
d=summary['lift_distribution'];assert d['active_coordinates']==len(lifts) and d['lower_bound_hits']==sum(v<=float(np.float32(-.2)) for v in lifts) and d['upper_bound_hits']==sum(v>=float(np.float32(.2)) for v in lifts)
for k,n in [('mean',statistics.mean(lifts)),('median',statistics.median(lifts)),('min',min(lifts)),('max',max(lifts))]:errors.append(abs(d[k]-n))
d=summary['gain_distribution'];assert d['active_coordinates']==len(gains) and d['lower_bound_hits']==sum(v<=.5 for v in gains) and d['upper_bound_hits']==sum(v>=2 for v in gains)
for k,n in [('mean',statistics.mean(gains)),('median',statistics.median(gains)),('min',min(gains)),('max',max(gains))]:errors.append(abs(d[k]-n))
assert classification==summary['classification'] and max(errors)<=1e-10
result=dict(status='PASS',images=100,history_states=50100,selected_output_metrics=200,scalar_checks=len(errors),max_abs_error=max(errors),classification=classification,independent_renderer_max_abs=max(render_errors),ev_gamma_raws_exact=True,all_states_finite_bounded=True,all_earliest_selected_raws_exact=True,preflight_before_normals=True,freeze_before_metrics=True,seconds=time.perf_counter()-start)
(a.out/'independent_replay.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
