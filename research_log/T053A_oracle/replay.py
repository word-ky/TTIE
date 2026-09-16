"""Independent field interpolation, frozen renderer, history and metric replay."""
import argparse,json,math,hashlib,statistics,collections,time
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
def read(p):return json.loads(p.read_bytes())
f=read(a.out/'freeze.json');pre=read(a.preflight/'preflight.json');receipt=read(a.out/'evaluation_receipt.json');summary=read(a.out/'summary.json');pairs=read(a.out/'pairs.json')
assert sha(a.out/'freeze.json')==receipt['freeze_sha256'] and sha(a.preflight/'preflight.json')==f['preflight_sha256']
assert all(pre['completed_utc']<v['utc'] for v in f['opened']) and all(f['completed_utc']<v['utc'] for v in receipt['normal_opens'])
assert sha(a.accepted/'freeze.json')==pre['prior_freeze_sha256']=='83ebd03b57e4c5e43191194bc66d507f56090ed593d9681d140ffa29b4e3baa8'
assert sha(a.accepted/'pairs.json')==receipt['prior_csv_sha256']=='b0c571e305f28e30b4acbe9b1dc0e7d3b06b5fc577f266fcf5d42641368555ff'
assert sha(a.split)==pre['split_sha256'];split=read(a.split)['selected'];baseline=read(a.accepted/'pairs.json')
starts={}
for name,key in [('starts','starts_sha256'),('lifts','lifts_sha256'),('qs','qs_sha256'),('us','us_sha256'),('bs','bs_sha256')]:
 assert sha(a.preflight/(name+'.pt'))==pre[key];starts[name]=torch.load(a.preflight/(name+'.pt'),weights_only=True,map_location='cpu')
weights=np.exp(-np.arange(-5,6,dtype=np.float64)**2/4.5);weights/=weights.sum()
def smooth(v):return convolve1d(convolve1d(v,weights,axis=0,mode='reflect'),weights,axis=1,mode='reflect')
def interpolate(grid,h,w):
 # CUDA source coordinates use a single-rounding multiply-add. Emulate that arithmetic in NumPy.
 def coords(n):return np.maximum(((np.arange(n,dtype=np.float32)+.5).astype(np.float64)*float(np.float32(8/n))-.5).astype(np.float32),0)
 yy=coords(h);xx=coords(w)
 y0=np.floor(yy).astype(int);x0=np.floor(xx).astype(int);y1=np.minimum(y0+1,7);x1=np.minimum(x0+1,7);fy=yy-y0.astype(np.float32);fx=xx-x0.astype(np.float32)
 top=grid[y0[:,None],x0[None,:]]*(1-fx)+grid[y0[:,None],x1[None,:]]*fx
 bot=grid[y1[:,None],x0[None,:]]*(1-fx)+grid[y1[:,None],x1[None,:]]*fx
 return top*(1-fy[:,None])+bot*fy[:,None]
errors=[];render_errors=[];field_errors=[];steps=[];fields={k:[] for k in ['starting_controls','selected_controls','starting_field','selected_field']};values={k:[] for k in ['psnr','ssim','t052_psnr','t052_ssim','delta_psnr','delta_ssim']}
for i,(row,item,old,main) in enumerate(zip(f['rows'],split,baseline,pairs)):
 root=a.out/f'{i:03d}';assert row['low']==item['low']==old['low']==main['low']
 for name,h in row['files'].items():assert sha(root/name)==h
 hist=torch.load(root/'history.pt',weights_only=True,map_location='cpu');v=torch.load(root/'output.pt',weights_only=True,map_location='cpu');bs=hist['b']
 assert bs.shape==(501,1,1,8,8) and hist['mse'].shape==(501,) and torch.isfinite(bs).all() and torch.isfinite(hist['mse']).all() and torch.equal(bs[0],starts['bs'][i]) and bs.min()>=-.4 and bs.max()<=.4
 step=min(range(501),key=lambda j:float(hist['mse'][j]));steps.append(step)
 assert step==row['best_step']==main['best_step'] and torch.equal(v['b'],bs[step])
 for name,key in [('raw','starts'),('lift','lifts'),('q','qs'),('u','us')]:assert torch.equal(v[name],starts[key][i]) and thash(v[name])==row[name+'_sha256']==pre['rows'][i][name+'_sha256']
 for name,key in [('b','b_sha256'),('offset','offset_sha256'),('u','u_sha256'),('ev','ev_sha256'),('image','output_sha256'),('knots','knots_sha256')]:assert thash(v[name])==row[key]
 d=np.logaddexp(0,v['q'].numpy().astype(np.float64));d/=d.sum(-1,keepdims=True);knots=np.concatenate([np.zeros((4,1)),np.cumsum(d,axis=-1)[:,:7],np.ones((4,1))],-1);assert np.max(np.abs(knots-v['knots'].numpy()))<=1e-6
 assert sha(a.low_root/item['low'])==item['low_sha256']
 with Image.open(a.low_root/item['low']) as image:low=torch.from_numpy((np.asarray(image.convert('RGB'),dtype=np.float64)/255).astype(np.float32)).permute(2,0,1).unsqueeze(0)
 raw=v['raw'];bounded=raw.tanh();ev=2*bounded[:,0:1];gamma=(math.log(2)*bounded[:,1:2]).exp();gain=(math.log(2)*bounded[:,2:3]).exp();h,w=low.shape[-2:];yy=(torch.arange(h)>=h//2).long();xx=(torch.arange(w)>=w//2).long()
 ev=ev[:,:,yy[:,None],xx[None,:]];gamma=gamma[:,:,yy[:,None],xx[None,:]];gain=gain[:,:,yy[:,None],xx[None,:]]
 z=low*torch.exp2(ev);z=(z+1e-6).pow(gamma)-torch.pow(1e-6,gamma);z=(.5+(z*gain+v['lift'][yy[:,None],xx[None,:]]-.5)).clamp(0,1)
 field=interpolate(2*np.tanh(v['u'].numpy()[0,0]),h,w);field_errors.append(float(np.max(np.abs(field-v['ev'].numpy()[0,0]))));assert field_errors[-1]<=1e-6
 offset=interpolate(v['b'].numpy()[0,0],h,w);field_errors.append(float(np.max(np.abs(offset-v['offset'].numpy()[0,0]))));assert field_errors[-1]<=1e-6
 # Use saved exact field after independent interpolation verification to isolate float32 renderer rounding.
 z=(z*torch.exp2(v['ev'])+v['offset']).clamp(0,1).numpy();scaled=z*8;segment=np.minimum(np.floor(scaled).astype(int),7);region=(yy[:,None]*2+xx[None,:]).numpy();index=region*9+segment;y=v['knots'].numpy().flatten();expected=y[index]+(y[index+1]-y[index])*(scaled-segment).astype(np.float32)
 mask=np.array(pre['rows'][i]['gate']['active']).reshape(2,2)[yy[:,None],xx[None,:]];expected=np.where(mask,expected,low.numpy());render_errors.append(float(np.max(np.abs(expected-v['image'].numpy()))));assert render_errors[-1]<=1e-6
 assert torch.equal(v['image'][:,:,~torch.from_numpy(mask)],low[:,:,~torch.from_numpy(mask)])
 initial_root=a.accepted/f'{i:03d}'
 for name,digest in pre['rows'][i]['prior_files'].items():assert sha(initial_root/name)==digest
 initial=torch.load(initial_root/'output.pt',weights_only=True,map_location='cpu')
 for name in ['raw','lift','q','u','knots','ev']:assert torch.equal(v[name],initial[name])
 initial_field=interpolate(initial['b'].numpy()[0,0],h,w);field_errors.append(float(np.max(np.abs(initial_field-initial['offset'].numpy()[0,0]))));assert field_errors[-1]<=1e-6
 fields['starting_controls'].extend(initial['b'].numpy().flatten().tolist());fields['selected_controls'].extend(v['b'].numpy().flatten().tolist());fields['starting_field'].extend(initial['offset'].numpy()[0,0][mask].tolist());fields['selected_field'].extend(v['offset'].numpy()[0,0][mask].tolist())
 assert sha(a.normal_root/item['normal'])==item['normal_sha256']
 with Image.open(a.normal_root/item['normal']) as image:target=(np.asarray(image.convert('RGB'),dtype=np.float64)/255).astype(np.float32).astype(np.float64)
 output=v['image'].numpy()[0].transpose(1,2,0).astype(np.float64);mse=float(np.mean((output-target)**2));psnr=-10*math.log10(max(mse,1e-12));mu1=smooth(output);mu2=smooth(target);var1=smooth(output*output)-mu1*mu1;var2=smooth(target*target)-mu2*mu2;cov=smooth(output*target)-mu1*mu2;ssim=float(np.mean(((2*mu1*mu2+.01**2)*(2*cov+.03**2))/((mu1*mu1+mu2*mu2+.01**2)*(var1+var2+.03**2))))
 errors.append(abs(mse-float(hist['mse'][step])))
 calculated=dict(psnr=psnr,ssim=ssim,t052_psnr=old['psnr'],t052_ssim=old['ssim'],delta_psnr=psnr-old['psnr'],delta_ssim=ssim-old['ssim'])
 for k,value in calculated.items():values[k].append(value);errors.append(abs(value-main[k]))
assert len(steps)==100 and collections.Counter(map(str,steps))==summary['best_step_histogram'] and sum(v==500 for v in steps)==summary['at_step500']
for k,v in values.items():errors.extend([abs(statistics.mean(v)-summary['metrics'][k]['mean']),abs(statistics.median(v)-summary['metrics'][k]['median'])])
for k in ['psnr','ssim']:
 v=values['delta_'+k];assert summary['win_equal_loss'][k]==dict(win=sum(x>0 for x in v),equal=sum(x==0 for x in v),loss=sum(x<0 for x in v))
for name,data in fields.items():
 fv=summary['b_distributions'][name];arr=np.asarray(data,dtype=np.float64);assert fv['count']==len(data) and fv['exact_hits']=={str(bound):int(np.count_nonzero(arr==float(np.float32(bound)))) for bound in [-.4,-.2,.2,.4]}
 for k,val in [('min',min(data)),('max',max(data)),('mean',statistics.mean(data)),('median',statistics.median(data))]:errors.append(abs(val-fv[k]))
classification='additive-range bottleneck supported' if statistics.mean(values['delta_psnr'])>=.5 and statistics.median(values['delta_psnr'])>=.25 and statistics.mean(values['delta_ssim'])>=0 else 'not supported under fixed range-closure probe'
assert classification==summary['classification'] and max(errors)<=1e-10
result=dict(status='PASS',images=100,history_states=50100,selected_output_metrics=200,scalar_checks=len(errors),max_abs_error=max(errors),independent_renderer_max_abs=max(render_errors),independent_interpolation_max_abs=max(field_errors),all_old_coordinates_exact=True,inactive_outputs_exact=True,preflight_before_normals=True,freeze_before_metrics=True,classification=classification,seconds=time.perf_counter()-start)
(a.out/'independent_replay.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
