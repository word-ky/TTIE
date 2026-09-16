"""Independent field interpolation, frozen renderer, history and metric replay."""
import argparse,json,math,hashlib,statistics,collections,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import convolve1d
p=argparse.ArgumentParser()
for k in ['split','low-root','normal-root','accepted','preflight','out']:p.add_argument('--'+k,type=Path,required=True)
p.add_argument('--receipt',type=Path,required=True)
a=p.parse_args();torch.set_num_threads(1);start=time.perf_counter()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def thash(x):return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
f=read(a.out/'freeze.json');pre=read(a.preflight/'preflight.json');receipt=read(a.out/'evaluation_receipt.json');summary=read(a.out/'summary.json');pairs=read(a.out/'pairs.json')
assert sha(a.out/'freeze.json')==receipt['freeze_sha256'] and sha(a.preflight/'preflight.json')==f['preflight_sha256']
assert all(pre['completed_utc']<v['utc'] for v in f['opened']) and all(f['completed_utc']<v['utc'] for v in receipt['normal_opens'])
assert sha(a.accepted/'freeze.json')==pre['prior_freeze_sha256']=='d9130a48a8959b0ef77715bfb1f1fd22d61b19000b9994f62dab96fa9ec83c55'
assert sha(a.accepted/'pairs.json')==receipt['prior_csv_sha256']=='34c7752f8d415c982728ec58a85906c5186b3367a516f1d76957a8c239a96d5e'
assert sha(a.split)==pre['split_sha256'];split=read(a.split)['selected'];baseline=read(a.accepted/'pairs.json')
starts={}
for name,key in [('starts','starts_sha256'),('lifts','lifts_sha256'),('qs','qs_sha256'),('us','us_sha256'),('bs','bs_sha256'),('vs','vs_sha256')]:
 assert sha(a.preflight/(name+'.pt'))==pre[key];starts[name]=torch.load(a.preflight/(name+'.pt'),weights_only=True,map_location='cpu')
weights=np.exp(-np.arange(-5,6,dtype=np.float64)**2/4.5);weights/=weights.sum()
def smooth(v):return convolve1d(convolve1d(v,weights,axis=0,mode='reflect'),weights,axis=1,mode='reflect')
def interpolate(grid,h,w):
 # CUDA source coordinates use a single-rounding multiply-add. Emulate that arithmetic in NumPy.
 def coords(n):return np.maximum(((np.arange(n,dtype=np.float32)+.5).astype(np.float64)*float(np.float32(8/n))-.5).astype(np.float32),0)
 yy=coords(h);xx=coords(w)
 y0=np.floor(yy).astype(int);x0=np.floor(xx).astype(int);y1=np.minimum(y0+1,7);x1=np.minimum(x0+1,7);fy=yy-y0.astype(np.float32);fx=xx-x0.astype(np.float32)
 # Match CUDA's contracted weighted sums, globally for every grid value.
 def fma(a,b,c):return (a.astype(np.float64)*b.astype(np.float64)+c.astype(np.float64)).astype(np.float32)
 top=fma(1-fx,grid[y0[:,None],x0[None,:]],fx*grid[y0[:,None],x1[None,:]])
 bot=fma(1-fx,grid[y1[:,None],x0[None,:]],fx*grid[y1[:,None],x1[None,:]])
 return fma(1-fy[:,None],top,fy[:,None]*bot)
errors=[];render_errors=[];field_errors=[];basis_errors=[];steps=[];fields={k:[] for k in ['selected_controls','selected_field']};values={k:[] for k in ['psnr','ssim','t054_psnr','t054_ssim','delta_psnr','delta_ssim']}
for i,(row,item,old,main) in enumerate(zip(f['rows'],split,baseline,pairs)):
 root=a.out/f'{i:03d}';prior_root=a.accepted/f'{i:03d}';assert row['low']==item['low']==old['low']==main['low']
 for name,h in row['files'].items():assert sha(root/name)==h
 for name,h in pre['rows'][i]['prior_files'].items():assert sha(prior_root/name)==h
 hist=torch.load(root/'history.pt',weights_only=True,map_location='cpu');v=torch.load(root/'output.pt',weights_only=True,map_location='cpu');initial=torch.load(prior_root/'output.pt',weights_only=True,map_location='cpu');vs=hist['v']
 assert vs.shape==(1001,1,1,8,8) and hist['mse'].shape==(1001,) and torch.isfinite(vs).all() and torch.isfinite(hist['mse']).all() and torch.equal(vs[0],initial['v']) and torch.equal(vs[0],starts['vs'][i]) and thash(vs[0])==pre['rows'][i]['v_sha256']
 step=min(range(1001),key=lambda j:float(hist['mse'][j]));steps.append(step);assert step==row['best_step']==main['best_step'] and torch.equal(v['v'],vs[step])
 for name,key in [('raw','starts'),('lift','lifts'),('q','qs'),('u','us'),('b','bs')]:assert torch.equal(v[name],starts[key][i]) and thash(v[name])==row[name+'_sha256']==pre['rows'][i][name+'_sha256']
 for name in ['raw','lift','q','u','b','knots','ev']:assert torch.equal(v[name],initial[name])
 for name,key in [('image','output_sha256'),('v','v_sha256'),('c','c_sha256'),('c_controls','c_controls_sha256'),('y0','y0_sha256'),('detail','detail_sha256'),('knots','knots_sha256'),('ev','ev_sha256')]:assert thash(v[name])==row[key]
 assert torch.equal(v['y0'],initial['y0']) and thash(v['y0'])==pre['rows'][i]['y0_sha256'] and thash(v['detail'])==pre['rows'][i]['basis_sha256']
 y0=v['y0'].numpy();h,w=y0.shape[-2:];k=np.array([1,4,6,4,1],dtype=np.float64)/16
 # scipy mirror excludes the boundary value, matching PyTorch reflect padding.
 blurred=convolve1d(convolve1d(y0.astype(np.float64),k,axis=-1,mode='mirror'),k,axis=-2,mode='mirror');basis=y0.astype(np.float64)-blurred;basis_errors.append(float(np.max(np.abs(basis-v['detail'].numpy()))));assert basis_errors[-1]<=1e-6
 expected_c=np.tanh(interpolate(v['v'].numpy()[0,0],h,w));field_errors.append(float(np.max(np.abs(expected_c-v['c'].numpy()[0,0]))));assert field_errors[-1]<=1e-6
 expected_controls=np.tanh(v['v'].numpy());assert np.max(np.abs(expected_controls-v['c_controls'].numpy()))<=1e-6 and v['c'].min()>=-1 and v['c'].max()<=1
 yy=(np.arange(h)>=h//2).astype(int);xx=(np.arange(w)>=w//2).astype(int);mask=np.array(pre['rows'][i]['gate']['active']).reshape(2,2)[yy[:,None],xx[None,:]]
 assert torch.equal(v['mask'],torch.from_numpy(mask)) and thash(v['mask'])==pre['rows'][i]['mask_sha256']
 expected=np.where(mask,np.clip(y0+v['c'].numpy()*v['detail'].numpy(),0,1),y0);render_errors.append(float(np.max(np.abs(expected-v['image'].numpy()))));assert render_errors[-1]<=1e-6
 assert torch.equal(v['image'][:,:,~torch.from_numpy(mask)],v['y0'][:,:,~torch.from_numpy(mask)])
 fields['selected_controls'].extend(v['c_controls'].numpy().flatten().tolist());fields['selected_field'].extend(v['c'].numpy().flatten().tolist())
 assert sha(a.normal_root/item['normal'])==item['normal_sha256']
 with Image.open(a.normal_root/item['normal']) as image:target=(np.asarray(image.convert('RGB'),dtype=np.float64)/255).astype(np.float32).astype(np.float64)
 output=v['image'].numpy()[0].transpose(1,2,0).astype(np.float64);mse=float(np.mean((output-target)**2));psnr=-10*math.log10(max(mse,1e-12));mu1=smooth(output);mu2=smooth(target);var1=smooth(output*output)-mu1*mu1;var2=smooth(target*target)-mu2*mu2;cov=smooth(output*target)-mu1*mu2;ssim=float(np.mean(((2*mu1*mu2+.01**2)*(2*cov+.03**2))/((mu1*mu1+mu2*mu2+.01**2)*(var1+var2+.03**2))))
 errors.append(abs(mse-float(hist['mse'][step])))
 calculated=dict(psnr=psnr,ssim=ssim,t054_psnr=old['psnr'],t054_ssim=old['ssim'],delta_psnr=psnr-old['psnr'],delta_ssim=ssim-old['ssim'])
 for key,value in calculated.items():values[key].append(value);errors.append(abs(value-main[key]))
assert len(steps)==100 and collections.Counter(map(str,steps))==summary['best_step_histogram'] and sum(v==1000 for v in steps)==summary['at_step1000']
for k,v in values.items():errors.extend([abs(statistics.mean(v)-summary['metrics'][k]['mean']),abs(statistics.median(v)-summary['metrics'][k]['median'])])
for k in ['psnr','ssim']:
 v=values['delta_'+k];assert summary['win_equal_loss'][k]==dict(win=sum(x>0 for x in v),equal=sum(x==0 for x in v),loss=sum(x<0 for x in v))
for name,data in fields.items():
 fv=summary['c_distributions'][name];arr=np.asarray(data,dtype=np.float64);assert fv['count']==len(data) and fv['exact_hits']=={str(bound):int(np.count_nonzero(arr==float(np.float32(bound)))) for bound in [-1.,1.]}
 errors.extend([abs(float(np.mean(arr<=float(np.float32(-.99))))-fv['fraction_le_neg099']),abs(float(np.mean(arr>=float(np.float32(.99))))-fv['fraction_ge_pos099'])])
 for k,val in [('min',min(data)),('max',max(data)),('mean',statistics.mean(data)),('median',statistics.median(data))]:errors.append(abs(val-fv[k]))
classification='material local-detail underconvergence supported' if statistics.mean(values['delta_psnr'])>=.25 and statistics.median(values['delta_psnr'])>=.10 and statistics.mean(values['delta_ssim'])>=.010 else 'material local-detail underconvergence not supported under fixed extension'
assert classification==summary['classification'] and max(errors)<=1e-10
result=dict(status='PASS',images=100,history_states=100100,selected_output_metrics=200,scalar_checks=len(errors),max_abs_error=max(errors),independent_renderer_max_abs=max(render_errors),independent_basis_max_abs=max(basis_errors),independent_interpolation_max_abs=max(field_errors),all_old_coordinates_exact=True,inactive_outputs_exact=True,preflight_before_normals=True,freeze_before_metrics=True,classification=classification,seconds=time.perf_counter()-start)
a.receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
