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
assert sha(a.accepted/'freeze.json')==pre['prior_freeze_sha256']=='64ca0dd47ece431012fac01c9b0a319e24e42e62cff4683c587effe765abd6ae'
assert sha(a.accepted/'pairs.json')==receipt['prior_csv_sha256']=='c1c1fc5830c50f9ef9ca331747ea0df0a9a3af2d8e7617fff5f7d33f03ad104c'
assert sha(a.split)==pre['split_sha256'];split=read(a.split)['selected'];baseline=read(a.accepted/'pairs.json')
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
errors=[];render_errors=[];field_errors=[];basis_errors=[];steps=[];fields={k:[] for k in ['selected_controls','selected_field']};values={k:[] for k in ['psnr','ssim','t055_psnr','t055_ssim','delta_psnr','delta_ssim']}
for i,(row,item,old,main) in enumerate(zip(f['rows'],split,baseline,pairs)):
 root=a.out/f'{i:03d}';prior_root=a.accepted/f'{i:03d}';assert row['low']==item['low']==old['low']==main['low']
 for name,h in row['files'].items():assert sha(root/name)==h
 for name,h in pre['rows'][i]['prior_files'].items():assert sha(prior_root/name)==h
 hist=torch.load(root/'history.pt',weights_only=True,map_location='cpu');v=torch.load(root/'output.pt',weights_only=True,map_location='cpu');initial=torch.load(prior_root/'output.pt',weights_only=True,map_location='cpu');ws=hist['w']
 assert ws.shape==(501,1,1,8,8) and hist['mse'].shape==(501,) and torch.isfinite(ws).all() and torch.isfinite(hist['mse']).all() and torch.count_nonzero(ws[0])==0
 step=min(range(501),key=lambda j:float(hist['mse'][j]));steps.append(step);assert step==row['best_step']==main['best_step'] and torch.equal(v['w'],ws[step])
 for name in ['raw','lift','q','u','b','v','knots','ev','y0','detail','mask','c','c_controls']:assert torch.equal(v[name],initial[name]),name
 for name,hsh in row['hashes'].items():assert thash(v[name])==hsh
 assert torch.equal(v['one'],initial['image']) and thash(v['d2'])==pre['rows'][i]['d2_sha256']
 startfile=a.preflight/pre['rows'][i]['start_file'];assert sha(startfile)==pre['rows'][i]['start_file_sha256'];assert torch.equal(torch.load(startfile,weights_only=True,map_location='cpu'),v['one'])
 y0=v['y0'].numpy();h,w=y0.shape[-2:];k5=np.array([1,4,6,4,1],dtype=np.float64)/16;k9=np.array([1,8,28,56,70,56,28,8,1],dtype=np.float64)/256
 def blur(y,k):return convolve1d(convolve1d(y.astype(np.float64),k,axis=-1,mode='mirror'),k,axis=-2,mode='mirror')
 basis=blur(y0,k5)-blur(y0,k9);basis_errors.append(float(np.max(np.abs(basis-v['d2'].numpy()))));assert basis_errors[-1]<=1e-6
 expected_c=np.tanh(interpolate(v['w'].numpy()[0,0],h,w));field_errors.append(float(np.max(np.abs(expected_c-v['c2'].numpy()[0,0]))));assert field_errors[-1]<=1e-6
 assert np.max(np.abs(np.tanh(v['w'].numpy())-v['c2_controls'].numpy()))<=1e-6 and v['c2'].min()>=-1 and v['c2'].max()<=1
 yy=(np.arange(h)>=h//2).astype(int);xx=(np.arange(w)>=w//2).astype(int);mask=np.array(pre['rows'][i]['gate']['active']).reshape(2,2)[yy[:,None],xx[None,:]];assert torch.equal(v['mask'],torch.from_numpy(mask))
 expected=np.where(mask,np.clip(v['one'].numpy()+v['c2'].numpy()*v['d2'].numpy(),0,1),v['one'].numpy());render_errors.append(float(np.max(np.abs(expected-v['image'].numpy()))));assert render_errors[-1]<=1e-6
 assert torch.equal(v['image'][:,:,~torch.from_numpy(mask)],v['one'][:,:,~torch.from_numpy(mask)])
 fields['selected_controls'].extend(v['c2_controls'].numpy().flatten().tolist());fields['selected_field'].extend(v['c2'].numpy().flatten().tolist())
 assert sha(a.normal_root/item['normal'])==item['normal_sha256']
 with Image.open(a.normal_root/item['normal']) as image:target=(np.asarray(image.convert('RGB'),dtype=np.float64)/255).astype(np.float32).astype(np.float64)
 output=v['image'].numpy()[0].transpose(1,2,0).astype(np.float64);mse=float(np.mean((output-target)**2));psnr=-10*math.log10(max(mse,1e-12));mu1=smooth(output);mu2=smooth(target);var1=smooth(output*output)-mu1*mu1;var2=smooth(target*target)-mu2*mu2;cov=smooth(output*target)-mu1*mu2;ssim=float(np.mean(((2*mu1*mu2+.01**2)*(2*cov+.03**2))/((mu1*mu1+mu2*mu2+.01**2)*(var1+var2+.03**2))))
 errors.append(abs(mse-float(hist['mse'][step])))
 calculated=dict(psnr=psnr,ssim=ssim,t055_psnr=old['psnr'],t055_ssim=old['ssim'],delta_psnr=psnr-old['psnr'],delta_ssim=ssim-old['ssim'])
 for key,value in calculated.items():values[key].append(value);errors.append(abs(value-main[key]))
assert len(steps)==100 and collections.Counter(map(str,steps))==summary['best_step_histogram'] and sum(v==500 for v in steps)==summary['at_step500']
for k,v in values.items():errors.extend([abs(statistics.mean(v)-summary['metrics'][k]['mean']),abs(statistics.median(v)-summary['metrics'][k]['median'])])
for k in ['psnr','ssim']:
 v=values['delta_'+k];assert summary['win_equal_loss'][k]==dict(win=sum(x>0 for x in v),equal=sum(x==0 for x in v),loss=sum(x<0 for x in v))
for name,data in fields.items():
 fv=summary['c2_distributions'][name];arr=np.asarray(data,dtype=np.float64);assert fv['count']==len(data) and fv['exact_hits']=={str(bound):int(np.count_nonzero(arr==float(np.float32(bound)))) for bound in [-1.,1.]}
 errors.extend([abs(float(np.mean(arr<=float(np.float32(-.99))))-fv['fraction_le_neg099']),abs(float(np.mean(arr>=float(np.float32(.99))))-fv['fraction_ge_pos099'])])
 for k,val in [('min',min(data)),('max',max(data)),('mean',statistics.mean(data)),('median',statistics.median(data))]:errors.append(abs(val-fv[k]))
classification='coarser detail band materially supported' if statistics.mean(values['delta_psnr'])>=.5 and statistics.median(values['delta_psnr'])>=.25 and statistics.mean(values['delta_ssim'])>=.020 else 'not supported under fixed probe'
assert classification==summary['classification'] and max(errors)<=1e-10
result=dict(status='PASS',images=100,history_states=50100,selected_output_metrics=200,scalar_checks=len(errors),max_abs_error=max(errors),independent_renderer_max_abs=max(render_errors),independent_basis_max_abs=max(basis_errors),independent_interpolation_max_abs=max(field_errors),all_old_coordinates_exact=True,inactive_outputs_exact=True,preflight_before_normals=True,freeze_before_metrics=True,classification=classification,seconds=time.perf_counter()-start)
(a.out/'independent_replay.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
