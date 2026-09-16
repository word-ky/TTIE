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
assert sha(a.accepted/'freeze.json')==pre['prior_freeze_sha256'] and sha(a.accepted/'pairs.json')==receipt['prior_csv_sha256']=='3c92f22701a69591f78737b3d90a6cfb84fe3a62efe6911ad5db29189752ffd1'
assert sha(a.split)==pre['split_sha256'];split=json.loads(a.split.read_bytes())['selected'];baseline=json.loads((a.accepted/'pairs.json').read_bytes());starts=torch.load(a.preflight/'starts.pt',weights_only=True,map_location='cpu');assert sha(a.preflight/'starts.pt')==pre['starts_sha256']
weights=np.exp(-np.arange(-5,6,dtype=np.float64)**2/4.5);weights/=weights.sum()
def smooth(v):return convolve1d(convolve1d(v,weights,axis=0,mode='reflect'),weights,axis=1,mode='reflect')
errors=[];values={k:[] for k in ['psnr','ssim','t049_psnr','t049_ssim','delta_psnr','delta_ssim','total_vs_t048_psnr','total_vs_t048_ssim']};steps=[];segments=[];knot_values=[];render_errors=[];initial_lifts=torch.load(a.preflight/'lifts.pt',weights_only=True,map_location='cpu');assert sha(a.preflight/'lifts.pt')==pre['lifts_sha256'];initial_qs=torch.load(a.preflight/'qs.pt',weights_only=True,map_location='cpu');assert sha(a.preflight/'qs.pt')==pre['qs_sha256']
for i,(row,item,old,main) in enumerate(zip(f['rows'],split,baseline,pairs)):
    assert row['low']==item['low']==old['low']==main['low'];root=a.out/f'{i:03d}'
    for name,h in row['files'].items():assert sha(root/name)==h
    h=torch.load(root/'history.pt',weights_only=True,map_location='cpu');v=torch.load(root/'output.pt',weights_only=True,map_location='cpu')
    assert h['q'].shape==(1001,4,8) and h['mse'].shape==(1001,) and torch.isfinite(h['q']).all() and torch.isfinite(h['mse']).all()
    step=min(range(1001),key=lambda j:float(h['mse'][j]));steps.append(step)
    assert step==row['best_step']==main['best_step'] and torch.equal(v['q'],h['q'][step]) and torch.equal(starts[i],v['raw']) and torch.equal(initial_lifts[i],v['lift']) and torch.equal(h['q'][0],initial_qs[i]) and thash(initial_qs[i])==pre['rows'][i]['q_sha256']
    assert thash(v['raw'])==row['raw_sha256'] and thash(v['image'])==row['output_sha256'] and thash(starts[i])==pre['rows'][i]['raw_sha256']
    active=np.asarray(pre['rows'][i]['gate']['active']);q=h['q'].numpy().astype(np.float64)
    assert np.all(q[:,~active]==0) and thash(v['q'])==row['q_sha256'] and thash(v['knots'])==row['knots_sha256'] and thash(v['lift'])==row['lift_sha256']
    d=np.logaddexp(0,q);width=d/d.sum(axis=-1,keepdims=True);all_knots=np.concatenate([np.zeros_like(width[...,:1]),np.cumsum(width,axis=-1)[...,:7],np.ones_like(width[...,:1])],axis=-1)
    assert np.isfinite(all_knots).all() and np.all(np.diff(all_knots,axis=-1)>=0)
    assert np.max(np.abs(all_knots[step]-v['knots'].numpy()))<=1e-6
    selected=v['knots'].numpy()[active];assert np.all(np.diff(selected,axis=-1)>=0) and np.all(selected[:,0]==0) and np.all(selected[:,-1]==1)
    segments.extend(np.diff(selected,axis=-1).flatten().tolist());knot_values.extend(selected[:,1:-1].flatten().tolist())
    assert sha(a.low_root/item['low'])==item['low_sha256']
    with Image.open(a.low_root/item['low']) as image:low=torch.from_numpy((np.asarray(image.convert('RGB'),dtype=np.float64)/255).astype(np.float32)).permute(2,0,1).unsqueeze(0)
    raw=v['raw'];bounded=raw.tanh();ev=2*bounded[:,0:1];gamma=(math.log(2)*bounded[:,1:2]).exp();gain=(math.log(2)*bounded[:,2:3]).exp()
    height,width=low.shape[-2:];yy=(torch.arange(height)>=height//2).long();xx=(torch.arange(width)>=width//2).long()
    ev=ev[:,:,yy[:,None],xx[None,:]];gamma=gamma[:,:,yy[:,None],xx[None,:]];gain=gain[:,:,yy[:,None],xx[None,:]]
    expected=low*torch.exp2(ev);expected=(expected+1e-6).pow(gamma)-torch.pow(1e-6,gamma);expected=expected*gain
    expected=(.5+(expected+v['lift'][yy[:,None],xx[None,:]]-.5)).clamp(0,1)
    # Independently interpolate in float64 from softplus/cumulative knots.
    base=expected.numpy();region=(yy[:,None]*2+xx[None,:]).numpy();toned=np.empty_like(base)
    for region_id in range(4):
        mask=region==region_id;toned[:,:,mask]=np.interp(base[:,:,mask],np.arange(9)/8,all_knots[step,region_id])
    expected=torch.where(torch.from_numpy(active.reshape(2,2))[yy[:,None],xx[None,:]],torch.from_numpy(toned),low)
    assert torch.equal(v['image'][:,:,~torch.from_numpy(active.reshape(2,2))[yy[:,None],xx[None,:]]],low[:,:,~torch.from_numpy(active.reshape(2,2))[yy[:,None],xx[None,:]]])
    render_errors.append(float((expected-v['image']).abs().max()));assert render_errors[-1]<=1e-6
    assert sha(a.normal_root/item['normal'])==item['normal_sha256']
    with Image.open(a.normal_root/item['normal']) as image:y=(np.asarray(image.convert('RGB'),dtype=np.float64)/255).astype(np.float32).astype(np.float64)
    x=v['image'][0].permute(1,2,0).numpy().astype(np.float64);mse=float(torch.from_numpy(x-y).square().mean());psnr=float(-10*np.log10(mse));mx,my=smooth(x),smooth(y)
    ssim=float(np.mean(((2*mx*my+.0001)*(2*(smooth(x*y)-mx*my)+.0009))/((mx*mx+my*my+.0001)*(smooth(x*x)-mx*mx+smooth(y*y)-my*my+.0009))))
    errors.append(abs(mse-float(h['mse'][step])))
    val=dict(psnr=psnr,ssim=ssim,t049_psnr=float(old['psnr']),t049_ssim=float(old['ssim']),delta_psnr=psnr-float(old['psnr']),delta_ssim=ssim-float(old['ssim']),total_vs_t048_psnr=psnr-float(old['t048_psnr']),total_vs_t048_ssim=ssim-float(old['t048_ssim']))
    for k,n in val.items():values[k].append(n);errors.append(abs(n-main[k]))
assert len(steps)==100 and collections.Counter(map(str,steps))==summary['best_step_histogram'] and sum(s==1000 for s in steps)==summary['at_step1000']
for k,v in values.items():
    for name,n in [('mean',statistics.mean(v)),('median',statistics.median(v))]:errors.append(abs(n-summary['metrics'][k][name]))
for k in ['psnr','ssim']:
    v=values['delta_'+k];assert dict(win=sum(x>0 for x in v),equal=sum(x==0 for x in v),loss=sum(x<0 for x in v))==summary['win_equal_loss'][k]
classification='material monotonic-tone underconvergence supported' if statistics.mean(values['delta_psnr'])>=1 and statistics.median(values['delta_psnr'])>=.5 else 'material monotonic-tone underconvergence not supported under fixed extension'
d=summary['tone_distribution'];assert d['active_curves']==len(segments)//8 and d['segment_count']==len(segments) and d['zero_width_segments']==sum(v==0 for v in segments) and d['fixed_endpoints']==[0,1]
for k,n in [('segment_min',min(segments)),('segment_max',max(segments)),('segment_mean',statistics.mean(segments)),('segment_median',statistics.median(segments)),('interior_knot_min',min(knot_values)),('interior_knot_max',max(knot_values)),('interior_knot_mean',statistics.mean(knot_values)),('interior_knot_median',statistics.median(knot_values))]:errors.append(abs(d[k]-n))
assert classification==summary['classification'] and max(errors)<=1e-10
result=dict(status='PASS',images=100,history_states=100100,selected_output_metrics=200,scalar_checks=len(errors),max_abs_error=max(errors),classification=classification,independent_renderer_max_abs=max(render_errors),all_t048_coordinates_exact=True,all_luts_monotonic=True,inactive_outputs_exact=True,all_states_finite_bounded=True,all_earliest_selected_raws_exact=True,preflight_before_normals=True,freeze_before_metrics=True,seconds=time.perf_counter()-start)
(a.out/'independent_replay.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
