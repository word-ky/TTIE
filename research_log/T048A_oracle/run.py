"""REFERENCE_ORACLE_ONLY: gain and lift jointly; frozen EV/gamma, exactly500updates."""
from core import *
import argparse,time
p=argparse.ArgumentParser()
for k in ['split','low-root','normal-root','accepted','preflight','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(7);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert sha(a.split)==SPLIT_SHA and sha(a.accepted/'freeze.json')==PRIOR_FREEZE
split=json.loads(a.split.read_bytes())['selected'];pre=json.loads((a.preflight/'preflight.json').read_bytes());assert len(pre['rows'])==100 and pre['normal_decodes']==0 and pre['all100_max_abs']<=1e-6
assert sha(a.preflight/'starts.pt')==pre['starts_sha256'];starts=torch.load(a.preflight/'starts.pt',weights_only=True,map_location='cpu');assert sha(a.preflight/'lifts.pt')==pre['lifts_sha256'];lifts=torch.load(a.preflight/'lifts.pt',weights_only=True,map_location='cpu')
for name,h in pre['source_binding'].items():assert sha(name)==h,name
allowed={str((root/r[k]).resolve()) for r in split for root,k in [(a.low_root,'low'),(a.normal_root,'normal')]};opened=[];original=o.Image.open
def validation_only(path,*args,**kwargs):
    name=str(Path(path).resolve());assert name in allowed;opened.append(dict(path=name,utc=utc()));return original(path,*args,**kwargs)
o.Image.open=validation_only;a.out.mkdir(parents=True,exist_ok=False)
write(a.out/'config.json',dict(label=LABEL,created_utc=utc(),preflight_sha256=sha(a.preflight/'preflight.json'),preflight_completed_utc=pre['completed_utc'],split_sha256=SPLIT_SHA,optimizer='fresh Adam at exact T047 winner; gain raw and lift only',parameter_groups=dict(gain_raw_lr=.05,lift_lr=.01),updates_per_image=500,starts_per_image=1,objective='full RGB MSE float64 accumulation through frozen float32 renderer plus lift',lift_bounds=[-.2,.2],ev_gamma_frozen=True,insertion='common gain -> additive lift -> accepted identity contrast arithmetic -> clamp -> hard gate compositing',selection='earliest strict minimum over steps0..500',gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,source_binding=pre['source_binding']))
rows=[]
for i,(r,b) in enumerate(zip(split,pre['rows'])):
    begin=time.perf_counter();assert r['low']==b['low'] and thash(starts[i])==b['raw_sha256']
    for name,h in b['prior_files'].items():assert sha(a.accepted/f'{i:03d}'/name)==h
    assert sha(a.low_root/r['low'])==r['low_sha256'] and sha(a.normal_root/r['normal'])==r['normal_sha256']
    low=native(a.low_root/r['low']).cuda();assert thash(lifts[i])==b['lift_sha256'];model=Affine(low,starts[i].cuda(),b['gate'],b['box'],lifts[i].cuda())
    reference=native(a.normal_root/r['normal']).cuda();result=optimize(model,reference,steps=500)
    assert result['updates']==500 and torch.equal(result['lift'][0],lifts[i]) and torch.equal(result['gain_raw'][0],starts[i,:,2:3])
    assert torch.equal(model.ev_gamma.cpu(),starts[i,:,:2]) and torch.count_nonzero(result['lift'][:,~model.active.cpu()])==0
    with torch.no_grad():output=model().cpu()
    assert torch.isfinite(output).all() and 0<=output.min()<=output.max()<=1
    dest=a.out/f'{i:03d}';dest.mkdir();torch.save(dict(lift=result['lift'],gain_raw=result['gain_raw'],mse=result['mse']),dest/'history.pt');torch.save(dict(label=LABEL,image=output,raw=model.raw().detach().cpu(),lift=model.b.detach().cpu()),dest/'output.pt')
    step=result['best_step'];row=dict(index=i,low=r['low'],best_step=step,initial_mse=float(result['mse'][0]),best_mse=float(result['mse'][step]),final_mse=float(result['mse'][-1]),updates=500,seconds=time.perf_counter()-begin,raw_sha256=thash(model.raw()),lift_sha256=thash(model.b),gain_raw_sha256=thash(model.gain_raw),output_sha256=thash(output),files={n:sha(dest/n) for n in ['history.pt','output.pt']})
    rows.append(row);write(a.out/'progress.json',dict(completed=i+1,utc=utc(),last=row));print(f'{i+1}/100 best_step={row["best_step"]} seconds={row["seconds"]:.2f}',flush=True)
assert len(opened)==200 and len(rows)==100
write(a.out/'freeze.json',dict(label=LABEL,completed_utc=utc(),rows=rows,images=100,starts=100,updates=50000,states_per_image=501,opened=opened,preflight_sha256=sha(a.preflight/'preflight.json'),config_sha256=sha(a.out/'config.json'),official_test_access=False))
print('FROZEN100 selected outputs; no PSNR/SSIM aggregates computed',flush=True)
