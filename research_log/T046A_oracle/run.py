"""REFERENCE_ORACLE_ONLY: one fresh Adam at T035 winner, exactly1000updates."""
from core import *
import argparse,time
p=argparse.ArgumentParser()
for k in ['split','low-root','normal-root','accepted','preflight','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(7);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert sha(a.split)==SPLIT_SHA and sha(a.accepted/'freeze.json')==PRIOR_FREEZE
split=json.loads(a.split.read_bytes())['selected'];pre=json.loads((a.preflight/'preflight.json').read_bytes());assert len(pre['rows'])==100 and pre['normal_decodes']==0 and pre['all100_max_abs']<=1e-6
assert sha(a.preflight/'starts.pt')==pre['starts_sha256'];starts=torch.load(a.preflight/'starts.pt',weights_only=True,map_location='cpu')
for name,h in pre['source_binding'].items():assert sha(name)==h,name
allowed={str((root/r[k]).resolve()) for r in split for root,k in [(a.low_root,'low'),(a.normal_root,'normal')]};opened=[];original=o.Image.open
def validation_only(path,*args,**kwargs):
    name=str(Path(path).resolve());assert name in allowed;opened.append(dict(path=name,utc=utc()));return original(path,*args,**kwargs)
o.Image.open=validation_only;a.out.mkdir(parents=True,exist_ok=False)
write(a.out/'config.json',dict(label=LABEL,created_utc=utc(),preflight_sha256=sha(a.preflight/'preflight.json'),preflight_completed_utc=pre['completed_utc'],split_sha256=SPLIT_SHA,optimizer='fresh Adam at exact T035 winner; no inherited moments',lr=.05,updates_per_image=1000,starts_per_image=1,objective='full RGB MSE float64 accumulation through unchanged float32 renderer',selection='earliest strict minimum over steps0..1000',gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,source_binding=pre['source_binding']))
rows=[]
for i,(r,b) in enumerate(zip(split,pre['rows'])):
    begin=time.perf_counter();assert r['low']==b['low'] and thash(starts[i])==b['raw_sha256']
    for name,h in b['prior_files'].items():assert sha(a.accepted/f'{i:03d}'/name)==h
    assert sha(a.low_root/r['low'])==r['low_sha256'] and sha(a.normal_root/r['normal'])==r['normal_sha256']
    low=native(a.low_root/r['low']).cuda();decision=dict(gate=b['gate'],diagnostics=dict(action_box=b['box']));model,box=frozen_model(decision,low)
    reference=native(a.normal_root/r['normal']).cuda();result=optimize_start(model,box,low,reference,starts[i].cuda(),steps=1000)
    assert result['updates']==1000 and torch.equal(result['raw_history'][0],starts[i])
    grid=physical(result['raw_history'][:,0]);active=torch.tensor(b['gate']['active']).reshape(2,2)
    assert (grid[:,2:5]>=.5).all() and (grid[:,2:5]<=2).all() and torch.equal(grid[:,2:5][:,:,~active],torch.ones_like(grid[:,2:5][:,:,~active]))
    with torch.no_grad():output=model(low).cpu();selected_grid=model.physical_grid().cpu()
    assert torch.isfinite(output).all() and 0<=output.min()<=output.max()<=1
    dest=a.out/f'{i:03d}';dest.mkdir();torch.save(dict(raw=result['raw_history'],mse=torch.tensor(result['history'],dtype=torch.float64)),dest/'history.pt');torch.save(dict(label=LABEL,image=output,raw=result['best_raw'],grid=selected_grid),dest/'output.pt')
    row=dict(index=i,low=r['low'],best_step=result['best_step'],initial_mse=result['initial_mse'],best_mse=result['best_mse'],final_mse=result['final_mse'],updates=1000,seconds=time.perf_counter()-begin,raw_sha256=thash(result['best_raw']),output_sha256=thash(output),files={n:sha(dest/n) for n in ['history.pt','output.pt']})
    rows.append(row);write(a.out/'progress.json',dict(completed=i+1,utc=utc(),last=row));print(f'{i+1}/100 best_step={row["best_step"]} seconds={row["seconds"]:.2f}',flush=True)
assert len(opened)==200 and len(rows)==100
write(a.out/'freeze.json',dict(label=LABEL,completed_utc=utc(),rows=rows,images=100,starts=100,updates=100000,states_per_image=1001,opened=opened,preflight_sha256=sha(a.preflight/'preflight.json'),config_sha256=sha(a.out/'config.json'),official_test_access=False))
print('FROZEN100 selected outputs; no PSNR/SSIM aggregates computed',flush=True)
