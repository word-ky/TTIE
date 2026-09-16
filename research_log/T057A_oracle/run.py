from core import *
import argparse,time
p=argparse.ArgumentParser()
for k in ['split','low-root','normal-root','accepted','preflight','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(7);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert sha(a.split)==SPLIT_SHA and sha(a.accepted/'freeze.json')==PRIOR_FREEZE
split=json.loads(a.split.read_bytes())['selected'];pre=json.loads((a.preflight/'preflight.json').read_bytes());assert len(pre['rows'])==100 and pre['normal_decodes']==0 and pre['all100_bit_exact']
for name,h in pre['source_binding'].items():assert sha(name)==h
allowed={str((root/r[k]).resolve()) for r in split for root,k in [(a.low_root,'low'),(a.normal_root,'normal')]};opened=[];original=o.Image.open
def validation_only(path,*args,**kwargs):
 name=str(Path(path).resolve());assert name in allowed;opened.append(dict(path=name,utc=utc()));return original(path,*args,**kwargs)
o.Image.open=validation_only;a.out.mkdir(parents=True,exist_ok=False)
write(a.out/'config.json',dict(label=LABEL,created_utc=utc(),preflight_sha256=sha(a.preflight/'preflight.json'),split_sha256=SPLIT_SHA,optimizer='fresh Adam; only w_c',lr=.05,updates_per_image=500,starts_per_image=1,grid=[8,8],coefficient_range=[-1,1],interpolation='raw w_c bilinear align_corners=False then tanh',B5=[1,4,6,4,1],B5_denominator=16,zero_rgb_sum_tolerance=1e-6,padding='reflect',operator='D_chroma=(y0-B5(y0))-mean_RGB(y0-B5(y0)); clamp(frozenT055+c_c*D_chroma) on active mask; inactive frozenT055',objective='full RGB float64 MSE through float32 renderer',selection='earliest strict minimum states0..500',gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,source_binding=pre['source_binding']))
rows=[]
for i,(item,b) in enumerate(zip(split,pre['rows'])):
 begin=time.perf_counter();assert item['low']==b['low'];root=a.accepted/f'{i:03d}'
 for name,h in b['prior_files'].items():assert sha(root/name)==h
 assert sha(a.preflight/b['start_file'])==b['start_file_sha256'];start=torch.load(a.preflight/b['start_file'],weights_only=True,map_location='cpu');v=torch.load(root/'output.pt',weights_only=True,map_location='cpu')
 assert sha(a.low_root/item['low'])==item['low_sha256'];low=native(a.low_root/item['low']).cuda();m=Chroma(low,v['raw'].cuda(),b['gate'],b['box'],v['lift'].cuda(),v['q'].cuda(),v['u'].cuda(),v['b'].cuda(),v['v'].cuda())
 assert torch.equal(m.one.cpu(),start) and torch.equal(start,v['image']) and thash(m.d_chroma)==b['d_chroma_sha256']
 assert float(m.d_chroma.sum(dim=1).abs().max())<=1e-6
 frozen={name:t.detach().cpu().clone() for name,t in m.named_buffers()}
 for name in OLD:assert torch.equal(frozen[name],v[name]),name
 assert sha(a.normal_root/item['normal'])==item['normal_sha256'];target=native(a.normal_root/item['normal']).cuda();hist=optimize(m,target,steps=500)
 assert hist['updates']==500 and torch.count_nonzero(hist['w_c'][0])==0
 for name,t in m.named_buffers():assert torch.equal(t.cpu(),frozen[name]),name
 with torch.no_grad():image=m().cpu();c_c=m.coefficient().cpu();controls=m.w_c.tanh().cpu()
 assert torch.isfinite(image).all() and torch.equal(image[:,:,~m.mask.cpu()],start[:,:,~m.mask.cpu()])
 saved={name:getattr(m,name).detach().cpu() for name in OLD+['one','d_chroma','w_c']};saved.update(image=image,c_c=c_c,c_c_controls=controls,label=LABEL)
 dest=a.out/f'{i:03d}';dest.mkdir();torch.save(saved,dest/'output.pt');torch.save(dict(w_c=hist['w_c'],mse=hist['mse']),dest/'history.pt')
 row=dict(index=i,low=item['low'],best_step=hist['best_step'],updates=500,seconds=time.perf_counter()-begin,hashes={name:thash(value) for name,value in saved.items() if isinstance(value,torch.Tensor)},files={name:sha(dest/name) for name in ['output.pt','history.pt']});rows.append(row);write(a.out/'progress.json',dict(completed=i+1,utc=utc(),last=row));print(f'{i+1}/100 best_step={row["best_step"]} seconds={row["seconds"]:.2f}',flush=True)
assert len(opened)==200
write(a.out/'freeze.json',dict(label=LABEL,completed_utc=utc(),rows=rows,images=100,updates=50000,states_per_image=501,opened=opened,preflight_sha256=sha(a.preflight/'preflight.json'),config_sha256=sha(a.out/'config.json'),official_test_access=False));print('FROZEN100; no metrics computed')
