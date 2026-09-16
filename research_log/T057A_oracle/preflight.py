"""Freeze all100 low-only T055 identities before any normal decode."""
from core import *
import argparse,time
p=argparse.ArgumentParser()
for k in ['split','low-root','accepted','prior-preflight','binding','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();start=time.perf_counter();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
binding=json.loads(a.binding.read_bytes())
for name,h in binding.items():assert sha(name)==h,name
assert sha(a.split)==SPLIT_SHA and sha(a.accepted/'freeze.json')==PRIOR_FREEZE and sha(a.prior_preflight/'preflight.json')==PRIOR_PREFLIGHT
prior=json.loads((a.prior_preflight/'preflight.json').read_bytes());f=json.loads((a.accepted/'freeze.json').read_bytes());split=json.loads(a.split.read_bytes())['selected']
assert len(split)==len(f['rows'])==len(prior['rows'])==100 and sha(a.accepted/'config.json')==f['config_sha256']
config=json.loads((a.accepted/'config.json').read_bytes())
assert sha(a.accepted/'config.json')=='04172bb832de9a6530974d2430ee9bcd155a833f878142f5898056e76a0f1c94'
allowed={str((a.low_root/r['low']).resolve()) for r in split};opened=[];original=o.Image.open
def low_only(path,*args,**kwargs):
    name=str(Path(path).resolve());assert name in allowed;opened.append(name);return original(path,*args,**kwargs)
o.Image.open=low_only;a.out.mkdir(parents=True,exist_ok=False);rows=[];starts=[];lifts=[];qs=[];us=[];bs=[];vs=[]
for i,(r,old,b) in enumerate(zip(split,f['rows'],prior['rows'])):
    assert old['index']==i and r['low']==old['low']==b['low'];root=a.accepted/f'{i:03d}'
    for name,h in old['files'].items():assert sha(root/name)==h,name
    saved=torch.load(root/'output.pt',weights_only=True,map_location='cpu');hist=torch.load(root/'history.pt',weights_only=True,map_location='cpu');step=int(hist['mse'].argmin())
    assert step==old['best_step'] and thash(saved['u'])==old['u_sha256'] and thash(saved['b'])==old['b_sha256'] and thash(saved['q'])==old['q_sha256'] and thash(saved['raw'])==b['raw_sha256']==old['raw_sha256'] and thash(saved['lift'])==b['lift_sha256']==old['lift_sha256'] and thash(saved['image'])==old['output_sha256']
    assert sha(a.low_root/r['low'])==r['low_sha256'];low=native(a.low_root/r['low']).cuda();model=Chroma(low,saved['raw'].cuda(),b['gate'],b['box'],saved['lift'].cuda(),saved['q'].cuda(),saved['u'].cuda(),saved['b'].cuda(),saved['v'].cuda())
    assert torch.equal(saved['v'],hist['v'][step]) and thash(saved['v'])==old['v_sha256']
    for name in ['raw','lift','q','u','b','knots','ev','y0','detail']:
        assert torch.equal(getattr(model,name).cpu(),saved[name]) and thash(saved[name])==old[name+'_sha256']
    assert thash(model.mask)==thash(torch.tensor(b['gate']['active'],dtype=torch.bool).reshape(2,2).repeat_interleave(low.shape[-2]//2,0).repeat_interleave(low.shape[-1]//2,1))
    with torch.no_grad():output=model().cpu()
    assert torch.equal(model.c.cpu(),saved['c'])
    error=float((output-saved['image']).abs().max());assert error==0 and torch.equal(output,saved['image'])
    zero_sum=float(model.d_chroma.sum(dim=1).abs().max());assert zero_sum<=1e-6
    start_path=a.out/f'start_{i:03d}.pt';torch.save(output,start_path)
    rows.append(dict(index=i,low=r['low'],gate=b['gate'],box=b['box'],prior_files=old['files'],prior_best_step=step,raw_sha256=thash(saved['raw']),lift_sha256=thash(saved['lift']),q_sha256=thash(saved['q']),u_sha256=thash(saved['u']),b_sha256=thash(saved['b']),v_sha256=thash(saved['v']),mask_sha256=thash(model.mask),start_file=start_path.name,start_file_sha256=sha(start_path),zero_rgb_sum_max=zero_sum,d_chroma_sha256=thash(model.d_chroma),basis_sha256=thash(model.detail),y0_sha256=thash(model.y0),accepted_output_sha256=thash(saved['image']),reconstructed_output_sha256=thash(output),max_abs_error=error,bit_exact=torch.equal(output,saved['image'])))
    starts.append(saved['raw'].clone());lifts.append(saved['lift'].clone());qs.append(saved['q'].clone());us.append(saved['u'].clone());bs.append(saved['b'].clone());vs.append(saved['v'].clone())
assert len(opened)==100
torch.save(torch.stack(vs),a.out/'vs.pt');torch.save(torch.stack(bs),a.out/'bs.pt');torch.save(torch.stack(us),a.out/'us.pt');torch.save(torch.stack(starts),a.out/'starts.pt');torch.save(torch.stack(lifts),a.out/'lifts.pt');torch.save(torch.stack(qs),a.out/'qs.pt');write(a.out/'preflight.json',dict(label=LABEL,completed_utc=utc(),rows=rows,normal_decodes=0,opened_lows=opened,starts_sha256=sha(a.out/'starts.pt'),lifts_sha256=sha(a.out/'lifts.pt'),qs_sha256=sha(a.out/'qs.pt'),us_sha256=sha(a.out/'us.pt'),vs_sha256=sha(a.out/'vs.pt'),bs_sha256=sha(a.out/'bs.pt'),prior_freeze_sha256=PRIOR_FREEZE,prior_preflight_sha256=PRIOR_PREFLIGHT,split_sha256=SPLIT_SHA,source_binding=binding,source_binding_sha256=sha(a.binding),all100_zero_rgb_sum_max=max(r['zero_rgb_sum_max'] for r in rows),zero_rgb_sum_tolerance=1e-6,all100_max_abs=max(r['max_abs_error'] for r in rows),all100_bit_exact=all(r['bit_exact'] for r in rows),seconds=time.perf_counter()-start))
print('PREFLIGHT100 PASS',sha(a.out/'preflight.json'),flush=True)
