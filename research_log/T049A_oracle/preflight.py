"""Freeze all100 low-only T048 identities before any normal decode."""
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
for name,h in config['source_binding'].items():assert sha(name)==h,name
allowed={str((a.low_root/r['low']).resolve()) for r in split};opened=[];original=o.Image.open
def low_only(path,*args,**kwargs):
    name=str(Path(path).resolve());assert name in allowed;opened.append(name);return original(path,*args,**kwargs)
o.Image.open=low_only;a.out.mkdir(parents=True,exist_ok=False);rows=[];starts=[];lifts=[]
for i,(r,old,b) in enumerate(zip(split,f['rows'],prior['rows'])):
    assert old['index']==i and r['low']==old['low']==b['low'];root=a.accepted/f'{i:03d}'
    for name,h in old['files'].items():assert sha(root/name)==h,name
    saved=torch.load(root/'output.pt',weights_only=True,map_location='cpu');hist=torch.load(root/'history.pt',weights_only=True,map_location='cpu');step=int(hist['mse'].argmin())
    assert step==old['best_step'] and torch.equal(saved['lift'],hist['lift'][step]) and torch.equal(saved['raw'][:,:2],torch.load(a.prior_preflight/'starts.pt',weights_only=True,map_location='cpu')[i,:,:2]) and torch.equal(saved['raw'][:,2:3],hist['gain_raw'][step]) and thash(saved['raw'])==old['raw_sha256'] and thash(saved['image'])==old['output_sha256']
    assert sha(a.low_root/r['low'])==r['low_sha256'];low=native(a.low_root/r['low']).cuda();model=Tone(low,saved['raw'].cuda(),b['gate'],b['box'],saved['lift'].cuda())
    with torch.no_grad():output=model().cpu()
    error=float((output-saved['image']).abs().max());assert error<=1e-6
    rows.append(dict(index=i,low=r['low'],gate=b['gate'],box=b['box'],prior_files=old['files'],prior_best_step=step,raw_sha256=thash(saved['raw']),lift_sha256=thash(saved['lift']),accepted_output_sha256=thash(saved['image']),reconstructed_output_sha256=thash(output),max_abs_error=error,bit_exact=torch.equal(output,saved['image'])))
    starts.append(saved['raw'].clone());lifts.append(saved['lift'].clone())
assert len(opened)==100
torch.save(torch.stack(starts),a.out/'starts.pt');torch.save(torch.stack(lifts),a.out/'lifts.pt');write(a.out/'preflight.json',dict(label=LABEL,completed_utc=utc(),rows=rows,normal_decodes=0,opened_lows=opened,starts_sha256=sha(a.out/'starts.pt'),lifts_sha256=sha(a.out/'lifts.pt'),prior_freeze_sha256=PRIOR_FREEZE,prior_preflight_sha256=PRIOR_PREFLIGHT,split_sha256=SPLIT_SHA,source_binding=binding,source_binding_sha256=sha(a.binding),all100_max_abs=max(r['max_abs_error'] for r in rows),all100_bit_exact=all(r['bit_exact'] for r in rows),seconds=time.perf_counter()-start))
print('PREFLIGHT100 PASS',sha(a.out/'preflight.json'),flush=True)
