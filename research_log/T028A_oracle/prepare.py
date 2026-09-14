"""Bind/reconstruct all100 accepted states before task reference deployment."""
from core import *

p=argparse.ArgumentParser(description=LABEL)
for name in ['split','low-root','accepted','out']:p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();assert sha(a.split)==SPLIT_SHA
split=json.loads(a.split.read_bytes())['selected'];assert len(split)==100
freeze=json.loads((a.accepted/'freeze.json').read_bytes())
allowed={str((a.low_root/r['low']).resolve()) for r in split}
opened=[];original_open=Image.open
def low_only(path,*args,**kw):
    resolved=str(Path(path).resolve());assert resolved in allowed
    opened.append(resolved);return original_open(path,*args,**kw)
Image.open=low_only
torch.set_num_threads(1);torch.manual_seed(7)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert torch.cuda.is_available();a.out.mkdir(parents=True,exist_ok=False)
rows=[];starts=[]
for i,r in enumerate(split):
    d=a.accepted/f'{i:03d}';f=freeze['rows'][i];assert f['low']==r['low']
    hashes={n:sha(d/n) for n in ['decision.json','output.pt','trajectory.pt']}
    assert all(hashes[n]==f['files'][n]['sha256'] for n in hashes)
    assert sha(a.low_root/r['low'])==r['low_sha256']
    decision=json.loads((d/'decision.json').read_bytes())
    saved=torch.load(d/'output.pt',map_location='cpu',weights_only=True)
    low=native(a.low_root/r['low']).cuda();model,box=frozen_model(decision,low)
    with torch.no_grad():
        model.raw.copy_(saved['raw'].cuda());out=model(low)
        error=float((out.cpu()-saved['image']).abs().max())
        grid_error=float((model.physical_grid()[:,:2].cpu()-saved['grid']).abs().max())
    assert error==0 and grid_error==0
    assert torch.isfinite(saved['raw']).all()
    starts.append(torch.stack([torch.zeros_like(saved['raw']),saved['raw']]))
    rows.append(dict(index=i,low=r['low'],files=hashes,low_sha256=r['low_sha256'],
        gate=decision['gate'],box=decision['diagnostics']['action_box'],reproduction_max_abs=error,grid_max_abs=grid_error))
torch.save(dict(label=LABEL,starts=torch.stack(starts)),a.out/'bound_starts.pt')
assert len(opened)==100 and set(opened)==allowed
write(a.out/'preflight.json',dict(label=LABEL,completed_utc=utc(),pairs=100,
    accepted_source='b2359721c89db732d17e03be273e0bdb71bb377a',
    accepted_merge='51f84a9d96880bca8e908c9b3cdff496d7277e39',
    accepted_freeze_sha256=sha(a.accepted/'freeze.json'),accepted_config_sha256=sha(a.accepted/'config.json'),
    accepted_metrics_sha256=sha(a.accepted/'metrics.csv'),split_sha256=sha(a.split),
    starts_sha256=sha(a.out/'bound_starts.pt'),all100_exact_reconstruction=True,
    opened_lows=opened,normal_images_opened=0,rows=rows,gpu=torch.cuda.get_device_name()))
print('PREFLIGHT PASS:100 exact reconstructions and both starts bound before reference deployment',flush=True)
