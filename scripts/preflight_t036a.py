"""Low-only accepted-source binding and actual T026 smoke reproduction."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import torch
from PIL import Image
from scripts.run_t036a import sha,write,utc,initialize,models
from ttie.lolv2_gamma_core import native_rgb,low_image_opener
from ttie.semantic_ttt import Region2
from ttie.common_gain import CommonRegion2,CommonBox
from ttie.gamma_range_ttt import trajectory

p=argparse.ArgumentParser()
for k in ['accepted','low-root','split','assets','binding','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();initialize();binding=json.loads(a.binding.read_bytes())
for n,h in binding.items():assert sha(n)==h,n
assets=json.loads(a.assets.read_bytes());accepted_config=json.loads((a.accepted/'config.json').read_bytes())
assert assets['files']==accepted_config['assets']['files']
split=json.loads(a.split.read_bytes());assert sha(a.split)=='b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b'
opened=[];allowed={str((a.low_root/r['low']).resolve()) for r in split['selected']};Image.open=low_image_opener(allowed,opened)
scorer,gate,head=models(assets);rows=[];repro=None
for i,r in enumerate(split['selected']):
    low=native_rgb(a.low_root/r['low']).cuda();d=a.accepted/f'{i:03d}'
    decision=json.loads((d/'decision.json').read_bytes());saved=torch.load(d/'output.pt',weights_only=True,map_location='cuda:0')
    obj=SimpleNamespace(active=torch.tensor(decision['gate']['active'],device='cuda',dtype=torch.bool),winner=torch.tensor(decision['gate']['winner'],device='cuda'))
    old=Region2(obj.active).cuda();new=CommonRegion2(obj.active).cuda();box=CommonBox(obj,2)
    errors=[]
    for raw in [torch.zeros_like(saved['raw']),saved['raw']]:
        with torch.no_grad():old.raw.copy_(raw);new.raw[:,:2].copy_(raw);errors.append(float((old(low)-new(low)).abs().max()))
    with torch.no_grad():
        errors.append(float((new(low)-saved['image']).abs().max()))
        new.raw[:,2:].fill_(.8);box(new)
        inactive=~obj.active.reshape(2,2)
        assert torch.equal(new.physical_grid()[:,2:5,:, :][:,:,inactive],torch.ones_like(new.physical_grid()[:,2:5,:, :][:,:,inactive]))
    assert max(errors)<=1e-6
    if i==0:
        result,t,selection=trajectory(low,scorer,gate,head,basis='region2',max_steps=40)
        repro=float((result['image']-saved['image'].cpu()).abs().max())
        assert repro<=1e-6 and selection['selected_step']==decision['selection']['selected_step']
        del result,t
    rows.append(dict(index=i,low=r['low'],renderer_errors=errors,accepted_output_sha256=sha(d/'output.pt')))
assert len(opened)==100
a.out.parent.mkdir(parents=True,exist_ok=True)
write(a.out,dict(completed_utc=utc(),rows=rows,source_binding=binding,assets_sha256=sha(a.assets),accepted_config_sha256=sha(a.accepted/'config.json'),
    identity_renderer_max_abs=max(max(r['renderer_errors']) for r in rows),baseline_reproduction_max_abs=repro,baseline_smoke_index=0,
    inactive_gain_identity=True,normal_decodes=0,opened_lows=opened,adaptation_api='image, scorer, gate receipt, frozen energy, basis, max_steps; no reference path'))
print('Preflight100identity/inactive + actualbaseline40step reproduction PASS',flush=True)
