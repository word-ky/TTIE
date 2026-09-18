"""Target-free stage: inherited E batch gradient and 80 separate fresh Adam updates."""
import argparse,json,torch
from PIL import Image
from ttie.energy_model import load_energy
from ttie.common_gain import CommonRegion2
from research_log.T059E.core import nested_split,selected_rows
from research_log.T059S.core import *
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=False)
Image.open=lambda *a,**k:(_ for _ in ()).throw(AssertionError('NO_REFERENCE_BEFORE_FREEZE'))
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
source=json.loads(Path('research_log/T059S/source_binding.json').read_bytes())
for n,h in source.items():assert sha(n)==h,n
assert sha(E/'head.pt')==HEAD
assert sha(E/'split.json')=='d44f86b7d1920f8f5efc386888ded4c65f543717165b030978f40ad595956621'
assert sha(B/'training_manifest.json')=='92fd60d01ca0780880d724464c4d0a76ba1721ab5b8a2d054d4035a141673125'
banks=json.loads((B/'training_manifest.json').read_bytes());split=nested_split(banks);selected=split['row_indices']['heldout'];wanted=set(selected)
saved_split=json.loads((E/'split.json').read_bytes());assert saved_split['row_indices']==split['row_indices']
records=[split['canonical'][i] for i in selected];anchors=[(i,r) for i,r in enumerate(records) if r['state_index']==0]
assert len(selected)==1529 and len(anchors)==80 and len({r['bank_index'] for _,r in anchors})==80 and len({r['image_id'] for _,r in anchors})==16
files={str(E/'head.pt'):HEAD,str(E/'split.json'):sha(E/'split.json'),str(B/'training_manifest.json'):sha(B/'training_manifest.json')};features=[]
for bi in split['bank_indices']['heldout']:
    entry=banks[bi];directory=B/entry['directory']
    for n,h in entry['files'].items():assert sha(directory/n)==h['sha256'];files[str(directory/n)]=h['sha256']
    features.append(torch.load(directory/'bank.pt',weights_only=True,map_location='cpu')['features'])
x=torch.cat(features);head=load_energy(E/'head.pt').eval().requires_grad_(False)
leaf=x.detach().cpu().float().requires_grad_(True);q,=torch.autograd.grad(head(leaf).sum(),leaf)
archive=json.loads(Path('research_log/T059A_archives.json').read_bytes());jm_path=JR/'artifacts/T059A/manifest.json'
assert sha(jm_path)==archive['files']['artifacts/T059A/manifest.json'];files[str(jm_path)]=sha(jm_path);jm=json.loads(jm_path.read_bytes());values={};reads=[]
for n,h in sorted(archive['files'].items()):
    if n.startswith('artifacts/T059A/jacobian_') and n.endswith('.pt'):
        subset,receipt=selected_rows(JR/n,wanted,'J',(28,64));values.update(subset)
        if receipt['ranges']:reads.append(dict(**receipt,accepted_full_chunk_sha256=h))
assert set(values)==wanted
for i,v in values.items():assert thash(v)==jm['rows'][i]['J_sha256']
jac=torch.stack([values[i] for i in selected]);g=torch.einsum('bfi,bf->bi',jac,q)
assert torch.isfinite(q).all() and torch.isfinite(g).all()
save_tensor(out/'field.pt',dict(x=x,q=q.detach(),J=jac,g=g.detach(),global_indices=torch.tensor(selected)))
actions=[]
for pos,r in anchors:
    bi=r['bank_index'];entry=banks[bi];directory=B/entry['directory'];bound=jm['rows'][r['index']]
    saved=torch.load(directory/'bank.pt',weights_only=True,map_location='cpu');images=torch.load(directory/'bank_images.pt',weights_only=True,map_location='cpu');decision=json.loads((directory/'bank_decisions.json').read_bytes())
    active=torch.tensor(decision['gate']['active'],dtype=torch.bool,device='cuda');legacy=CommonRegion2(active).cuda().requires_grad_(False)
    raw=probe_raw(saved['states'][0],active.cpu(),1.)
    with torch.no_grad():legacy.raw.copy_(raw.cuda());y0=legacy(images[0].cuda());grid=legacy.physical_grid()[:,:2]
    model=Bridge(y0,active,grid);before={n:thash(t) for n,t in model.state_dict().items()}
    assert before==bound['state_before'] and thash(raw)==bound['raw_sha256']
    optimizer=step(model,g[pos]);assert optimizer.state[model.v]['step']==1
    with torch.no_grad():c1=model.coefficient();y1=model()
    tensors=dict(q=q[pos].detach(),g=g[pos].detach(),v1=model.v.detach().cpu(),c1=c1.cpu(),y0=y0.cpu(),y1=y1.cpu(),mask=model.mask.cpu(),detail=model.detail.cpu())
    assert all(torch.isfinite(v).all() for v in tensors.values())
    mask=model.mask.expand_as(y0);assert torch.equal(y1[~mask],y0[~mask])
    name=f"bank_{bi:03d}.pt";save_tensor(out/name,tensors)
    actions.append(dict(**r,position=pos,file=name,sha256=sha(out/name),tensor_hashes={k:thash(v) for k,v in tensors.items()},active_regions=int(active.sum()),state_before=before,detail_action_steps=1,optimizer_defaults=optimizer.defaults,persisted_utc=utc()))
atomic_json(out/'actions.json',actions)
for n,h in files.items():assert sha(n)==h,n
for n,h in source.items():assert sha(n)==h,n
atomic_json(out/'action_freeze.json',dict(utc=utc(),actions=80,detail_action_steps=80,head_sha256=HEAD,files={n:sha(out/n) for n in ['field.pt','actions.json']+[r['file'] for r in actions]},input_hashes_before=files,input_hashes_after=files,jacobian_ranges=reads,source_bindings=source,clean_or_reference_reads_before_action_freeze=0,device=torch.cuda.get_device_name(),gradient_device='CPU, identical full1529-row batch to accepted E detail_statistics',eligibility='After freeze: inherited reference-gradient L2 norm >1e-12; all80 actions executed irrespective of eligibility',relative_convention='(mse1-mse0)/mse0; exactly unchanged zero-MSE banks assigned0 and excluded by zero reference gradient; PSNR uses max(MSE,1e-12) for description only'))
print(json.dumps(dict(status='ALL80_ACTIONS_FROZEN',utc=utc(),head=HEAD)),flush=True)
