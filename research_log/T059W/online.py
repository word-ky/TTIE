"""Image-only online construction. Does not open bank.pt, decisions, or any cached field/J."""
import argparse,torch
from PIL import Image
from ttie.clip_signal import FrozenCLIP
from ttie.learned_prototypes import Prototypes
from ttie.semantic_ttt import SemanticScorer,FixedObjective
from ttie.common_gain import CommonRegion2
from ttie.energy_model import load_energy,features
from research_log.T059A.support import detail_jacobian
from research_log.T059W.core import *
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);out=p.parse_args().out;out.mkdir(parents=True,exist_ok=False)
Image.open=lambda *a,**k:(_ for _ in ()).throw(AssertionError('NO_IMAGE_FILE_REFERENCE_ACCESS'))
torch.set_num_threads(1);torch.manual_seed(7);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
binding=json.loads(Path('research_log/T059W/source_binding.json').read_bytes());selection=json.loads(Path('research_log/T059W/selection.json').read_bytes());inputs=selection['inputs']
for n,h in binding.items():assert sha(n)==h
for n,h in inputs.items():assert sha(n)==h
sf=json.loads(Path('research_log/T059W/selection_freeze.json').read_bytes());assert sha('research_log/T059W/selection.json')==sf['sha256'];assert sf['utc']<utc()
config=json.loads((B/'config.json').read_bytes());gate_receipt=config['frozen_receipt'];proto=ROOT/'research_log/remote_runs/20260912-071126-ttie-t006-a6000/artifacts/audit/prototypes.pt'
encoder=FrozenCLIP.from_checkpoint(config['model_identity']['path'],'cuda:0');scorer=SemanticScorer(encoder,Prototypes(torch.load(proto,map_location='cuda:0',weights_only=True)['raw'])).eval().requires_grad_(False)
head=load_energy(E/'head.pt').eval().requires_grad_(False);head_before={k:thash(v) for k,v in head.state_dict().items()};records=[]
for r in selection['anchors']:
    low=torch.load(r['image_file'],map_location='cpu',weights_only=True)[0].cuda()
    obj=FixedObjective(scorer,low,gate_receipt);legacy=CommonRegion2(obj.active).cuda().requires_grad_(False)
    assert torch.count_nonzero(legacy.raw)==0
    with torch.no_grad():y0=legacy(low);grid=legacy.physical_grid()[:,:2]
    model=Bridge(y0,obj.active,grid);x=features(obj,scorer(model()),model.grid);J=detail_jacobian(x,model.v).detach().cpu()
    leaf=x.detach().cpu().float().requires_grad_(True);q,=torch.autograd.grad(head(leaf).sum(),leaf);g=torch.einsum('fi,f->i',J,q).detach()
    optimizer=step(model,g)
    with torch.no_grad():y1=model();c1=model.coefficient()
    tensors=dict(x=x.detach().cpu(),J=J,q=q.detach(),g=g,v1=model.v.detach().cpu(),c1=c1.cpu(),y0=y0.cpu(),y1=y1.cpu(),mask=model.mask.cpu())
    assert all(torch.isfinite(t).all() for t in tensors.values());name=f"online_{r['bank_index']:03d}.pt";save_tensor(out/name,tensors)
    records.append(dict(**r,file=name,sha256=sha(out/name),tensor_hashes={k:thash(v) for k,v in tensors.items()},active_regions=int(obj.active.sum()),persisted_utc=utc()))
    print(json.dumps(dict(completed=len(records),bank=r['bank_index'])),flush=True)
assert len(records)==16 and head_before=={k:thash(v) for k,v in head.state_dict().items()}
assert all(p.grad is None for p in scorer.parameters())
for n,h in inputs.items():assert sha(n)==h
for n,h in binding.items():assert sha(n)==h
atomic_json(out/'online_records.json',records);atomic_json(out/'online_freeze.json',dict(utc=utc(),rows=16,files={n:sha(out/n) for n in ['online_records.json']+[r['file'] for r in records]},input_hashes_before=inputs,input_hashes_after=inputs,source_bindings=binding,head_state_before=head_before,head_state_after=head_before,normalization=head.normalization(),cached_x_J_q_g_action_reads=0,clean_reference_reads=0,reference_gradient_reads=0,target_domain_access=0,lolv2_access=0,official_test_access=0,inference_reference_leakage=0,optimizer_steps=16,online_feature_J_evaluations=16,gate_image_forwards=16,physical_gpu=1,device=torch.cuda.get_device_name()))
print('ONLINE16_FROZEN',flush=True)
