"""Image-only online construction. Does not open bank.pt, decisions, or any cached field/J."""
import argparse,torch
from PIL import Image
from types import SimpleNamespace
from ttie.clip_signal import FrozenCLIP
from ttie.learned_prototypes import Prototypes
from ttie.semantic_ttt import SemanticScorer,FixedObjective
from ttie.common_gain import CommonRegion2
from ttie.energy_model import load_energy,features
from research_log.T059A.support import detail_jacobian
from research_log.T059Y.core import *
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);out=p.parse_args().out;out.mkdir(parents=True,exist_ok=False)
started=utc()
torch.set_num_threads(1);torch.manual_seed(7);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
binding=json.loads(Path('research_log/T059Y/source_binding.json').read_bytes());selection=json.loads(Path('research_log/T059Y/cohort.json').read_bytes());inputs=selection['inputs']
allowed=set(inputs)|{str(Path(n).resolve()) for n in binding}|{str(Path('research_log/T059Y/source_binding.json').resolve())}
install_firewall(ROOT,allowed,out)
validate_inputs(binding);validate_inputs(inputs)
sf=json.loads(Path('research_log/T059Y/cohort_freeze.json').read_bytes());assert sha('research_log/T059Y/cohort.json')==sf['sha256'];assert sf['utc']<utc()
config=json.loads((B/'config.json').read_bytes());gate_receipt=config['frozen_receipt'];proto=ROOT/'research_log/remote_runs/20260912-071126-ttie-t006-a6000/artifacts/audit/prototypes.pt'
encoder=FrozenCLIP.from_checkpoint(config['model_identity']['path'],'cuda:0');scorer=SemanticScorer(encoder,Prototypes(torch.load(proto,map_location='cuda:0',weights_only=True)['raw'])).eval().requires_grad_(False)
head=load_energy(E/'head.pt').eval().requires_grad_(False);head_before={k:thash(v) for k,v in head.state_dict().items()};records=[]
for r in selection['anchors']:
    base=torch.load(r['base_file'],map_location='cpu',weights_only=True);trajectory=torch.load(r['trajectory_file'],map_location='cpu',weights_only=True)
    verify_state(base,trajectory,r['selected_step']);assert all(thash(base[k])==r['state_hashes'][k] for k in ['image','raw','grid'])
    gate=r['gate'];obj=SimpleNamespace(active=torch.tensor(gate['active'],device='cuda:0',dtype=torch.bool),winner=torch.tensor(gate['winner'],device='cuda:0',dtype=torch.long),evidence=torch.tensor(gate['evidence'],device='cuda:0',dtype=torch.float64),calibration=selection['calibration'])
    y0=base['image'].cuda();grid=base['grid'].cuda();before_grid=thash(grid);before_raw=thash(base['raw'])
    model=Bridge(y0,obj.active,grid);x=features(obj,scorer(model()),model.grid);J=detail_jacobian(x,model.v).detach().cpu()
    leaf=x.detach().cpu().float().requires_grad_(True);q,=torch.autograd.grad(head(leaf).sum(),leaf);g=torch.einsum('fi,f->i',J,q).detach()
    optimizer=step(model,g)
    assert thash(grid)==before_grid and thash(base['raw'])==before_raw
    with torch.no_grad():y1=model();c1=model.coefficient()
    tensors=dict(global_grid=grid.cpu(),global_raw=base['raw'],x=x.detach().cpu(),J=J,q=q.detach(),g=g,v1=model.v.detach().cpu(),c1=c1.cpu(),y0=y0.cpu(),y1=y1.cpu(),mask=model.mask.cpu())
    assert all(torch.isfinite(t).all() for t in tensors.values());name=f"online_{r['index']:03d}.pt";save_tensor(out/name,tensors)
    acted=bool(torch.count_nonzero(g)) and not torch.equal(y0,y1)
    if not bool(obj.active.any()) or not bool(torch.count_nonzero(g)):assert torch.equal(y0,y1)
    records.append(dict(**r,norm=float(g.double().norm()),acted=acted,optimizer_steps=1,file=name,sha256=sha(out/name),tensor_hashes={k:thash(v) for k,v in tensors.items()},active_regions=int(obj.active.sum()),persisted_utc=utc()))
    print(json.dumps(dict(completed=len(records),index=r['index'])),flush=True)
assert len(records)==100 and head_before=={k:thash(v) for k,v in head.state_dict().items()}
assert all(p.grad is None for p in scorer.parameters())
for n,h in inputs.items():assert sha(n)==h
for n,h in binding.items():assert sha(n)==h
atomic_json(out/'online_records.json',records);atomic_json(out/'online_freeze.json',dict(utc=utc(),inference_started_utc=started,rows=100,files={n:sha(out/n) for n in ['online_records.json']+[r['file'] for r in records]},input_hashes_before=inputs,input_hashes_after=inputs,source_bindings=binding,head_state_before=head_before,head_state_after=head_before,normalization=head.normalization(),cached_matched_detail_x_J_q_g_action_reads=0,clean_reference_reads=0,reference_gradient_reads=0,bound_development_low_images=100,decoded_low_images=0,development_normal_reads=0,official_test_access=0,inference_reference_leakage=0,optimizer_steps=100,online_feature_J_evaluations=100,gate_image_forwards=0,persisted_T026_gates=100,physical_gpu=1,device=torch.cuda.get_device_name()))
print('ONLINE100_FROZEN',flush=True)
