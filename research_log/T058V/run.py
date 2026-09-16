"""All canonical source-bank probes, no source-target image/label access."""
from research_log.T058A_tangent.core import *
from research_log.T058V.numerics import STEPS,ulp,boundaries,crossings,agreement
import traceback
import argparse,time
from PIL import Image
from ttie.clip_signal import FrozenCLIP
from ttie.learned_prototypes import Prototypes
from ttie.semantic_ttt import SemanticScorer
from ttie.energy_model import load_energy
from ttie.common_gain import CommonRegion2

p=argparse.ArgumentParser()
for k in ['bank-root','receipt','manifest','checkpoint','prototypes','binding','canonical-selection','stopped-run','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();setup();start=time.perf_counter()
assert sha(a.receipt)==RECEIPT and sha(a.manifest)==MANIFEST and sha(a.bank_root/'training_manifest.json')==BANK and sha(a.checkpoint)==ENERGY
receipt=json.loads(a.receipt.read_bytes());manifest=json.loads(a.manifest.read_bytes());banks=json.loads((a.bank_root/'training_manifest.json').read_bytes())
assert banks==receipt['source_records_manifest']
ids={r['image_id'] for r in receipt['split_manifests']['train_t014_sobolev']};assert len(ids)==80 and {r['image_id'] for r in banks}==ids
binding=json.loads(a.binding.read_bytes())
for n,h in binding.items():assert sha(n)==h,n
def deny(*args,**kwargs):raise AssertionError('Stage A cannot open any source clean target or image file')
Image.open=deny
features=[]
for row in banks:
    for name,h in row['files'].items():assert sha(a.bank_root/row['directory']/name)==h['sha256']
    t=torch.load(a.bank_root/row['directory']/'bank.pt',map_location='cpu',weights_only=True);assert len(t['states'])==row['states'];features.append(t['features'])
all_features=torch.cat(features);assert all_features.shape==(7346,28)
head=load_energy(a.checkpoint).cuda();buffers=head.state_dict()
assert torch.equal(all_features.double().mean(0).float(),buffers['x_mean'].cpu())
std=all_features.double().std(0,unbiased=False);assert torch.equal(torch.where(std==0,torch.ones_like(std),std).float(),buffers['x_scale'].cpu())
model_id=receipt['model_identity'];proto=receipt['frozen_gate_receipt']['prototype_identity']
assert sha(model_id['path'])==model_id['sha256'] and sha(a.prototypes)==proto['sha256']
scorer=SemanticScorer(FrozenCLIP.from_checkpoint(model_id['path'],'cuda:0'),Prototypes(torch.load(a.prototypes,map_location='cuda:0',weights_only=True)['raw'])).eval().requires_grad_(False)
def model_hashes():return {prefix+'.'+n:thash(v) for prefix,m in [('scorer',scorer),('head',head)] for n,v in m.state_dict().items()}
assert sha(a.canonical_selection)=='08227ee09f4cd428ee034d1d33cea7827037e853f2fe8b0a7efedcccfecc964c'
canonical=json.loads(a.canonical_selection.read_bytes());assert canonical['banks']==banks
historical=json.loads(Path('research_log/T058A_failure_receipt.json').read_bytes())
for n,h in historical['files'].items():assert sha(a.stopped_run/n)==h
original_binding=json.loads(Path('research_log/T058A_source_binding.json').read_bytes())
for n,h in original_binding.items():assert sha(n)==h
pre=json.loads((a.stopped_run/'artifacts/stage_a/preflight.json').read_bytes());old_fd=json.loads((a.stopped_run/'artifacts/stage_a/finite_difference_energy.json').read_bytes())
selection=[]
for bi,entry in enumerate(banks):
    for j in range(entry['states']):
        if len(selection)<16:selection.append(dict(index=len(selection),bank_index=bi,state_index=j,image_id=entry['image_id']))
assert len(selection)==16
initial=model_hashes();a.out.mkdir(parents=True,exist_ok=False)
write(a.out/'selection.json',dict(canonical_selection_sha256=sha(a.canonical_selection),rows=selection))
rows=[dict(**x,status='not_run',d_rev=None,d_fwd=None,error=None,tolerance=None,ladder=None,boundaries=None) for x in selection]
error=None;model_hash_receipts=[]
for row in rows:
    entry=banks[row['bank_index']];directory=a.bank_root/entry['directory']
    saved=torch.load(directory/'bank.pt',map_location='cpu',weights_only=True);images=torch.load(directory/'bank_images.pt',map_location='cpu',weights_only=True)
    decision=json.loads((directory/'bank_decisions.json').read_bytes());low=images[0].cuda();obj=objective(scorer,decision['gate'],receipt['frozen_gate_receipt']['calibration'],'cuda:0')
    legacy_model=CommonRegion2(obj.active).cuda().requires_grad_(False);raw=probe_raw(saved['states'][row['state_index']],obj.active.cpu(),1.)
    with torch.no_grad():legacy_model.raw.copy_(raw.cuda());y0=legacy_model(low);grid=legacy_model.physical_grid()[:,:2]
    assert thash(y0)==pre['rows'][row['index']]['y0_sha256'] and thash(raw)==pre['rows'][row['index']]['raw_sha256']
    model=Detail(y0,obj.active,grid);before={n:thash(v) for n,v in model.state_dict().items()};legacy_before={n:thash(v) for n,v in legacy_model.state_dict().items()}
    direction=(torch.arange(64,device='cuda').reshape_as(model.v)%2*2-1).to(model.v)/8
    def fn(v):return energy(model,obj,head,v)[0]
    value=fn(model.v);g,=torch.autograd.grad(value,model.v);rev=float((g.double()*direction.double()).sum())
    assert torch.isfinite(g).all() and torch.isfinite(value)
    row.update(status='reverse_computed',d_rev=rev,energy=float(value.detach()),energy_ulp=ulp(float(value.detach())),tolerance=1e-6+1e-4*abs(rev),boundaries=boundaries(y0,model.mask),y0_sha256=thash(y0),original_h001_receipt=old_fd[row['index']] if row['index']<len(old_fd) else None)
    try:
        # Genuine forward AD. No reverse-gradient dot substitute or model fallback.
        primal,jvp=torch.func.jvp(fn,(model.v.detach(),),(direction,))
        fwd=float(jvp.detach());assert torch.isfinite(primal) and torch.isfinite(jvp)
        row.update(d_fwd=fwd,error=abs(rev-fwd),forward_primal=float(primal.detach()),forward_primal_abs_error=abs(float(primal.detach())-float(value.detach())),status='AD_agree' if agreement(rev,fwd) else 'AD_disagree')
        if not agreement(rev,fwd):error='REVERSE_FORWARD_AD_DISAGREEMENT'
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc);row.update(status='forward_jvp_unavailable',forward_error=error);(a.out/'forward_error.txt').write_text(traceback.format_exc())
    # Fixed numerical explanations are evaluated only if forward AD was available
    # and agrees; an unsupported graph or disagreement is an immediate hard stop.
    if error is None:
        row['ladder']=[]
        for h in STEPS:
            with torch.no_grad():plus=float(fn(h*direction));minus=float(fn(-h*direction))
            assert np.isfinite(plus) and np.isfinite(minus)
            up,um=ulp(plus),ulp(minus);gap=plus-minus
            row['ladder'].append(dict(h=h,plus=plus,minus=minus,central=gap/(2*h),abs_error_vs_rev=abs(gap/(2*h)-rev),plus_ulp=up,minus_ulp=um,gap=gap,gap_in_plus_ulps=gap/up,gap_in_minus_ulps=gap/um,plus_clamp=crossings(model,h*direction),minus_clamp=crossings(model,-h*direction)))
    after={n:thash(v) for n,v in model.state_dict().items()};legacy_after={n:thash(v) for n,v in legacy_model.state_dict().items()}
    assert before==after and legacy_before==legacy_after and torch.count_nonzero(model.v)==0 and model.v.grad is None
    model_hash_receipts.append(dict(index=row['index'],before=before,after=after,legacy_before=legacy_before,legacy_after=legacy_after))
    write(a.out/'states.json',rows);print(json.dumps({k:row[k] for k in ['index','status','d_rev','d_fwd','error']}),flush=True)
    if error is not None:break
after_models=model_hashes();assert after_models==initial and all(p.grad is None for m in [head,scorer] for p in m.parameters())
for n,h in historical['files'].items():assert sha(a.stopped_run/n)==h
for n,h in binding.items():assert sha(n)==h
assert sha(a.checkpoint)==ENERGY
verdict='T058 derivative AD convention numerically validated' if error is None and all(r['status']=='AD_agree' for r in rows) else 'T058 derivative verifier unresolved'
write(a.out/'receipt.json',dict(classification=verdict,completed_utc=utc(),attempted=sum(r['status']!='not_run' for r in rows),required=16,error=error,model_hashes_before=initial,model_hashes_after=after_models,state_hashes=model_hash_receipts,historical_files_unchanged=historical['files'],source_bindings=binding,source_jpg_opens=0,stage_b_executions=0,target_domain_access=0,official_test_access=0,optimizer_updates=0,persistent_state_changes=0,scientific_source_changes=0,selection_sha256=sha(a.out/'selection.json'),states_sha256=sha(a.out/'states.json'),seconds=time.perf_counter()-start,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()))
print(verdict,flush=True)
