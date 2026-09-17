"""All canonical source-bank probes, no source-target image/label access."""
from research_log.T058A_tangent.core import *
from research_log.T058Y.numerics import boundary
from research_log.T058AA.decomposition import decompose,EXPLAINED,UNRESOLVED
import copy
from research_log.T058Z.shadow import shadow_energy
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
assert sha(a.out/'selection.json')=='4e18e346478421ebbbd192fdbe6c5a861cdc0a45761ecf7a3becb7afeec079c2'
prior_root=Path('/home/wenchang/asdasdsad/wjq/TTIE/runs/20260917-074144-ttie-t058z-cast')
prior_files=json.loads(Path('research_log/T058Z_archives.json').read_bytes())['files']
for n,h in prior_files.items():assert sha(prior_root/n)==h
historical_z=json.loads((prior_root/'artifacts/T058Z/states.json').read_bytes())[0]
assert historical_z['index']==3
rows=[dict(**x,status='not_run',E32=None,d32_rev=None,E64=None,d64_rev=None,criteria=None,ladder=None,boundary=None) for x in selection if x['index']==3]
scorer64=copy.deepcopy(scorer).cpu().double();head64=copy.deepcopy(head).cpu().double()
def shadow_hashes():return {prefix+'.'+n:thash(v) for prefix,m in [('scorer',scorer64),('head',head64)] for n,v in m.state_dict().items()}
shadow_initial=shadow_hashes()
for original,shadow in [(scorer,scorer64),(head,head64)]:
    for n,v in original.state_dict().items():
        got=shadow.state_dict()[n];assert torch.equal(v.cpu().to(got.dtype),got)
        if got.is_floating_point():assert got.dtype==torch.float64 and got.device.type=='cpu'
error=None;hash_receipts=[];execution_order=[3]
for index in execution_order:
    row=rows[0];entry=banks[row['bank_index']];directory=a.bank_root/entry['directory']
    saved=torch.load(directory/'bank.pt',map_location='cpu',weights_only=True);images=torch.load(directory/'bank_images.pt',map_location='cpu',weights_only=True);decision=json.loads((directory/'bank_decisions.json').read_bytes())
    low=images[0].cuda();obj=objective(scorer,decision['gate'],receipt['frozen_gate_receipt']['calibration'],'cuda:0')
    legacy=CommonRegion2(obj.active).cuda().requires_grad_(False);raw=probe_raw(saved['states'][row['state_index']],obj.active.cpu(),1.)
    with torch.no_grad():legacy.raw.copy_(raw.cuda());y0=legacy(low);grid=legacy.physical_grid()[:,:2]
    assert thash(y0)==pre['rows'][index]['y0_sha256'] and thash(raw)==pre['rows'][index]['raw_sha256']
    model=Detail(y0,obj.active,grid);before={n:thash(t) for n,t in model.state_dict().items()};legacy_before={n:thash(t) for n,t in legacy.state_dict().items()}
    direction=(torch.arange(64,device='cuda').reshape_as(model.v)%2*2-1).to(model.v)/8
    value=energy(model,obj,head)[0];g,=torch.autograd.grad(value,model.v)
    assert torch.isfinite(value) and torch.isfinite(g).all()
    row.update(status='float32_done',E32=float(value.detach()),d32_rev=float((g.double()*direction.double()).sum()),y0_sha256=thash(y0),historical_h001=old_fd[index] if index<len(old_fd) else None)
    model64=copy.deepcopy(model).cpu().double();direction64=direction.cpu().double();obj64=objective(scorer64,decision['gate'],receipt['frozen_gate_receipt']['calibration'],'cpu')
    shadow_before={n:thash(t) for n,t in model64.state_dict().items()}
    for n,v in model.state_dict().items():assert torch.equal(v.cpu().to(model64.state_dict()[n].dtype),model64.state_dict()[n])
    row['boundary']=boundary(model64,direction64)
    assert row['boundary']['boundary_directional_count']>0
    try:
        v64,y64,phi64,cast_proof=shadow_energy(model64,obj64,head64,verify=True)
        row['cast_isolation']=cast_proof
        row['observed_shadow_dtypes']=dict(energy=str(v64.dtype),output=str(y64.dtype),features=str(phi64.dtype))
        if phi64.dtype!=torch.float64 or y64.dtype!=torch.float64 or v64.dtype!=torch.float64:
            raise RuntimeError('Unchanged composition retains a non-float64 intermediate; full CPU float64 shadow would require a functional/cast-site rewrite')
        g64,=torch.autograd.grad(v64,model64.v)
        assert torch.isfinite(v64) and torch.isfinite(g64).all()
        row.update(E64=float(v64.detach()),d64_rev=float((g64*direction64).sum()))
        row['decomposition']=decompose(model64,obj64,head64,direction64,y64,row['d64_rev'],historical_z['d64_rev'])
        row['historical_T058Z_h0005']=historical_z['ladder'][-1]
        row['historical_T058Z_d64_rev']=historical_z['d64_rev']
        row['criteria']=row['decomposition']['criteria']
        if len(row['criteria'])==6 and all(c['passed'] for c in row['criteria'].values()):row['status']='explained'
        else:error='CLAMP_DECOMPOSITION_CRITERION_FAILURE';row['status']='unresolved'
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc);row.update(status='shadow_unavailable',shadow_error=error);(a.out/'shadow_error.txt').write_text(traceback.format_exc())
    after={n:thash(t) for n,t in model.state_dict().items()};legacy_after={n:thash(t) for n,t in legacy.state_dict().items()};shadow_after={n:thash(t) for n,t in model64.state_dict().items()}
    assert before==after and legacy_before==legacy_after and shadow_before==shadow_after and torch.count_nonzero(model.v)==0 and torch.count_nonzero(model64.v)==0
    hash_receipts.append(dict(index=index,before=before,after=after,legacy_before=legacy_before,legacy_after=legacy_after,shadow_before=shadow_before,shadow_after=shadow_after))
    write(a.out/'states.json',rows);print(json.dumps({k:row[k] for k in ['index','status','E32','d32_rev','E64','d64_rev']}),flush=True)
    if error is not None:break
after_models=model_hashes();shadow_final=shadow_hashes();assert initial==after_models and shadow_initial==shadow_final
assert all(p.grad is None for m in [scorer,head,scorer64,head64] for p in m.parameters())
for n,h in historical['files'].items():assert sha(a.stopped_run/n)==h
for n,h in prior_files.items():assert sha(prior_root/n)==h
for n,h in binding.items():assert sha(n)==h
assert sha(a.checkpoint)==ENERGY
verdict=EXPLAINED if error is None and all(r['status']=='explained' for r in rows) else UNRESOLVED
write(a.out/'receipt.json',dict(classification=verdict,completed_utc=utc(),attempted=sum(r['status']!='not_run' for r in rows),required=1,new_fd_evaluations=0,execution_order=execution_order,error=error,model_hashes_before=initial,model_hashes_after=after_models,shadow_hashes_before=shadow_initial,shadow_hashes_after=shadow_final,state_hashes=hash_receipts,shadow_device='cpu',shadow_parameter_dtype='torch.float64',shadow_nonpersistent=True,historical_files_unchanged=historical['files'],prior_verifier_files_unchanged=prior_files,source_bindings=binding,source_jpg_opens=0,stage_b_executions=0,target_domain_access=0,official_test_access=0,optimizer_updates=0,persistent_state_changes=0,scientific_source_changes=0,selection_sha256=sha(a.out/'selection.json'),states_sha256=sha(a.out/'states.json'),seconds=time.perf_counter()-start,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()))
print(verdict,flush=True)
