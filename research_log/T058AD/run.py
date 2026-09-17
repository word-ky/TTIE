"""All canonical source-bank probes, no source-target image/label access."""
from research_log.T058A_tangent.core import *
from research_log.T058AD.storage import atomic_json,save_chunk,reopen,continuity
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
pre=json.loads((a.stopped_run/'artifacts/stage_a/preflight.json').read_bytes())
assert len(pre['rows'])==7346 and sha(a.stopped_run/'artifacts/stage_a/selection.json')=='7e5062869f267164cc02a14de249ab0ec8e430bde88b0eeb2a9d28d4f3b7b06b'
selection=[]
for bi,entry in enumerate(banks):
    for j in range(entry['states']):
        if len(selection)<1024:selection.append(dict(index=len(selection),bank_index=bi,state_index=j,image_id=entry['image_id']))
assert len(selection)==1024
prior_root=Path('/home/wenchang/asdasdsad/wjq/TTIE/runs/20260917-105829-ttie-t058ac-copy')
prior_files=json.loads(Path('research_log/T058AC_archives.json').read_bytes())['files']
for n,h in prior_files.items():assert sha(prior_root/n)==h
refs=json.loads((prior_root/'artifacts/T058AC/states.json').read_bytes())
assert len(refs)==16 and all(x['status']=='credible' for x in refs)
assert sha(prior_root/'artifacts/T058AC/selection.json')=='4e18e346478421ebbbd192fdbe6c5a861cdc0a45761ecf7a3becb7afeec079c2'
initial=model_hashes();a.out.mkdir(parents=True,exist_ok=False)
atomic_json(a.out/'selection.json',dict(canonical_selection_sha256=sha(a.canonical_selection),historical_T058A_selection_sha256=historical['selection_sha256'],preflight_sha256=historical['preflight_sha256'],rows=selection))
rows=[];checks=[];chunks=[];pending=[];pending_indices=[];error=None;failed_index=None;cached_bank=None;bank_hashes=[]
def checkpoint():
    if pending:
        chunks.append(save_chunk(a.out,pending_indices.copy(),pending.copy()));pending.clear();pending_indices.clear()
    atomic_json(a.out/'manifest.json',dict(label='STAGE_A_LEARNED_ENERGY_GRADIENTS_ONLY',complete=False,range=[0,1023],completed_rows=len(rows),rows=rows,chunks=chunks,selection_sha256=sha(a.out/'selection.json'),updated_utc=utc()))
checkpoint()
try:
    for spec in selection:
        index=spec['index'];failed_index=index
        if index==16:assert len(checks)==16 and all(x['passed'] for x in checks),'FIRST16_CONTINUITY_NOT_PASSED'
        entry=banks[spec['bank_index']];directory=a.bank_root/entry['directory']
        if cached_bank!=spec['bank_index']:
            if cached_bank is not None:
                assert legacy_initial=={n:thash(t) for n,t in legacy.state_dict().items() if n!='raw'}
            saved=torch.load(directory/'bank.pt',map_location='cpu',weights_only=True);images=torch.load(directory/'bank_images.pt',map_location='cpu',weights_only=True);decision=json.loads((directory/'bank_decisions.json').read_bytes())
            assert decision['names'][0]=='identity';low=images[0].cuda();obj=objective(scorer,decision['gate'],receipt['frozen_gate_receipt']['calibration'],'cuda:0')
            legacy=CommonRegion2(obj.active).cuda().requires_grad_(False);legacy_initial={n:thash(t) for n,t in legacy.state_dict().items() if n!='raw'};cached_bank=spec['bank_index']
            bank_hashes.append(dict(bank_index=cached_bank,directory=entry['directory'],files=entry['files']))
        raw=probe_raw(saved['states'][spec['state_index']],obj.active.cpu(),1.)
        with torch.no_grad():legacy.raw.copy_(raw.cuda());y0=legacy(low);grid=legacy.physical_grid()[:,:2]
        assert thash(y0)==pre['rows'][index]['y0_sha256'] and thash(raw)==pre['rows'][index]['raw_sha256']
        model=Detail(y0,obj.active,grid);before={n:thash(t) for n,t in model.state_dict().items()};legacy_before={n:thash(t) for n,t in legacy.state_dict().items()}
        value,output,phi=energy(model,obj,head);g_e,=torch.autograd.grad(value,model.v)
        assert all(bool(torch.isfinite(t).all()) for t in [value,output,phi,g_e]),'NONFINITE_SCIENTIFIC_VALUE'
        assert torch.equal(output,y0) and torch.count_nonzero(model.v)==0
        mask=model.mask.expand_as(output);assert torch.equal(output[~mask],model.y0[~mask])
        if not bool(obj.active.any()):assert torch.count_nonzero(g_e)==0
        after={n:thash(t) for n,t in model.state_dict().items()};legacy_after={n:thash(t) for n,t in legacy.state_dict().items()}
        assert before==after and legacy_before==legacy_after and model.v.grad is None and torch.equal(legacy.raw.cpu(),raw)
        if index<16:
            ref=refs[index];assert all(ref[k]==spec[k] for k in ['index','bank_index','state_index','image_id'])
            tensor_info=ref['full_gradient']['gradients']['g32'];g_ref=torch.tensor(tensor_info['values'],dtype=torch.float32).reshape(tensor_info['shape']);assert thash(g_ref)==tensor_info['sha256']
            check=continuity(g_e,g_ref);check.update(index=index,current_sha256=thash(g_e),reference_sha256=thash(g_ref));checks.append(check);atomic_json(a.out/'continuity.json',checks)
            assert check['passed'],'FIRST16_CONTINUITY_FAILURE'
        gradient=g_e.detach().cpu()
        rows.append(dict(**spec,state_name=decision['names'][spec['state_index']],condition=entry['condition'],bank_directory=entry['directory'],energy=float(value.detach()),gradient_norm=float(gradient.double().norm()),gradient_sha256=thash(gradient),gradient_dtype=str(gradient.dtype),gradient_shape=list(gradient.shape),finite=True,active_regions=int(obj.active.sum()),inactive_regions=4-int(obj.active.sum()),active_rgb=int(mask.sum()),inactive_rgb=int((~mask).sum()),inactive_output_changes=0,state_before=before,state_after=after,legacy_before=legacy_before,legacy_after=legacy_after,raw_sha256=thash(raw),y0_sha256=thash(y0)))
        pending.append(gradient);pending_indices.append(index)
        if len(pending)==64 or index==15 or index==1023:
            checkpoint();print(json.dumps(dict(committed_rows=len(rows),last_index=index,continuity_passed=sum(x['passed'] for x in checks))),flush=True)
    assert len(rows)==1024 and len(checks)==16 and all(x['passed'] for x in checks)
    final=model_hashes();assert initial==final and all(p.grad is None for m in [head,scorer] for p in m.parameters())
    for n,h in binding.items():assert sha(n)==h
    for n,h in historical['files'].items():assert sha(a.stopped_run/n)==h
    for n,h in prior_files.items():assert sha(prior_root/n)==h
    for bank in bank_hashes:
        for n,h in bank['files'].items():assert sha(a.bank_root/bank['directory']/n)==h['sha256']
    assert sha(a.checkpoint)==ENERGY and sha(model_id['path'])==model_id['sha256'] and sha(a.prototypes)==proto['sha256']
    verification=reopen(a.out,1024)
    atomic_json(a.out/'reopen.json',verification)
    frozen=dict(classification='T058-AD Stage-A shard 0 frozen',completed_utc=utc(),range=[0,1023],processed_once_in_order=True,rows=1024,continuity_passed=16,continuity_sha256=sha(a.out/'continuity.json'),manifest_sha256=sha(a.out/'manifest.json'),selection_sha256=sha(a.out/'selection.json'),independent_reopen=verification,model_hashes_before=initial,model_hashes_after=final,source_bindings=binding,historical_files_unchanged=historical['files'],prior_verifier_files_unchanged=prior_files,source_bank_bindings=bank_hashes,source_jpg_opens=0,source_clean_opens=0,reference_gradient_executions=0,stage_b_executions=0,target_domain_access=0,official_test_access=0,optimizer_updates=0,persistent_scientific_state_changes=0,new_fd_evaluations=0,seconds=time.perf_counter()-start,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
    atomic_json(a.out/'receipt.json',frozen)
    atomic_json(a.out/'complete.json',dict(classification=frozen['classification'],rows=1024,range=[0,1023],receipt_sha256=sha(a.out/'receipt.json'),manifest_sha256=sha(a.out/'manifest.json'),completed_utc=utc()))
    print(frozen['classification'],flush=True)
except Exception as exc:
    error=type(exc).__name__+': '+str(exc)
    (a.out/'error.txt').write_text(traceback.format_exc())
    atomic_json(a.out/'failure.json',dict(classification='T058-AD Stage-A shard incomplete',failed_index=failed_index,error=error,computed_rows=len(rows),committed_rows=sum(x['count'] for x in chunks),continuity=checks,complete_marker_exists=(a.out/'complete.json').exists(),seconds=time.perf_counter()-start))
    raise
