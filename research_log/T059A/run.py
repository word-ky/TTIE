"""All canonical source-bank probes, no source-target image/label access."""
from research_log.T058A_tangent.core import *
from research_log.T059A.support import atomic_json,save_chunk,reopen_cache,verify_stage_a,detail_jacobian,reconstruct,compare,continuity
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
        if len(selection)<7346:selection.append(dict(index=len(selection),bank_index=bi,state_index=j,image_id=entry['image_id']))
assert len(selection)==7346
frozen,accepted,stage_a_before=verify_stage_a(banks)
assert len(frozen)==len(accepted)==7346
refs=json.loads(Path('research_log/T058AC_rows.json').read_bytes());assert len(refs)==16 and all(r['status']=='credible' for r in refs)
initial=model_hashes();a.out.mkdir(parents=True,exist_ok=False)
atomic_json(a.out/'selection.json',dict(canonical_selection_sha256=sha(a.canonical_selection),historical_T058A_selection_sha256=historical['selection_sha256'],preflight_sha256=historical['preflight_sha256'],rows=selection))
rows=[];checks=[];chunks=[];pending=[];pending_indices=[];error=None;failed_index=None;cached_bank=None;bank_hashes=[]
def checkpoint():
    if pending:
        chunks.append(save_chunk(a.out,pending_indices.copy(),pending.copy()));pending.clear();pending_indices.clear()
    atomic_json(a.out/'manifest.json',dict(label='SOURCE_DETAIL_FEATURE_JACOBIAN_CACHE_ONLY',complete=False,range=[0,7345],completed_rows=len(rows),rows=rows,chunks=chunks,selection_sha256=sha(a.out/'selection.json'),updated_utc=utc()))
checkpoint()
try:
    for spec in selection:
        index=spec['index'];failed_index=index
        bound=frozen[index];assert all(bound[k]==spec[k] for k in ['index','bank_index','state_index','image_id'])
        if index==16:assert len(checks)==16 and all(c['passed'] for c in checks),'FIRST16_CONTINUITY_NOT_PASSED'
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
        assert before==bound['state_before']==bound['state_after'] and legacy_before==bound['legacy_before']==bound['legacy_after']
        assert thash(raw)==bound['raw_sha256'] and thash(y0)==bound['y0_sha256']
        assert decision['names'][spec['state_index']]==bound['state_name'] and entry['condition']==bound['condition']
        value,output,phi=energy(model,obj,head)
        assert torch.is_grad_enabled() and phi.requires_grad and torch.equal(output,y0) and torch.count_nonzero(model.v)==0
        jac=detail_jacobian(phi,model.v);q,recon=reconstruct(head,phi,jac)
        assert all(bool(torch.isfinite(t).all()) for t in [value,output,phi,jac,q,recon]),'NONFINITE_JACOBIAN'
        assert jac.shape==(28,64) and jac.dtype==torch.float32
        old=accepted[index];assert thash(old)==bound['gradient_sha256']
        result=compare(recon,old)
        cache={k:t.detach().cpu() for k,t in dict(phi=phi,J=jac,q=q,reconstructed=recon,accepted=old).items()}
        if not result['allclose'] or not result['zero_status_matches']:
            torch.save(cache,a.out/'failure_case.pt');atomic_json(a.out/'failed_chain.json',dict(index=index,**result));raise AssertionError('FIXED_CHAIN_RULE_FAILURE')
        if index<16:
            ref=refs[index];assert all(ref[k]==spec[k] for k in ['index','bank_index','state_index','image_id'])
            info=ref['full_gradient']['gradients']['g32'];gref=torch.tensor(info['values'],dtype=torch.float32).reshape(info['shape']);assert thash(gref)==info['sha256']
            check=continuity(recon,gref);check.update(index=index,accepted_sha256=thash(old),reconstructed_sha256=thash(recon),reference_sha256=thash(gref));checks.append(check);atomic_json(a.out/'continuity.json',checks)
            assert check['passed'],'FIRST16_CONTINUITY_FAILURE'
        mask=model.mask.expand_as(output);assert torch.equal(output[~mask],model.y0[~mask])
        after={n:thash(t) for n,t in model.state_dict().items()};legacy_after={n:thash(t) for n,t in legacy.state_dict().items()}
        assert before==after and legacy_before==legacy_after and model.v.grad is None and torch.equal(legacy.raw.cpu(),raw)
        rows.append(dict(**spec,state_name=bound['state_name'],condition=entry['condition'],bank_directory=entry['directory'],chain=result,finite=True,J_shape=list(jac.shape),phi_shape=list(phi.shape),dtype=str(jac.dtype),active_regions=int(obj.active.sum()),active_mask_sha256=thash(model.mask),state_before=before,state_after=after,legacy_before=legacy_before,legacy_after=legacy_after,raw_sha256=thash(raw),y0_sha256=thash(y0),**{k+'_sha256':thash(t) for k,t in cache.items()}))
        pending.append(cache);pending_indices.append(index)
        if len(pending)==64 or index==15 or index==7345:
            checkpoint();print(json.dumps(dict(committed_rows=len(rows),last_index=index)),flush=True)
    assert len(rows)==7346 and len(checks)==16 and all(c['passed'] for c in checks)
    final=model_hashes();assert initial==final and all(p.grad is None for m in [head,scorer] for p in m.parameters())
    for n,h in binding.items():assert sha(n)==h
    for n,h in historical['files'].items():assert sha(a.stopped_run/n)==h
    for bank in bank_hashes:
        for n,h in bank['files'].items():assert sha(a.bank_root/bank['directory']/n)==h['sha256']
    assert sha(a.checkpoint)==ENERGY and sha(model_id['path'])==model_id['sha256'] and sha(a.prototypes)==proto['sha256']
    again,gradients,stage_a_after=verify_stage_a(banks);assert again==frozen and all(torch.equal(x,y) for x,y in zip(accepted,gradients))
    assert {k:v for k,v in stage_a_before.items() if k!='verified_utc'}=={k:v for k,v in stage_a_after.items() if k!='verified_utc'}
    verification=reopen_cache(a.out,7346);atomic_json(a.out/'reopen.json',verification)
    summary=dict(rows=7346,max_abs=max(r['chain']['max_abs'] for r in rows),max_l2=max(r['chain']['l2'] for r in rows),min_nonzero_cosine=min(r['chain']['cosine'] for r in rows if r['chain']['cosine'] is not None),zero_rows=sum(r['chain']['energy_zero'] for r in rows),all_chain_pass=True,all_zero_status_match=True,first16_pass=16,chunks=len(chunks))
    atomic_json(a.out/'summary.json',summary)
    receipt=dict(classification='T059-A cache written; independent reopen pending',completed_utc=utc(),rows=7346,summary=summary,stage_a_before=stage_a_before,stage_a_after=stage_a_after,manifest_sha256=sha(a.out/'manifest.json'),model_hashes_before=initial,model_hashes_after=final,source_bindings=binding,source_bank_bindings=bank_hashes,source_clean_opens=0,reference_gradient_access=0,optimizer_updates=0,persistent_scientific_state_changes=0,target_domain_access=0,lolv2_image_access=0,official_test_access=0,fd_evaluations=0,seconds=time.perf_counter()-start,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
    atomic_json(a.out/'receipt.json',receipt)
    print(json.dumps(summary),flush=True)
except Exception as exc:
    error=type(exc).__name__+': '+str(exc)
    (a.out/'error.txt').write_text(traceback.format_exc())
    atomic_json(a.out/'failure.json',dict(classification='T059-A detail Jacobian cache incomplete',failed_index=failed_index,error=error,computed_rows=len(rows),committed_rows=sum(x['count'] for x in chunks),complete_marker_exists=(a.out/'complete.json').exists(),seconds=time.perf_counter()-start))
    raise
