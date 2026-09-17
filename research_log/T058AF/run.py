"""One source-reference diagnostic; frozen learned gradients are read only."""
import argparse,json,time,traceback
from pathlib import Path
import torch
from PIL import Image
from ttie.natural import load_image
from ttie.common_gain import CommonRegion2
from research_log.T058A_tangent.core import Detail,setup,probe_raw,sha,thash,utc,alignment,MANIFEST,RECEIPT,BANK,ENERGY
from research_log.T058AF.support import *

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();setup();start=time.perf_counter();began=utc();a.out.mkdir(parents=True,exist_ok=False)
bank_root=ROOT/'runs/20260912-181047-ttie-t014-stage-a-repaired/artifacts/audit'
stopped=ROOT/'runs/20260917-023631-ttie-t058a-stage-a'
manifest_path=Path('research_log/T014_source_manifest.json');receipt_path=Path('research_log/T039A_bound_T014_receipt.json')
assert sha(manifest_path)==MANIFEST and sha(receipt_path)==RECEIPT and sha(bank_root/'training_manifest.json')==BANK
sources={r['image_id']:r for r in json.loads(manifest_path.read_bytes())['images'] if r['split']=='train_t014_sobolev'}
assert len(sources)==80
access=SourceOpens(ROOT/'shared/t008/val2017',sources);Image.open=access
rows=[];chunks=[];pending=[];indices=[];current_index=None
try:
    binding=json.loads(Path('research_log/T058AF_source_binding.json').read_bytes())
    for n,h in binding.items():assert sha(n)==h,n
    receipt=json.loads(receipt_path.read_bytes());banks=json.loads((bank_root/'training_manifest.json').read_bytes())
    assert banks==receipt['source_records_manifest'] and {r['image_id'] for r in banks}==set(sources)=={r['image_id'] for r in receipt['split_manifests']['train_t014_sobolev']}
    canonical=Path('research_log/T039A_result/stage_a/selection.json');assert sha(canonical)=='08227ee09f4cd428ee034d1d33cea7827037e853f2fe8b0a7efedcccfecc964c' and json.loads(canonical.read_bytes())['banks']==banks
    historical=json.loads(Path('research_log/T058A_failure_receipt.json').read_bytes())
    for n,h in historical['files'].items():assert sha(stopped/n)==h
    pre=json.loads((stopped/'artifacts/stage_a/preflight.json').read_bytes());assert len(pre['rows'])==7346
    assert sha(stopped/'artifacts/stage_a/selection.json')=='7e5062869f267164cc02a14de249ab0ec8e430bde88b0eeb2a9d28d4f3b7b06b'
    checkpoints={str(ROOT/'research_log/T014_energy.pt'):ENERGY,receipt['model_identity']['path']:receipt['model_identity']['sha256'],str(ROOT/'research_log/remote_runs/20260912-071126-ttie-t006-a6000/artifacts/audit/prototypes.pt'):receipt['frozen_gate_receipt']['prototype_identity']['sha256']}
    for n,h in checkpoints.items():assert sha(n)==h
    for bank in banks:
        for n,h in bank['files'].items():assert sha(bank_root/bank['directory']/n)==h['sha256']
    frozen,learned,verification=verify_stage_a(banks)
    assert len(frozen)==len(learned)==7346 and not access.opened
    atomic_json(a.out/'stage_a_verified.json',verification)
    access.verified_utc=verification['verified_utc']
    clean_cache={};cached=None
    for i,record in enumerate(frozen):
        current_index=i;assert record['index']==i
        bank=banks[record['bank_index']];directory=bank_root/bank['directory'];image_id=record['image_id'];assert image_id==bank['image_id']
        if cached!=record['bank_index']:
            saved=torch.load(directory/'bank.pt',map_location='cpu',weights_only=True);images=torch.load(directory/'bank_images.pt',map_location='cpu',weights_only=True);decision=json.loads((directory/'bank_decisions.json').read_bytes())
            assert decision['names'][0]=='identity';low=images[0].cuda();active=torch.tensor(decision['gate']['active'],dtype=torch.bool,device='cuda')
            legacy=CommonRegion2(active).cuda().requires_grad_(False);cached=record['bank_index']
        j=record['state_index'];assert record['state_name']==decision['names'][j] and record['condition']==bank['condition'] and record['bank_directory']==bank['directory']
        raw=probe_raw(saved['states'][j],active.cpu(),1.)
        with torch.no_grad():legacy.raw.copy_(raw.cuda());y0=legacy(low);grid=legacy.physical_grid()[:,:2]
        assert thash(raw)==record['raw_sha256']==pre['rows'][i]['raw_sha256'] and thash(y0)==record['y0_sha256']==pre['rows'][i]['y0_sha256']
        model=Detail(y0,active,grid);before={n:thash(t) for n,t in model.state_dict().items()};legacy_before={n:thash(t) for n,t in legacy.state_dict().items()}
        assert before==record['state_before']==record['state_after'] and legacy_before==record['legacy_before']==record['legacy_after']
        if image_id not in clean_cache:
            clean_cache[image_id]=load_image(ROOT/'shared/t008/val2017'/sources[image_id]['filename']).cuda()
            atomic_json(a.out/'source_opens.json',access.opened)
        clean=clean_cache[image_id];output=model();assert output.shape==clean.shape and torch.equal(output,y0)
        loss=reference_loss(output,clean);g_r,=torch.autograd.grad(loss,model.v)
        assert torch.isfinite(loss) and torch.isfinite(g_r).all(),'NONFINITE_REFERENCE'
        gradient=g_r.detach().cpu();g_e=learned[i]
        assert thash(g_e)==record['gradient_sha256']
        after={n:thash(t) for n,t in model.state_dict().items()};legacy_after={n:thash(t) for n,t in legacy.state_dict().items()}
        assert before==after and legacy_before==legacy_after and model.v.grad is None and all(p.grad is None for p in legacy.parameters())
        metric=alignment(g_e,gradient,torch.ones_like(g_e,dtype=torch.bool))
        rows.append(dict(index=i,bank_index=record['bank_index'],state_index=j,image_id=image_id,state_name=record['state_name'],condition=record['condition'],bank_directory=record['bank_directory'],energy_sha256=thash(g_e),reference_sha256=thash(gradient),reference_dtype=str(gradient.dtype),reference_shape=list(gradient.shape),reference_loss=float(loss.detach()),raw_sha256=thash(raw),y0_sha256=thash(y0),raw_y0_verified=True,state_before=before,state_after=after,legacy_before=legacy_before,legacy_after=legacy_after,**metric))
        pending.append(gradient);indices.append(i)
        if len(pending)==64 or i==7345:
            chunks.append(save_reference(a.out,indices,pending));pending=[];indices=[]
            atomic_json(a.out/'manifest.json',dict(label='SOURCE_REFERENCE_GRADIENT_DIAGNOSTIC_ONLY',rows=rows,chunks=chunks,completed_rows=len(rows)))
            print(json.dumps(dict(committed_rows=len(rows),source_opens=len(access.opened))),flush=True)
    assert len(rows)==7346 and len(access.opened)==80 and {r['image_id'] for r in access.opened}==set(sources)
    for n,h in binding.items():assert sha(n)==h
    for n,h in checkpoints.items():assert sha(n)==h
    for n,h in historical['files'].items():assert sha(stopped/n)==h
    for bank in banks:
        for n,h in bank['files'].items():assert sha(bank_root/bank['directory']/n)==h['sha256']
    again_rows,again_g,again_verification=verify_stage_a(banks)
    assert again_rows==frozen and all(torch.equal(x,y) for x,y in zip(learned,again_g))
    post={k:v for k,v in again_verification.items() if k!='verified_utc'};assert post=={k:v for k,v in verification.items() if k!='verified_utc'}
    readback=replay(a.out,learned,7346);atomic_json(a.out/'reopen.json',readback)
    result=summary(rows);atomic_json(a.out/'summary.json',result)
    final=dict(label='SOURCE_REFERENCE_GRADIENT_DIAGNOSTIC_ONLY',classification=result['classification'],started_utc=began,completed_utc=utc(),rows=7346,source_ids=80,source_manifest_sha256=MANIFEST,stage_a_verification=verification,stage_a_after_utc=again_verification['verified_utc'],stage_a_unchanged=True,source_bindings=binding,checkpoint_hashes_before=checkpoints,checkpoint_hashes_after=checkpoints,source_bank_manifest_sha256=BANK,source_bank_files_unchanged=True,opened_source_targets=access.opened,reference_gradient_executions=7346,learned_gradient_recomputations=0,optimizer_updates=0,persistent_scientific_state_changes=0,target_domain_access=0,lolv2_image_access=0,official_test_access=0,fd_evaluations=0,seconds=time.perf_counter()-start,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),files={n:sha(a.out/n) for n in ['stage_a_verified.json','source_opens.json','manifest.json','summary.json','reopen.json']})
    atomic_json(a.out/'receipt.json',final)
    atomic_json(a.out/'complete.json',dict(classification=result['classification'],rows=7346,receipt_sha256=sha(a.out/'receipt.json'),manifest_sha256=sha(a.out/'manifest.json'),summary_sha256=sha(a.out/'summary.json'),completed_utc=utc()))
    print(json.dumps(dict(classification=result['classification'],overall=result['overall'])),flush=True)
except Exception as exc:
    (a.out/'error.txt').write_text(traceback.format_exc())
    atomic_json(a.out/'failure.json',dict(error=type(exc).__name__+': '+str(exc),failed_index=current_index,completed_rows=len(rows),source_opens=access.opened,seconds=time.perf_counter()-start))
    raise
