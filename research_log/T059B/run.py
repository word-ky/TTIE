"""Source cached-data-only dual-tangent fit feasibility; no image access."""
import argparse,json,time,traceback,math
from pathlib import Path
import torch
from PIL import Image
from ttie.energy_model import load_energy,save_energy,RECIPE
from ttie.sobolev_train import training_statistics
from research_log.T058A_tangent.core import sha,thash,utc,RECEIPT,BANK,ENERGY
from research_log.T059A.support import atomic_json,reopen_cache
from research_log.T059B.fit import train_fixed,detail_statistics,verdict
ROOT=Path('/home/wenchang/asdasdsad/wjq/TTIE')
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();start=time.perf_counter();a.out.mkdir(parents=True,exist_ok=False);training_started=False
Image.open=lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('NO_NEW_IMAGE_ACCESS'))
torch.set_num_threads(1);torch.manual_seed(7)
assets={}
def bind(path,h):
    path=Path(path);assert sha(path)==h,str(path);assets[str(path)]=h
try:
    source=json.loads(Path('research_log/T059B_source_binding.json').read_bytes())
    for n,h in source.items():bind(n,h)
    bank_root=ROOT/'runs/20260912-181047-ttie-t014-stage-a-repaired/artifacts/audit'
    bind('research_log/T039A_bound_T014_receipt.json',RECEIPT);receipt=json.loads(Path('research_log/T039A_bound_T014_receipt.json').read_bytes())
    bind(bank_root/'training_manifest.json',BANK);banks=json.loads((bank_root/'training_manifest.json').read_bytes());assert banks==receipt['source_records_manifest']
    train_path=bank_root/'training.json';bind(train_path,'9daf1b722d97236fd357afee295a7efe7d148ef9d5adaa4c6b53512c037258b2');old_training=json.loads(train_path.read_bytes());assert old_training['recipe']==RECIPE and RECIPE['device']=='cpu'
    checkpoint=ROOT/'research_log/T014_energy.pt';bind(checkpoint,ENERGY);frozen=load_energy(checkpoint);frozen_hash={n:thash(v) for n,v in frozen.state_dict().items()}
    xs=[];ys=[];records=[];canonical=[]
    for bi,b in enumerate(banks):
        directory=bank_root/b['directory']
        for group in ['files','source_supervision_files']:
            for name,v in b[group].items():bind(directory/name,v['sha256'])
        bank=torch.load(directory/'bank.pt',map_location='cpu',weights_only=True);target=json.loads((directory/'targets.json').read_bytes());deriv=torch.load(directory/'source_derivatives.pt',map_location='cpu',weights_only=True)
        xs.append(bank['features']);ys.append(torch.tensor([r['mse'] for r in target],dtype=torch.float64));records.append(deriv)
        for j in range(b['states']):canonical.append(dict(index=len(canonical),bank_index=bi,state_index=j,image_id=b['image_id']))
    x=torch.cat(xs);mse=torch.cat(ys);legacy={k:torch.cat([r[k] for r in records]) for k in records[0]};assert x.shape==(7346,28) and legacy['jacobian'].shape==(7346,28,8)
    assert torch.equal(x.double().mean(0).float(),frozen.x_mean) and frozen.normalization()==old_training['normalization']
    cache_run=ROOT/'runs/20260917-165444-ttie-t059a-jacobian';cache=cache_run/'artifacts/T059A'
    af_run=ROOT/'runs/20260917-153308-ttie-t058af-stageb';af=af_run/'artifacts/T058AF'
    for run,name in [(cache_run,'T059A'),(af_run,'T058AF')]:
        archive=json.loads(Path('research_log/'+name+'_archives.json').read_bytes())
        for n,h in archive['files'].items():bind(run/n,h)
        marker=json.loads((run/'artifacts'/name/'complete.json').read_bytes());assert sha(run/'artifacts'/name/'complete.json')==sha('research_log/'+name+'_complete.json')
        for n,k in [('manifest.json','manifest_sha256'),('receipt.json','receipt_sha256')]:assert sha(run/'artifacts'/name/n)==marker[k]
    cache_readback=reopen_cache(cache,7346)
    cm=json.loads((cache/'manifest.json').read_bytes());am=json.loads((af/'manifest.json').read_bytes());jd=[];phi=[];accepted=[];gr=[]
    for i,(c,r,spec) in enumerate(zip(cm['rows'],am['rows'],canonical)):
        assert all(c[k]==r[k]==spec[k] for k in spec)
        assert c['raw_sha256']==r['raw_sha256'] and c['y0_sha256']==r['y0_sha256'] and c['accepted_sha256']==r['energy_sha256']
    for c in cm['chunks']:
        saved=torch.load(cache/c['file'],map_location='cpu',weights_only=True)
        jd.append(saved['J']);phi.append(saved['phi']);accepted.append(saved['accepted'])
    for c in am['chunks']:
        saved=torch.load(af/c['file'],map_location='cpu',weights_only=True)
        for i,g in zip(saved['indices'],saved['g_R']):assert i==len(gr) and thash(g)==am['rows'][i]['reference_sha256'];gr.append(g)
    jd=torch.cat(jd);phi=torch.cat(phi);truth=torch.stack(gr).flatten(1);mask=torch.tensor([not r['degenerate'] for r in am['rows']],dtype=torch.bool)
    assert jd.shape==(7346,28,64) and truth.shape==(7346,64) and torch.equal(mask,truth.double().norm(dim=1)>1e-12)
    feature_error=float((x-phi).abs().max());assert torch.allclose(x,phi,atol=1e-6,rtol=1e-6),'CACHED_FEATURE_PATH_MISMATCH'
    detail=dict(jacobian=jd,reference_gradient=truth,direction_mask=mask)
    for t in [x,mse,jd,truth,legacy['jacobian'],legacy['reference_gradient']]:assert torch.isfinite(t).all()
    baseline_legacy=training_statistics(frozen,x,mse,legacy);baseline_detail=detail_statistics(frozen,x,jd,truth,mask)
    target_legacy=old_training['final_train_statistics']['sobolev_primary'];target_detail=json.loads(Path('research_log/T058AF_summary.json').read_bytes())['overall']
    comparisons={}
    for key in ['value_huber','positive_fraction','median_cosine','direction_loss']:
        comparisons['legacy_'+key]=dict(actual=baseline_legacy[key],expected=target_legacy[key],passed=math.isclose(baseline_legacy[key],target_legacy[key],abs_tol=2e-6,rel_tol=2e-5))
    for key,other in [('positive_fraction','positive_dot_fraction'),('median_cosine','cosine_median')]:
        comparisons['detail_'+key]=dict(actual=baseline_detail[key],expected=target_detail[other],passed=math.isclose(baseline_detail[key],target_detail[other],abs_tol=2e-6,rel_tol=2e-5))
    passed=all(v['passed'] for v in comparisons.values()) and baseline_legacy['direction_rows']==target_legacy['direction_rows'] and baseline_detail['direction_rows']==target_detail['nondegenerate']
    preflight=dict(passed=passed,comparisons=comparisons,replay_tolerance=dict(atol=2e-6,rtol=2e-5,origin='fixed T014 chain numerical gate; fixed before execution'),legacy={k:v for k,v in baseline_legacy.items() if k!='cosines'},detail=baseline_detail,cached_feature_max_abs=feature_error,legacy_eligible=int(legacy['direction_mask'].sum()),detail_eligible=int(mask.sum()),cache_reopen=cache_readback,completed_utc=utc())
    atomic_json(a.out/'preflight.json',preflight);print(json.dumps(preflight),flush=True);assert passed,'FROZEN_BASELINE_REPLAY_MISMATCH'
    before_tensors={k:thash(t) for k,t in dict(x=x,mse=mse,legacy_J=legacy['jacobian'],legacy_reference=legacy['reference_gradient'],legacy_mask=legacy['direction_mask'],detail_J=jd,detail_reference=truth,detail_mask=mask).items()}
    training_started=True;head,history,initial=train_fixed(x,mse,legacy,detail)
    assert head.normalization()==frozen.normalization()
    new_legacy=training_statistics(head,x,mse,legacy);new_detail=detail_statistics(head,x,jd,truth,mask);result=verdict(new_legacy,new_detail)
    after_tensors={k:thash(t) for k,t in dict(x=x,mse=mse,legacy_J=legacy['jacobian'],legacy_reference=legacy['reference_gradient'],legacy_mask=legacy['direction_mask'],detail_J=jd,detail_reference=truth,detail_mask=mask).items()};assert before_tensors==after_tensors
    assert frozen_hash=={n:thash(v) for n,v in frozen.state_dict().items()}
    for n,h in assets.items():assert sha(n)==h,n
    atomic_json(a.out/'history.json',history);save_energy(head,a.out/'dual_tangent.pt');reloaded=load_energy(a.out/'dual_tangent.pt');assert all(torch.equal(v,reloaded.state_dict()[n]) for n,v in head.state_dict().items())
    summary=dict(**result,old=dict(legacy=preflight['legacy'],detail=baseline_detail),new=dict(legacy={k:v for k,v in new_legacy.items() if k!='cosines'},detail=new_detail),rows=7346,legacy_eligible=int(legacy['direction_mask'].sum()),detail_eligible=int(mask.sum()),legacy_ineligible=7346-int(legacy['direction_mask'].sum()),detail_ineligible=7346-int(mask.sum()))
    atomic_json(a.out/'summary.json',summary)
    final=dict(classification=result['classification'],source_bindings=source,asset_hashes_before=assets,asset_hashes_after=assets,tensor_hashes_before=before_tensors,tensor_hashes_after=after_tensors,frozen_head_before=frozen_hash,frozen_head_after=frozen_hash,initial_head_hashes=initial,head_sha256=sha(a.out/'dual_tangent.pt'),recipe=RECIPE,loss_weights=[1,1,1],training_runs=1,epochs=100,optimizer_steps=100*((7346+255)//256),checkpoint_rule='final_epoch',new_source_image_opens=0,reference_gradient_recomputations=0,target_domain_access=0,lolv2_access=0,official_test_access=0,inference_reference_leakage=0,seconds=time.perf_counter()-start,torch=torch.__version__,device='cpu (unchanged fixed T014 recipe)',completed_utc=utc())
    atomic_json(a.out/'receipt.json',final);atomic_json(a.out/'complete.json',dict(classification=result['classification'],receipt_sha256=sha(a.out/'receipt.json'),head_sha256=sha(a.out/'dual_tangent.pt'),summary_sha256=sha(a.out/'summary.json'),completed_utc=utc()));print(json.dumps(summary),flush=True)
except Exception as exc:
    (a.out/'error.txt').write_text(traceback.format_exc());atomic_json(a.out/'failure.json',dict(error=type(exc).__name__+': '+str(exc),training_started=training_started,seconds=time.perf_counter()-start,completed_utc=utc()));raise
