"""Three frozen cache-path controls; no optimizer or training."""
import argparse,json,time,traceback,math
from pathlib import Path
import torch
from PIL import Image
from ttie.energy_model import load_energy,save_energy,RECIPE
from ttie.sobolev_train import training_statistics
from research_log.T058A_tangent.core import sha,thash,utc,RECEIPT,BANK,ENERGY
from research_log.T059A.support import atomic_json,reopen_cache
from research_log.T059BR.controls import scalar_gate,detail_control,chain_control,difference
ROOT=Path('/home/wenchang/asdasdsad/wjq/TTIE')
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();start=time.perf_counter();a.out.mkdir(parents=True,exist_ok=False);training_started=False
Image.open=lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('NO_NEW_IMAGE_ACCESS'))
torch.set_num_threads(1);torch.manual_seed(7)
assets={}
def bind(path,h):
    path=Path(path);assert sha(path)==h,str(path);assets[str(path)]=h
try:
    source=json.loads(Path('research_log/T059BR_source_binding.json').read_bytes())
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
    for t in [x,mse,jd,truth,phi,legacy['jacobian'],legacy['reference_gradient']]:assert torch.isfinite(t).all()
    input_tensors={k:t for k,t in dict(x=x,phi=phi,mse=mse,legacy_J=legacy['jacobian'],legacy_reference=legacy['reference_gradient'],legacy_mask=legacy['direction_mask'],detail_J=jd,detail_reference=truth,detail_mask=mask).items()}
    before={k:thash(t) for k,t in input_tensors.items()}
    legacy_stats=training_statistics(frozen,x,mse,legacy)
    phi_stats,q_phi,g_phi=detail_control(frozen,phi,jd,truth,mask)
    x_stats,q_x,g_x=detail_control(frozen,x,jd,truth,mask)
    stored=torch.cat(accepted).flatten(1);assert stored.shape==(7346,64)
    chain=chain_control(g_phi,stored)
    old=old_training['final_train_statistics']['sobolev_primary'];af_stats=json.loads(Path('research_log/T058AF_summary.json').read_bytes())['overall']
    legacy_gates={k:scalar_gate(legacy_stats[k],old[k]) for k in ['value_huber','positive_fraction','median_cosine','direction_loss']}
    detail_gates={k:scalar_gate(x_stats[k],af_stats[n]) for k,n in [('positive_fraction','positive_dot_fraction'),('median_cosine','cosine_median')]}
    phi_diagnostics={k:scalar_gate(phi_stats[k],af_stats[n]) for k,n in [('positive_fraction','positive_dot_fraction'),('median_cosine','cosine_median')]}
    groups=dict(legacy_value=all(v['passed'] for v in legacy_gates.values()) and legacy_stats['direction_rows']==old['direction_rows'],phi_integrity=chain['passed'] and phi_stats['direction_rows']==af_stats['nondegenerate'],x_detail=all(v['passed'] for v in detail_gates.values()) and x_stats['direction_rows']==af_stats['nondegenerate'])
    summary=dict(classification='functional cache compatibility established' if all(groups.values()) else 'functional cache compatibility not established',groups=groups,rows=7346,legacy_control=dict(actual={k:v for k,v in legacy_stats.items() if k!='cosines'},expected={k:v for k,v in old.items() if k!='cosines'},gates=legacy_gates),phi_control=dict(actual=phi_stats,expected_nondegenerate=af_stats['nondegenerate'],chain=chain,alignment_diagnostics_not_gates=phi_diagnostics),x_detail_control=dict(actual=x_stats,expected=dict(positive_fraction=af_stats['positive_dot_fraction'],median_cosine=af_stats['cosine_median'],direction_rows=af_stats['nondegenerate']),gates=detail_gates),feature_difference=difference(x,phi))
    after={k:thash(t) for k,t in input_tensors.items()};assert before==after and frozen_hash=={n:thash(v) for n,v in frozen.state_dict().items()}
    for n,h in assets.items():assert sha(n)==h,n
    atomic_json(a.out/'summary.json',summary)
    receipt=dict(classification=summary['classification'],source_bindings=source,asset_hashes_before=assets,asset_hashes_after=assets,input_tensor_hashes_before=before,input_tensor_hashes_after=after,frozen_head_before=frozen_hash,frozen_head_after=frozen_hash,output_tensor_hashes={k:thash(v) for k,v in dict(q_phi=q_phi,g_phi=g_phi,q_x=q_x,g_x=g_x,accepted=stored).items()},frozen_controls=3,training_runs=0,optimizer_steps=0,new_source_image_opens=0,reference_gradient_recomputations=0,new_feature_or_scorer_forwards=0,target_domain_access=0,lolv2_access=0,official_test_access=0,inference_reference_leakage=0,device='cpu',torch=torch.__version__,seconds=time.perf_counter()-start,completed_utc=utc())
    atomic_json(a.out/'receipt.json',receipt);atomic_json(a.out/'complete.json',dict(classification=summary['classification'],receipt_sha256=sha(a.out/'receipt.json'),summary_sha256=sha(a.out/'summary.json'),completed_utc=utc()))
    print(json.dumps(summary),flush=True)
except Exception as exc:
    (a.out/'error.txt').write_text(traceback.format_exc());atomic_json(a.out/'failure.json',dict(error=type(exc).__name__+': '+str(exc),training_started=False,training_runs=0,seconds=time.perf_counter()-start,completed_utc=utc()));raise
