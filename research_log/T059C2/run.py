"""T059-C2 separate train/evaluate entry points using immutable source caches."""
import argparse,json,time,os,traceback
from pathlib import Path
import torch
from PIL import Image
from ttie.energy_model import save_energy,load_energy,RECIPE
from ttie.sobolev_train import training_statistics
from research_log.T059B.fit import train_fixed,detail_statistics
from research_log.T058A_tangent.core import sha,thash,utc,BANK
from research_log.T059A.support import atomic_json
from research_log.T059C2.core import split_manifest,read_reference_rows,numeric_gates
ROOT=Path('/home/wenchang/asdasdsad/wjq/TTIE')
BANKROOT=ROOT/'runs/20260912-181047-ttie-t014-stage-a-repaired/artifacts/audit'
JROOT=ROOT/'runs/20260917-165444-ttie-t059a-jacobian'
RROOT=ROOT/'runs/20260917-153308-ttie-t058af-stageb'

def initialize():
    Image.open=lambda *a,**k:(_ for _ in ()).throw(AssertionError('NO_IMAGE_ACCESS'))
    torch.set_num_threads(1);torch.manual_seed(7)
    assert RECIPE['device']=='cpu'

def bindings():
    source=json.loads(Path('research_log/T059C2_source_binding.json').read_bytes());assets={}
    def bind(p,h):
        p=Path(p);assert sha(p)==h,str(p);assets[str(p)]=h
    for n,h in source.items():bind(n,h)
    bind(BANKROOT/'training_manifest.json',BANK);banks=json.loads((BANKROOT/'training_manifest.json').read_bytes())
    split=split_manifest(banks)
    # Opaque SHA reads bind bytes only; no targets/reference tensors deserialized.
    for b in banks:
        for group in ['files','source_supervision_files']:
            for n,v in b[group].items():bind(BANKROOT/b['directory']/n,v['sha256'])
    for root,name in [(JROOT,'T059A'),(RROOT,'T058AF')]:
        archive=json.loads(Path('research_log/'+name+'_archives.json').read_bytes())
        for n,h in archive['files'].items():bind(root/n,h)
    return banks,split,assets,source

def load_side(banks,split,side):
    selected=split['row_indices'][side];wanted=set(selected);xs=[];ys=[];legacy=[];reads=[]
    for bi in split['bank_indices'][side]:
        b=banks[bi];directory=BANKROOT/b['directory']
        xs.append(torch.load(directory/'bank.pt',map_location='cpu',weights_only=True)['features'])
        target=json.loads((directory/'targets.json').read_bytes());deriv=torch.load(directory/'source_derivatives.pt',map_location='cpu',weights_only=True)
        ys.append(torch.tensor([r['mse'] for r in target],dtype=torch.float64));legacy.append(deriv)
        reads.append(dict(bank_index=bi,image_id=b['image_id'],files=[str(directory/'targets.json'),str(directory/'source_derivatives.pt')],utc=utc()))
    jmanifest=json.loads((JROOT/'artifacts/T059A/manifest.json').read_bytes());js={}
    for i in selected:assert all(jmanifest['rows'][i][k]==v for k,v in split['canonical'][i].items())
    for c in jmanifest['chunks']:
        payload=torch.load(JROOT/'artifacts/T059A'/c['file'],map_location='cpu',weights_only=True)
        for i,j in zip(payload['indices'],payload['J']):
            if i in wanted:js[i]=j
    refs={};reference_reads=[]
    # Chunk list is from already-bound archive filenames; never deserialize AF row metrics before checkpoint.
    archive=json.loads(Path('research_log/T058AF_archives.json').read_bytes())
    for n in sorted(archive['files']):
        if n.startswith('artifacts/T058AF/reference_') and n.endswith('.pt'):
            values,ranges=read_reference_rows(RROOT/n,wanted);refs.update(values)
            if ranges:reference_reads.append(dict(file=str(RROOT/n),ranges=ranges,utc=utc()))
    assert set(refs)==wanted and set(js)==wanted
    x=torch.cat(xs);mse=torch.cat(ys);leg={k:torch.cat([r[k] for r in legacy]) for k in legacy[0]}
    truth=torch.stack([refs[i] for i in selected]).flatten(1);jac=torch.stack([js[i] for i in selected]);mask=truth.double().norm(dim=1)>1e-12
    assert x.shape==(len(selected),28) and jac.shape==(len(selected),28,64)
    for t in [x,mse,jac,truth,leg['jacobian'],leg['reference_gradient']]:assert torch.isfinite(t).all()
    detail=dict(jacobian=jac,reference_gradient=truth,direction_mask=mask)
    tensors=dict(x=x,mse=mse,**{'legacy_'+k:v for k,v in leg.items()},**{'detail_'+k:v for k,v in detail.items()})
    return x,mse,leg,detail,dict(side=side,supervision_banks=reads,reference_storage_reads=reference_reads,loaded_reference_indices=selected,tensor_hashes={k:thash(v) for k,v in tensors.items()}),tensors

def statistics(head,x,mse,leg,detail):
    l=training_statistics(head,x,mse,leg);d=detail_statistics(head,x,detail['jacobian'],detail['reference_gradient'],detail['direction_mask'])
    return dict(rows=len(x),legacy={k:v for k,v in l.items() if k!='cosines'},detail=d,legacy_ineligible=len(x)-l['direction_rows'],detail_ineligible=len(x)-d['direction_rows'])

def unchanged(assets):
    for n,h in assets.items():assert sha(Path(n))==h,n

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','evaluate']);p.add_argument('--out',type=Path,required=True);a=p.parse_args();initialize();start=time.perf_counter()
    if a.stage=='train':a.out.mkdir(parents=True,exist_ok=False)
    else:
        marker=json.loads((a.out/'checkpoint_persisted.json').read_bytes());assert sha(a.out/'head.pt')==marker['head_sha256']
    banks,split,assets,source=bindings()
    if a.stage=='train':
        atomic_json(a.out/'split.json',dict(manifest_sha256=BANK,**split,created_utc=utc()))
        x,mse,leg,detail,access,tensors=load_side(banks,split,'train')
        atomic_json(a.out/'training_access.json',access)
        assert not set(access['loaded_reference_indices'])&set(split['row_indices']['heldout'])
        head,history,initial=train_fixed(x,mse,leg,detail)
        save_energy(head,a.out/'head.pt')
        with (a.out/'head.pt').open('rb') as f:os.fsync(f.fileno())
        headhash=sha(a.out/'head.pt');persisted=dict(head_sha256=headhash,checkpoint_rule='epoch100',heldout_supervision_reads_before_checkpoint=0,persisted_utc=utc(),training_runs=1,optimizer_steps=100*((len(x)+255)//256))
        atomic_json(a.out/'checkpoint_persisted.json',persisted)
        assert all(torch.equal(v,load_energy(a.out/'head.pt').state_dict()[n]) for n,v in head.state_dict().items())
        stats=statistics(head,x,mse,leg,detail);assert access['tensor_hashes']=={k:thash(v) for k,v in tensors.items()};unchanged(assets)
        atomic_json(a.out/'history.json',history);atomic_json(a.out/'training_statistics.json',stats)
        receipt=dict(source_bindings=source,asset_hashes_before=assets,asset_hashes_after=assets,normalization=head.normalization(),initial_head_hashes=initial,final_head_hashes={k:thash(v) for k,v in head.state_dict().items()},training_tensor_hashes_before=access['tensor_hashes'],training_tensor_hashes_after={k:thash(v) for k,v in tensors.items()},recipe=RECIPE,loss_weights=[1,1,1],training_rows=len(x),epochs=len(history),**persisted,seconds=time.perf_counter()-start,completed_utc=utc(),torch=torch.__version__)
        atomic_json(a.out/'train_receipt.json',receipt);print(json.dumps(dict(stage='train',**persisted,training_rows=len(x),seconds=receipt['seconds'])),flush=True)
    else:
        saved=json.loads((a.out/'split.json').read_bytes());assert all(saved[k]==v for k,v in json.loads(json.dumps(split)).items())
        train_receipt=json.loads((a.out/'train_receipt.json').read_bytes());assert assets==train_receipt['asset_hashes_after']
        evaluation_opened_utc=utc();x,mse,leg,detail,access,tensors=load_side(banks,split,'heldout')
        head=load_energy(a.out/'head.pt');stats=statistics(head,x,mse,leg,detail);result=numeric_gates(stats['legacy'],stats['detail'])
        # Read AF row metadata only after checkpoint persistence to verify selected reference hashes.
        af=json.loads((RROOT/'artifacts/T058AF/manifest.json').read_bytes())
        for j,i in enumerate(split['row_indices']['heldout']):assert thash(detail['reference_gradient'][j].reshape(1,1,8,8))==af['rows'][i]['reference_sha256']
        assert access['tensor_hashes']=={k:thash(v) for k,v in tensors.items()};assert sha(a.out/'head.pt')==marker['head_sha256'];unchanged(assets)
        summary=dict(**result,heldout=stats,training=json.loads((a.out/'training_statistics.json').read_bytes()))
        atomic_json(a.out/'heldout_access.json',access);atomic_json(a.out/'summary.json',summary)
        receipt=dict(classification=result['classification'],source_bindings=source,asset_hashes_before=assets,asset_hashes_after=assets,heldout_tensor_hashes_before=access['tensor_hashes'],heldout_tensor_hashes_after={k:thash(v) for k,v in tensors.items()},head_sha256=marker['head_sha256'],final_head_hashes={k:thash(v) for k,v in head.state_dict().items()},checkpoint_persisted_utc=marker['persisted_utc'],heldout_evaluation_opened_utc=evaluation_opened_utc,training_runs=1,optimizer_steps=marker['optimizer_steps'],heldout_supervision_reads_before_checkpoint=0,new_source_image_opens=0,reference_gradient_recomputations=0,new_feature_forwards=0,target_domain_access=0,lolv2_access=0,official_test_access=0,inference_reference_leakage=0,seconds=time.perf_counter()-start,completed_utc=utc())
        atomic_json(a.out/'evaluation_receipt.json',receipt)
        files={n:sha(a.out/n) for n in ['split.json','head.pt','checkpoint_persisted.json','history.json','training_statistics.json','train_receipt.json','training_access.json','heldout_access.json','summary.json','evaluation_receipt.json']}
        atomic_json(a.out/'complete.json',dict(classification=result['classification'],files=files,completed_utc=utc()));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
