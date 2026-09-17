"""One nested relative-value fit; isolated inner-held-out evaluation, outer labels unopened."""
import argparse,json,os,time,hashlib,zipfile
from pathlib import Path
import torch
from PIL import Image
from ttie.energy_model import load_energy,save_energy,RECIPE
from ttie.sobolev_train import training_statistics,cosine
from research_log.T059B.fit import detail_statistics
from research_log.T059D.core import bank_metrics,distribution
from research_log.T059E.fit import train_fixed_relative
from research_log.T059E.core import nested_split,selected_rows,gate_values
from research_log.T058A_tangent.core import sha,thash,utc,BANK
from research_log.T059A.support import atomic_json
ROOT=Path('/home/wenchang/asdasdsad/wjq/TTIE');BANKROOT=ROOT/'runs/20260912-181047-ttie-t014-stage-a-repaired/artifacts/audit'
JROOT=ROOT/'runs/20260917-165444-ttie-t059a-jacobian';RROOT=ROOT/'runs/20260917-153308-ttie-t058af-stageb'

def setup():
    Image.open=lambda *a,**k:(_ for _ in ()).throw(AssertionError('NO_IMAGE_ACCESS'))
    torch.set_num_threads(1);torch.manual_seed(7);assert RECIPE['device']=='cpu'
    source=json.loads(Path('research_log/T059E_source_binding.json').read_bytes())
    for n,h in source.items():assert sha(Path(n))==h,n
    assert sha(BANKROOT/'training_manifest.json')==BANK
    banks=json.loads((BANKROOT/'training_manifest.json').read_bytes());return banks,nested_split(banks),source

def load_side(banks,split,side):
    assert side in ['train','heldout'];selected=split['row_indices'][side];wanted=set(selected);assert not wanted&set(split['row_indices']['outer'])
    x=[];y=[];legacy=[];files={};bank_reads=[]
    for bi in split['bank_indices'][side]:
        b=banks[bi];directory=BANKROOT/b['directory']
        for group in ['files','source_supervision_files']:
            for n,h in b[group].items():assert sha(directory/n)==h['sha256'];files[str(directory/n)]=h['sha256']
        x.append(torch.load(directory/'bank.pt',map_location='cpu',weights_only=True)['features']);targets=json.loads((directory/'targets.json').read_bytes());y.append(torch.tensor([v['mse'] for v in targets],dtype=torch.float64));legacy.append(torch.load(directory/'source_derivatives.pt',map_location='cpu',weights_only=True))
        bank_reads.append(dict(bank_index=bi,image_id=b['image_id'],utc=utc()))
    caches={};cache_reads=[]
    for label,root,task,key,shape,prefix in [('jacobian',JROOT,'T059A','J',(28,64),'jacobian_'),('reference_gradient',RROOT,'T058AF','g_R',(1,1,8,8),'reference_')]:
        archive=json.loads(Path('research_log/'+task+'_archives.json').read_bytes());values={}
        for n,h in sorted(archive['files'].items()):
            if n.startswith('artifacts/'+task+'/'+prefix) and n.endswith('.pt'):
                subset,receipt=selected_rows(root/n,wanted,key,shape);values.update(subset)
                if receipt['ranges']:receipt.update(accepted_full_chunk_sha256=h,full_chunk_hash_recomputed=False,full_chunk_hash_note='Accepted immutable archive identity; no full-storage reads because mixed chunks contain excluded outer supervision',utc=utc());cache_reads.append(receipt)
        assert set(values)==wanted;caches[label]=torch.stack([values[i] for i in selected])
    x=torch.cat(x);mse=torch.cat(y);leg={k:torch.cat([v[k] for v in legacy]) for k in legacy[0]};truth=caches['reference_gradient'].flatten(1);detail=dict(jacobian=caches['jacobian'],reference_gradient=truth,direction_mask=truth.double().norm(dim=1)>1e-12)
    records=[split['canonical'][i] for i in selected];bank_indices=torch.tensor([r['bank_index'] for r in records]);state_indices=torch.tensor([r['state_index'] for r in records]);anchors=torch.empty(len(selected),dtype=torch.int64)
    for bi in split['bank_indices'][side]:
        ids=(bank_indices==bi).nonzero().flatten();a=ids[state_indices[ids]==0];assert len(a)==1;anchors[ids]=a[0]
    tensors=dict(x=x,mse=mse,anchors=anchors,**{'legacy_'+k:v for k,v in leg.items()},**{'detail_'+k:v for k,v in detail.items()})
    assert x.shape==(len(selected),28) and detail['jacobian'].shape==(len(selected),28,64)
    for t in tensors.values():assert torch.isfinite(t).all()
    access=dict(side=side,bank_reads=bank_reads,selected_row_indices=selected,cache_reads=cache_reads,bank_file_hashes=files,tensor_hashes={k:thash(v) for k,v in tensors.items()},outer_supervision_reads=0)
    return x,mse,leg,detail,anchors,bank_indices,state_indices,access,tensors

def verify_inputs(access):
    for n,h in access['bank_file_hashes'].items():assert sha(Path(n))==h,n
    # Verify only already-authorized selected numeric bytes, never full mixed storage.
    for c in access['cache_reads']:
        with zipfile.ZipFile(c['file']) as z:
            name=next(n for n in z.namelist() if n.endswith('/data.pkl'));assert hashlib.sha256(z.read(name)).hexdigest()==c['metadata_sha256']
        with open(c['file'],'rb') as f:
            for r in c['ranges']:
                f.seek(r['offset']);assert hashlib.sha256(f.read(r['bytes'])).hexdigest()==r['sha256']

def stats(head,x,mse,leg,detail,bi,si,indices):
    l=training_statistics(head,x,mse,leg);d=detail_statistics(head,x,detail['jacobian'],detail['reference_gradient'],detail['direction_mask'])
    leaf=x.detach().clone().requires_grad_();p=head.standardized(leaf);q,=torch.autograd.grad((p*head.y_scale+head.y_mean).sum(),leaf)
    lc=cosine(torch.einsum('bfi,bf->bi',leg['jacobian'],q),leg['reference_gradient']);dc=cosine(torch.einsum('bfi,bf->bi',detail['jacobian'],q),detail['reference_gradient'])
    target=((mse.double()+1e-6).log().float()-head.y_mean)/head.y_scale
    banks,dp,dt,a=bank_metrics(p.detach(),target,lc,dc,leg['direction_mask'],detail['direction_mask'],bi,si,torch.tensor(indices))
    rel=float(torch.nn.functional.huber_loss(dp,dt,delta=1.))
    return dict(rows=len(x),bank_relative_huber=rel,absolute_value_huber=l['value_huber'],legacy={k:v for k,v in l.items() if k!='cosines'},detail=d,legacy_ineligible=len(x)-l['direction_rows'],detail_ineligible=len(x)-d['direction_rows'],per_bank=banks,spearman_distribution=distribution([b['spearman'] for b in banks]),regret_distribution=distribution([b['argmin_regret'] for b in banks])),dict(p=p.detach(),t=target,delta_p=dp,delta_t=dt,bank_indices=bi,state_indices=si,anchor_indices=a,global_indices=torch.tensor(indices))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','evaluate']);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();start=time.perf_counter()
    if a.stage=='train':a.out.mkdir(parents=True,exist_ok=False)
    else:
        marker=json.loads((a.out/'checkpoint_persisted.json').read_bytes());assert sha(a.out/'head.pt')==marker['head_sha256']
    banks,split,source=setup()
    if a.stage=='train':atomic_json(a.out/'split.json',dict(manifest_sha256=BANK,**split,created_utc=utc()))
    else:
        saved=json.loads((a.out/'split.json').read_bytes());assert all(saved[k]==v for k,v in json.loads(json.dumps(split)).items())
    side='train' if a.stage=='train' else 'heldout';opened=utc();x,mse,leg,detail,anchors,bi,si,access,tensors=load_side(banks,split,side)
    atomic_json(a.out/(side+'_access.json'),access)
    if a.stage=='train':
        head,history,initial=train_fixed_relative(x,mse,leg,detail,anchors)
        save_energy(head,a.out/'head.pt')
        with (a.out/'head.pt').open('rb') as f:os.fsync(f.fileno())
        marker=dict(head_sha256=sha(a.out/'head.pt'),persisted_utc=utc(),epochs=100,training_runs=1,optimizer_steps=100*((len(x)+255)//256),inner_heldout_supervision_reads_before_checkpoint=0,outer_supervision_reads=0)
        atomic_json(a.out/'checkpoint_persisted.json',marker);atomic_json(a.out/'history.json',history)
        assert all(torch.equal(v,load_energy(a.out/'head.pt').state_dict()[k]) for k,v in head.state_dict().items())
    else:head=load_energy(a.out/'head.pt')
    statistic,row_values=stats(head,x,mse,leg,detail,bi,si,split['row_indices'][side]);atomic_json(a.out/(side+'_statistics.json'),statistic);torch.save(row_values,a.out/(side+'_row_values.pt'))
    assert access['tensor_hashes']=={k:thash(v) for k,v in tensors.items()};verify_inputs(access)
    for n,h in source.items():assert sha(Path(n))==h,n
    receipt=dict(source_bindings=source,manifest_sha256=BANK,side=side,opened_utc=opened,checkpoint_sha256=marker['head_sha256'],checkpoint_persisted_utc=marker['persisted_utc'],normalization=head.normalization(),normalization_provenance='Only inner-training x and absolute log(MSE+1e-6); original mean/population std recipe',final_head_hashes={k:thash(v) for k,v in head.state_dict().items()},bank_file_hashes_before=access['bank_file_hashes'],bank_file_hashes_after=access['bank_file_hashes'],selected_storage_ranges_before=access['cache_reads'],selected_storage_ranges_after=access['cache_reads'],tensor_hashes_before=access['tensor_hashes'],tensor_hashes_after={k:thash(v) for k,v in tensors.items()},row_tensor_hashes={k:thash(v) for k,v in row_values.items()},recipe=RECIPE,value_loss='bank-relative Huber delta1; no absolute mixture',loss_weights=[1,1,1],training_runs=1,optimizer_steps=marker['optimizer_steps'],outer_supervision_reads=0,inner_heldout_supervision_reads_before_checkpoint=0,new_source_image_opens=0,reference_gradient_recomputations=0,new_feature_forwards=0,target_domain_access=0,lolv2_access=0,official_test_access=0,inference_reference_leakage=0,seconds=time.perf_counter()-start,completed_utc=utc())
    if a.stage=='train':receipt['initial_head_hashes']=initial
    atomic_json(a.out/(side+'_receipt.json'),receipt)
    if a.stage=='evaluate':
        result=gate_values(statistic);summary=dict(**result,inner_train=json.loads((a.out/'train_statistics.json').read_bytes()),inner_heldout=statistic,outer_evaluated=False)
        atomic_json(a.out/'summary.json',summary)
        names=['split.json','checkpoint_persisted.json','head.pt','history.json','train_access.json','heldout_access.json','train_statistics.json','heldout_statistics.json','train_row_values.pt','heldout_row_values.pt','train_receipt.json','heldout_receipt.json','summary.json']
        atomic_json(a.out/'complete.json',dict(classification=result['classification'],files={n:sha(a.out/n) for n in names},completed_utc=utc()));print(json.dumps(result),flush=True)
    else:print(json.dumps(dict(stage='train',rows=len(x),**marker)),flush=True)
if __name__=='__main__':main()
