import pathlib,json,hashlib,datetime,os,argparse,torch
from research_log.T059O.core import context,nearest,classify,LIMIT
ROOT=pathlib.Path('/home/wenchang/asdasdsad/wjq/TTIE');N=ROOT/'runs/20260918-131356-ttie-t059n-selector/artifacts/T059N';E=ROOT/'runs/20260918-023558-ttie-t059e-relative/artifacts/T059E';BANK=ROOT/'runs/20260912-181047-ttie-t014-stage-a-repaired/artifacts/audit'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def thash(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):
 with p.open('w') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
def save(p,v):
 torch.save(v,p)
 with p.open('rb') as f:os.fsync(f.fileno())
def setup():
 torch.set_num_threads(1);binding=json.loads(pathlib.Path('research_log/T059O/source_binding.json').read_text());original=json.loads(pathlib.Path('research_log/T059O/accepted_source_binding.json').read_text());assert all(sha(n)==v for n,v in {**binding,**original}.items());inputs={str(N/'result.json'):'17bae9184757076ab5690b7f4dfadae59a746c3c9288fa9c97ee2ea5986bff8d',str(E/'split.json'):'d44f86b7d1920f8f5efc386888ded4c65f543717165b030978f40ad595956621',str(E/'train_access.json'):'501e8594ef1443cfa7bc8b2455992f6ae5c114a508b5f45f94b7327e6072cc94',str(BANK/'training_manifest.json'):'92fd60d01ca0780880d724464c4d0a76ba1721ab5b8a2d054d4035a141673125'};assert all(sha(n)==v for n,v in inputs.items());accepted=json.loads((N/'result.json').read_text())
 for name in ['split.json','normalization.json']:
  inputs[str(N/name)]=accepted['marker']['files'][name];assert sha(N/name)==inputs[str(N/name)]
 split=json.loads((N/'split.json').read_text());assert split==accepted['split'];parent=json.loads((E/'split.json').read_text());ordered=sorted(parent['image_ids']['train'],key=lambda i:parent['group_indices'][str(i)]);assert split['sorted_parent_image_ids']==ordered and split['image_ids']['selector']==ordered[5::6];assert [len(split['row_indices'][s]) for s in ['fit','selector']]==[3604,753];assert len(set(split['image_ids']['fit']))==40 and not set(split['image_ids']['fit'])&set(split['image_ids']['selector']);norm=json.loads((N/'normalization.json').read_text());buffers={k:torch.tensor(v,dtype=torch.float32) for k,v in norm['values'].items()};assert all(thash(v)==norm['hashes'][k] for k,v in buffers.items());assert norm['fit_rows']==split['row_indices']['fit'];return parent,split,json.loads((BANK/'training_manifest.json').read_text()),buffers,binding,original,inputs

def metadata(parent,split,side):
 ids=split['row_indices'][side];assert set(ids)<=set(parent['row_indices']['train']);assert not set(ids)&set(parent['row_indices']['heldout']+parent['row_indices']['outer']);records=[parent['canonical'][i] for i in ids];assert all(v['image_id'] in split['image_ids'][side] for v in records);return dict(global_index=torch.tensor(ids),bank=torch.tensor([v['bank_index'] for v in records]),state=torch.tensor([v['state_index'] for v in records]),image=torch.tensor([v['image_id'] for v in records]))
def load_features(split,banks,side):
 accepted=json.loads((E/'train_access.json').read_text());xs=[];files={}
 for bi in split['bank_indices'][side]:
  path=BANK/banks[bi]['directory']/'bank.pt';files[str(path)]=sha(path);assert files[str(path)]==accepted['bank_file_hashes'][str(path)];xs.append(torch.load(path,weights_only=True,map_location='cpu')['features'])
 return torch.cat(xs),files
def load_scalar(split,banks,side,norm,a):
 opened=utc();accepted=json.loads((E/'train_access.json').read_text());ys=[];files={}
 for bi in split['bank_indices'][side]:
  path=BANK/banks[bi]['directory']/'targets.json';files[str(path)]=sha(path);assert files[str(path)]==accepted['bank_file_hashes'][str(path)];ys.extend(v['mse'] for v in json.loads(path.read_text()))
 mse=torch.tensor(ys,dtype=torch.float64);log=(mse+1e-6).log()
 if side=='fit':
  scale=log.std(unbiased=False);assert torch.equal(log.mean().float(),norm['y_mean']);assert torch.equal(torch.where(scale==0,torch.ones_like(scale),scale).float(),norm['y_scale'])
 t=(log.float()-norm['y_mean'])/norm['y_scale'];return t-t[a],dict(side=side,opened_utc=opened,global_indices=split['row_indices'][side],files=files)

def maps(out):
 parent,split,banks,norm,b,orig,inputs=setup();out.mkdir(parents=True,exist_ok=False);vectors={}
 for side in ['fit','selector']:
  x,files=load_features(split,banks,side);inputs.update(files);q=metadata(parent,split,side);assert len(x)==len(q['global_index'])
  if side=='fit':
   assert torch.equal(x.double().mean(0).float(),norm['x_mean']);std=x.double().std(0,unbiased=False);assert torch.equal(torch.where(std==0,torch.ones_like(std),std).float(),norm['x_scale'])
  q.update(context(x,norm['x_mean'],norm['x_scale'],q['bank'],q['state']));assert q['z'].shape==(len(x),56);q['anchor_global']=q['global_index'][q['anchor']];vectors[side]=q
 save(out/'vectors.pt',vectors);vectors_time=utc();nn={};tr=vectors['fit'];assert torch.cuda.is_available()
 for side,q in vectors.items():
  v=nearest(q['z'],tr['z'],q['image'],tr['image'],tr['global_index'],loo=side=='fit');v['image']=tr['image'][v['index']];assert not (q['image']==v['image']).any();v['candidate_count']=(q['image'][:,None]!=tr['image'][None,:]).sum(1) if side=='fit' else torch.full((len(q['image']),),3604);nn[side]=v
 save(out/'maps.pt',nn);write(out/'maps_persisted.json',dict(persisted_utc=utc(),vectors_persisted_utc=vectors_time,vectors_sha256=sha(out/'vectors.pt'),maps_sha256=sha(out/'maps.pt'),vector_hashes={s:{k:thash(q[k]) for k in ['u','u0','z','anchor','anchor_global']} for s,q in vectors.items()},normalization_hashes={k:thash(v) for k,v in norm.items()},bank_counts={s:len(q['bank'].unique()) for s,q in vectors.items()},one_unique_state0_per_bank=True,split=split,inputs=inputs,source_binding=b,accepted_sources=orig,source_scalar_reads=0,gpu=torch.cuda.get_device_name(0),physical_gpu=1));print('MAPS_FROZEN',utc())
def predict(out):
 parent,split,banks,norm,b,orig,inputs=setup();marker=json.loads((out/'maps_persisted.json').read_text());assert sha(out/'maps.pt')==marker['maps_sha256'] and sha(out/'vectors.pt')==marker['vectors_sha256'];v=torch.load(out/'vectors.pt',weights_only=True,map_location='cpu');nn=torch.load(out/'maps.pt',weights_only=True,map_location='cpu');target,access=load_scalar(split,banks,'fit',norm,v['fit']['anchor']);assert marker['persisted_utc']<access['opened_utc'];pred={s:target[q['index']] for s,q in nn.items()};save(out/'predictions.pt',pred);save(out/'fit_target.pt',target);write(out/'predictions_persisted.json',dict(persisted_utc=utc(),predictions_sha256=sha(out/'predictions.pt'),fit_target_sha256=sha(out/'fit_target.pt'),fit_access=access,selector_scalar_reads=0));print('PREDICTIONS_FROZEN',utc())
def evaluate(out):
 parent,split,banks,norm,b,orig,inputs=setup();marker=json.loads((out/'maps_persisted.json').read_text());pm=json.loads((out/'predictions_persisted.json').read_text());assert sha(out/'maps.pt')==marker['maps_sha256'] and sha(out/'vectors.pt')==marker['vectors_sha256'];assert sha(out/'predictions.pt')==pm['predictions_sha256'] and sha(out/'fit_target.pt')==pm['fit_target_sha256'];vectors=torch.load(out/'vectors.pt',weights_only=True,map_location='cpu');nn=torch.load(out/'maps.pt',weights_only=True,map_location='cpu');pred=torch.load(out/'predictions.pt',weights_only=True,map_location='cpu');held,access=load_scalar(split,banks,'selector',norm,vectors['selector']['anchor']);assert pm['persisted_utc']<access['opened_utc'];targets=dict(fit=torch.load(out/'fit_target.pt',weights_only=True),selector=held);stats={};rows={}
 for side in ['fit','selector']:
  loss=torch.nn.functional.huber_loss(pred[side],targets[side],reduction='none');h=float(loss.mean());q=nn[side];stats[side]=dict(huber=h,margin=LIMIT-h,rows=len(loss),queries_with_ties=int((q['tie_count']>1).sum()),maximum_ties=int(q['tie_count'].max()),candidate_count_min=int(q['candidate_count'].min()),candidate_count_max=int(q['candidate_count'].max()));rows[side]=dict(pred=pred[side],target=targets[side],loss=loss)
  for kind in ['image','bank']:
   ids=vectors[side][kind];stats[side]['per_'+kind]=[dict(id=int(i),rows=int((ids==i).sum()),huber=float(loss[ids==i].mean())) for i in ids.unique()]
 before={**marker['inputs'],**pm['fit_access']['files'],**access['files']};after={n:sha(n) for n in before};assert before==after and all(sha(n)==v for n,v in {**b,**orig}.items());assert sha(out/'maps.pt')==marker['maps_sha256'] and sha(out/'vectors.pt')==marker['vectors_sha256'] and sha(out/'predictions.pt')==pm['predictions_sha256'];save(out/'evaluation_rows.pt',rows);result=dict(status='DONE',classification=classify(stats['fit']['huber'],stats['selector']['huber']),statistics=stats,threshold=LIMIT,marker=marker,prediction_marker=pm,selector_access=access,inputs_before=before,inputs_after=after,evaluation_rows_sha256=sha(out/'evaluation_rows.pt'),counters={n:0 for n in ['training_runs','optimizer_steps','model_forwards','selector_scalar_reads_before_map_freeze','selector_scalar_reads_before_prediction_freeze','inner_held_supervision_reads','outer_supervision_reads','new_source_image_opens','new_feature_forwards','reference_gradient_recomputations','legacy_gradient_tensor_reads','detail_gradient_tensor_reads','target_domain_access','lolv2_access','official_test_access','inference_reference_leakage']},completed_utc=utc());write(out/'result.json',result);print(json.dumps(dict(classification=result['classification'],metrics={s:{k:stats[s][k] for k in ['huber','margin','queries_with_ties','maximum_ties']} for s in stats})))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['maps','predict','evaluate']);a.add_argument('--out',type=pathlib.Path,required=True);v=a.parse_args();dict(maps=maps,predict=predict,evaluate=evaluate)[v.stage](v.out)
