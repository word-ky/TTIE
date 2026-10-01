import pathlib,json,hashlib,datetime,os,argparse,torch
from research_log.T059P.core import context,nearest,classify,LIMIT,extract_bank
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
 torch.set_num_threads(1);binding=json.loads(pathlib.Path('research_log/T059P/source_binding.json').read_text());original=json.loads(pathlib.Path('research_log/T059P/accepted_source_binding.json').read_text());assert all(sha(n)==v for n,v in {**binding,**original}.items());inputs={str(N/'result.json'):'17bae9184757076ab5690b7f4dfadae59a746c3c9288fa9c97ee2ea5986bff8d',str(E/'split.json'):'d44f86b7d1920f8f5efc386888ded4c65f543717165b030978f40ad595956621',str(E/'train_access.json'):'501e8594ef1443cfa7bc8b2455992f6ae5c114a508b5f45f94b7327e6072cc94',str(BANK/'training_manifest.json'):'92fd60d01ca0780880d724464c4d0a76ba1721ab5b8a2d054d4035a141673125'};assert all(sha(n)==v for n,v in inputs.items());accepted=json.loads((N/'result.json').read_text())
 for name in ['split.json','normalization.json']:
  inputs[str(N/name)]=accepted['marker']['files'][name];assert sha(N/name)==inputs[str(N/name)]
 split=json.loads((N/'split.json').read_text());assert split==accepted['split'];parent=json.loads((E/'split.json').read_text());ordered=sorted(parent['image_ids']['train'],key=lambda i:parent['group_indices'][str(i)]);assert split['sorted_parent_image_ids']==ordered and split['image_ids']['selector']==ordered[5::6];assert [len(split['row_indices'][s]) for s in ['fit','selector']]==[3604,753];assert len(set(split['image_ids']['fit']))==40 and not set(split['image_ids']['fit'])&set(split['image_ids']['selector']);norm=json.loads((N/'normalization.json').read_text());buffers={k:torch.tensor(v,dtype=torch.float32) for k,v in norm['values'].items()};assert all(thash(v)==norm['hashes'][k] for k,v in buffers.items());assert norm['fit_rows']==split['row_indices']['fit'];return parent,split,json.loads((BANK/'training_manifest.json').read_text()),buffers,binding,original,inputs

def metadata(parent,split,side):
 ids=split['row_indices'][side];assert set(ids)<=set(parent['row_indices']['train']);assert not set(ids)&set(parent['row_indices']['heldout']+parent['row_indices']['outer']);records=[parent['canonical'][i] for i in ids];assert all(v['image_id'] in split['image_ids'][side] for v in records);return dict(global_index=torch.tensor(ids),bank=torch.tensor([v['bank_index'] for v in records]),state=torch.tensor([v['state_index'] for v in records]),image=torch.tensor([v['image_id'] for v in records]))
def load_scalar(split,banks,side,norm,a):
 opened=utc();accepted=json.loads((E/'train_access.json').read_text());ys=[];files={}
 for bi in split['bank_indices'][side]:
  path=BANK/banks[bi]['directory']/'targets.json';files[str(path)]=sha(path);assert files[str(path)]==accepted['bank_file_hashes'][str(path)];ys.extend(v['mse'] for v in json.loads(path.read_text()))
 mse=torch.tensor(ys,dtype=torch.float64);log=(mse+1e-6).log()
 if side=='fit':
  scale=log.std(unbiased=False);assert torch.equal(log.mean().float(),norm['y_mean']);assert torch.equal(torch.where(scale==0,torch.ones_like(scale),scale).float(),norm['y_scale'])
 t=(log.float()-norm['y_mean'])/norm['y_scale'];return t-t[a],dict(side=side,opened_utc=opened,global_indices=split['row_indices'][side],files=files)

def provenance(split,banks):
 t=ROOT/'releases/20260912-181039-ttie-t014-stage-a-repaired';receipt_path=BANK/'T014_energy_receipt.json';config_path=BANK/'config.json';receipt=json.loads(receipt_path.read_text());config=json.loads(config_path.read_text());identity=config['model_identity'];assert identity==receipt['model_identity'];assert identity['sha256']=='1bd3c7172de5b207ceac554f5ab5266166f3b9baccc9af5989bc801016d080ad';inputs={str(receipt_path):sha(receipt_path),str(config_path):sha(config_path),identity['path']:identity['sha256']}
 for name in ['ttie/__init__.py','ttie/clip_signal.py','ttie/natural.py','ttie/sobolev_pilot.py','ttie/energy_io.py']:
  expected=receipt['source_code_sha256'][name];assert sha(t/name)==expected;inputs[str(t/name)]=expected
  if pathlib.Path(name).exists():assert sha(name)==expected
 for side in ['fit','selector']:
  for bi in split['bank_indices'][side]:
   entry=banks[bi];assert entry['image_id'] in split['image_ids'][side];p=BANK/entry['directory']/'bank_images.pt';inputs[str(p)]=entry['files']['bank_images.pt']['sha256'];assert p.stat().st_size==entry['files']['bank_images.pt']['bytes']
 assert all(sha(p)==h for p,h in inputs.items())
 import open_clip
 cfg=open_clip.get_pretrained_cfg('ViT-B-32','laion2b_s34b_b79k');accepted=identity['pretrained_config'];assert list(cfg['mean'])==accepted['mean'] and list(cfg['std'])==accepted['std']
 pre=dict(size=224,mean=list(cfg['mean']),std=list(cfg['std']),resize='shorter side; bicubic antialias=True align_corners=False',crop='round centered 224',views=['full','top_left','top_right','bottom_left','bottom_right'],embedding='L2 normalized per view; preserve all five ordered outputs; flatten only for Euclidean distance',code_sha256=receipt['source_code_sha256']['ttie/natural.py'])
 return inputs,dict(model_identity=identity,preprocessing=pre,preprocessing_sha256=hashlib.sha256(json.dumps(pre,sort_keys=True).encode()).hexdigest(),accepted_source_sha=receipt['source_sha'],open_clip_version=open_clip.__version__,torch_version=str(torch.__version__))
def maps(out):
 parent,split,banks,norm,b,orig,inputs=setup();out.mkdir(parents=True,exist_ok=False);extra,clip=provenance(split,banks);inputs.update(extra);write(out/'provenance.json',dict(bound_utc=utc(),inputs=inputs,clip=clip));print('PROVENANCE_BOUND',utc(),flush=True)
 from ttie.clip_signal import FrozenCLIP
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(7);assert torch.cuda.is_available();encoder=FrozenCLIP.from_checkpoint(clip['model_identity']['path'],'cuda:0').eval();assert not any(p.requires_grad for p in encoder.parameters());vectors={};count=0;shape=None
 for side in ['fit','selector']:
  chunks=[]
  for bi in split['bank_indices'][side]:
   entry=banks[bi];e=extract_bank(encoder,BANK/entry['directory']/'bank_images.pt','cuda:0');assert len(e)==entry['states'];shape=tuple(e.shape[1:]) if shape is None else shape;assert tuple(e.shape[1:])==shape;chunks.append(e);count+=len(e)
  e=torch.cat(chunks);q=metadata(parent,split,side);assert len(e)==len(q['global_index']);q.update(context(e,q['bank'],q['state']));q['anchor_global']=q['global_index'][q['anchor']];vectors[side]=q;print('FEATURES',side,e.shape,utc(),flush=True)
 assert count==4357 and len(shape)==2 and shape[0]==5 and shape[1]>0;save(out/'vectors.pt',vectors);vectors_time=utc();del encoder;torch.cuda.empty_cache();nn={};tr=vectors['fit']
 for side,q in vectors.items():
  v=nearest(q['z'],tr['z'],q['image'],tr['image'],tr['global_index'],loo=side=='fit');v['image']=tr['image'][v['index']];assert not (q['image']==v['image']).any();v['candidate_count']=(q['image'][:,None]!=tr['image'][None,:]).sum(1) if side=='fit' else torch.full((len(q['image']),),3604);nn[side]=v
 save(out/'maps.pt',nn);write(out/'maps_persisted.json',dict(persisted_utc=utc(),vectors_persisted_utc=vectors_time,vectors_sha256=sha(out/'vectors.pt'),maps_sha256=sha(out/'maps.pt'),vector_hashes={s:{k:thash(q[k]) for k in q} for s,q in vectors.items()},map_hashes={s:{k:thash(q[k]) for k in q} for s,q in nn.items()},normalization_hashes={k:thash(v) for k,v in norm.items()},bank_counts={s:len(q['bank'].unique()) for s,q in vectors.items()},one_unique_state0_per_bank=True,split=split,inputs=inputs,source_binding=b,accepted_sources=orig,source_scalar_reads=0,gpu=torch.cuda.get_device_name(0),physical_gpu=1,clip=clip,embedding_shape_per_state=list(shape),flattened_embedding_dimension=shape[0]*shape[1],z_dimension=2*shape[0]*shape[1],state_batch_size=1,encoder_view_batch_size=5,dtype='float32',distance_dtype='float64',authorized_clip_state_image_forwards=count,authorized_view_embeddings=count*shape[0],fixed_from_checkpoint_text_initialization=True));print('MAPS_FROZEN',utc(),flush=True)
def predict(out):
 parent,split,banks,norm,b,orig,inputs=setup();marker=json.loads((out/'maps_persisted.json').read_text());assert sha(out/'maps.pt')==marker['maps_sha256'] and sha(out/'vectors.pt')==marker['vectors_sha256'];v=torch.load(out/'vectors.pt',weights_only=True,map_location='cpu');nn=torch.load(out/'maps.pt',weights_only=True,map_location='cpu');target,access=load_scalar(split,banks,'fit',norm,v['fit']['anchor']);assert marker['persisted_utc']<access['opened_utc'];pred={s:target[q['index']] for s,q in nn.items()};save(out/'predictions.pt',pred);save(out/'fit_target.pt',target);write(out/'predictions_persisted.json',dict(persisted_utc=utc(),predictions_sha256=sha(out/'predictions.pt'),prediction_tensor_hashes={s:thash(v) for s,v in pred.items()},fit_target_sha256=sha(out/'fit_target.pt'),fit_access=access,selector_scalar_reads=0));print('PREDICTIONS_FROZEN',utc())
def evaluate(out):
 parent,split,banks,norm,b,orig,inputs=setup();marker=json.loads((out/'maps_persisted.json').read_text());pm=json.loads((out/'predictions_persisted.json').read_text());assert sha(out/'maps.pt')==marker['maps_sha256'] and sha(out/'vectors.pt')==marker['vectors_sha256'];assert sha(out/'predictions.pt')==pm['predictions_sha256'] and sha(out/'fit_target.pt')==pm['fit_target_sha256'];vectors=torch.load(out/'vectors.pt',weights_only=True,map_location='cpu');nn=torch.load(out/'maps.pt',weights_only=True,map_location='cpu');pred=torch.load(out/'predictions.pt',weights_only=True,map_location='cpu');held,access=load_scalar(split,banks,'selector',norm,vectors['selector']['anchor']);assert pm['persisted_utc']<access['opened_utc'];targets=dict(fit=torch.load(out/'fit_target.pt',weights_only=True),selector=held);stats={};rows={}
 for side in ['fit','selector']:
  loss=torch.nn.functional.huber_loss(pred[side],targets[side],reduction='none');h=float(loss.mean());q=nn[side];stats[side]=dict(huber=h,margin=LIMIT-h,rows=len(loss),queries_with_ties=int((q['tie_count']>1).sum()),maximum_ties=int(q['tie_count'].max()),candidate_count_min=int(q['candidate_count'].min()),candidate_count_max=int(q['candidate_count'].max()));rows[side]=dict(pred=pred[side],target=targets[side],loss=loss)
  for kind in ['image','bank']:
   ids=vectors[side][kind];stats[side]['per_'+kind]=[dict(id=int(i),rows=int((ids==i).sum()),huber=float(loss[ids==i].mean())) for i in ids.unique()]
 before={**marker['inputs'],**pm['fit_access']['files'],**access['files']};after={n:sha(n) for n in before};assert before==after and all(sha(n)==v for n,v in {**b,**orig}.items());assert sha(out/'maps.pt')==marker['maps_sha256'] and sha(out/'vectors.pt')==marker['vectors_sha256'] and sha(out/'predictions.pt')==pm['predictions_sha256'];save(out/'evaluation_rows.pt',rows);result=dict(status='DONE',classification=classify(stats['fit']['huber'],stats['selector']['huber']),statistics=stats,threshold=LIMIT,marker=marker,prediction_marker=pm,selector_access=access,inputs_before=before,inputs_after=after,evaluation_rows_sha256=sha(out/'evaluation_rows.pt'),counters={n:0 for n in ['training_runs','optimizer_steps','reference_image_feature_forwards','selector_scalar_reads_before_map_freeze','selector_scalar_reads_before_prediction_freeze','inner_held_supervision_reads','outer_supervision_reads','new_source_cohort_opens','reference_gradient_recomputations','legacy_gradient_tensor_reads','detail_gradient_tensor_reads','target_domain_access','lolv2_access','official_test_access','inference_reference_leakage']},authorized_clip_state_image_forwards=marker['authorized_clip_state_image_forwards'],authorization='89f0baabb74530724c9c78e6014f2d42e15cbcab',source_sha=json.loads(pathlib.Path('research_log/T059P/publication.json').read_text())['source'],completed_utc=utc());write(out/'result.json',result);print(json.dumps(dict(classification=result['classification'],metrics={s:{k:stats[s][k] for k in ['huber','margin','queries_with_ties','maximum_ties']} for s in stats})))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['maps','predict','evaluate']);a.add_argument('--out',type=pathlib.Path,required=True);v=a.parse_args();dict(maps=maps,predict=predict,evaluate=evaluate)[v.stage](v.out)
