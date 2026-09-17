import argparse,pathlib,json,hashlib,datetime,os,torch
from research_log.T059I.core import five,tensor_only,classify,LIMIT
G=pathlib.Path('/home/wenchang/asdasdsad/wjq/TTIE/runs/20260918-050302-ttie-t059g-nn/artifacts/T059G')
HASH={'maps.pt':'a40c269baed073499defcfb651600cc7ff852cc47134cc2cd4e5c2283bb807c4','evaluation_rows.pt':'671a9b3afc74f2ffecab4c0359dbec902942a43d166f31af1cfd3d7cf763cf1b','result.json':'46177cfda13e3c00323b50c837981996aa313237de328aa4f5c632b3c95814d4'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):
 with p.open('w') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
def save(p,v):
 torch.save(v,p)
 with p.open('rb') as f:os.fsync(f.fileno())
def setup():
 torch.set_num_threads(1);b=json.loads(pathlib.Path('research_log/T059I/source_binding.json').read_text());assert all(sha(pathlib.Path(n))==h for n,h in b.items())
 for n in ['maps.pt','result.json']:assert sha(G/n)==HASH[n]
 g=json.loads((G/'result.json').read_text());m=torch.load(G/'maps.pt',map_location='cpu',weights_only=True)
 assert g['marker']['maps_sha256']==HASH['maps.pt'] and g['evaluation_rows_sha256']==HASH['evaluation_rows.pt'];assert g['marker']['metadata_hashes'][str(pathlib.Path('/home/wenchang/asdasdsad/wjq/TTIE/runs/20260918-023558-ttie-t059e-relative/artifacts/T059E/head.pt'))]=='e15e91c4e9be401bcac6d3de039ec2d1ae7141fb6cec40c388d22549652d43d0'
 for s,n,images,h in [('train',4357,48,.06445551663637161),('heldout',1529,16,.23172274231910706)]:
  assert m[s]['x'].shape==(n,28) and len(m[s]['image'].unique())==images;assert m[s]['global_index'].tolist()==g['split']['row_indices'][s] and g[s]['relative_huber']==h
 assert not set(m['train']['image'].tolist())&set(m['heldout']['image'].tolist())
 return m,g,b

def maps(out):
 m,g,b=setup();out.mkdir(parents=True,exist_ok=False);tr=m['train'];result={};assert torch.cuda.is_available()
 for s in ['train','heldout']:
  q=m[s];result[s]=five(q['x'],tr['x'],q['image'],tr['image'],tr['global_index'],loo=s=='train');v=result[s]
  assert all(len(x.unique())==5 for x in v['image']);assert not (v['image']==q['image'][:,None]).any();assert torch.equal(v['global_index'][:,0],q['neighbor']['global_index']);assert torch.equal(v['distance'][:,0],q['neighbor']['distance'])
 save(out/'maps.pt',result);write(out/'maps_persisted.json',dict(persisted_utc=utc(),sha256=sha(out/'maps.pt'),source_binding=b,G_hashes_inherited=HASH,G_features_result_hashes_fresh={n:sha(G/n) for n in ['maps.pt','result.json']},G_mixed_scalar_hash_deferred_until_held_evaluation=True,normalization_hashes=g['marker']['normalization_hashes'],gpu=torch.cuda.get_device_name(0),physical_gpu=1,held_scalar_targets_opened=0,tie_semantics='per-image first minimum in canonical row order; donor order(distance,global_row,image_id)',original_G_Hubers_recorded={s:g[s]['relative_huber'] for s in ['train','heldout']}))
 print('MAPS_FROZEN',utc())
def predict(out):
 m,g,b=setup();marker=json.loads((out/'maps_persisted.json').read_text());assert sha(out/'maps.pt')==marker['sha256'];donors=torch.load(out/'maps.pt',map_location='cpu',weights_only=True);opened=utc();target,receipt=tensor_only(G/'evaluation_rows.pt','train','target');assert len(target)==4357
 pred={s:target[donors[s]['index']].mean(1) for s in ['train','heldout']};save(out/'predictions.pt',pred);write(out/'predictions_persisted.json',dict(training_target_opened_utc=opened,training_target_receipt=receipt,persisted_utc=utc(),sha256=sha(out/'predictions.pt'),maps_sha256=marker['sha256'],held_scalar_targets_opened=0))
 print('PREDICTIONS_FROZEN',utc())
def evaluate(out):
 m,g,b=setup();mm=json.loads((out/'maps_persisted.json').read_text());pm=json.loads((out/'predictions_persisted.json').read_text());assert sha(out/'maps.pt')==mm['sha256'] and sha(out/'predictions.pt')==pm['sha256'];opened=utc();assert mm['persisted_utc']<pm['training_target_opened_utc']<pm['persisted_utc']<opened
 before={n:sha(G/n) for n in HASH};assert before==HASH;pred=torch.load(out/'predictions.pt',map_location='cpu',weights_only=True);target={};access=[];baseline={};metrics={}
 for s,h in [('train',.06445551663637161),('heldout',.23172274231910706)]:
  target[s],receipt=tensor_only(G/'evaluation_rows.pt',s,'target');access.append(receipt);old,receipt=tensor_only(G/'evaluation_rows.pt',s,'pred');access.append(receipt);assert torch.equal(old,target['train'][m[s]['neighbor']['index']]);baseline[s]=float(torch.nn.functional.huber_loss(old,target[s]));assert baseline[s]==h
  metrics[s]=float(torch.nn.functional.huber_loss(pred[s],target[s]))
 assert access[0]['sha256']==pm['training_target_receipt']['sha256'];after={n:sha(G/n) for n in HASH};assert before==after;assert all(sha(pathlib.Path(n))==h for n,h in b.items());assert sha(out/'maps.pt')==mm['sha256'] and sha(out/'predictions.pt')==pm['sha256']
 d=torch.load(out/'maps.pt',map_location='cpu',weights_only=True);r=dict(status='DONE',classification=classify(metrics['train'],metrics['heldout']),consensus_huber=metrics,threshold=LIMIT,margins={s:LIMIT-v for s,v in metrics.items()},original_G_Huber_exact=baseline,inputs_before=before,inputs_after=after,map_marker=mm,prediction_marker=pm,held_scalar_opened_utc=opened,scalar_access=access,source_binding=b,tie_statistics={s:dict(selected_donors_with_row_ties=int((d[s]['within_image_tie_count']>1).sum()),queries_with_fifth_distance_ties=int((d[s]['fifth_distance_tie_count']>1).sum())) for s in d},counters={k:0 for k in ['training_runs','optimizer_steps','model_forwards','new_source_image_opens','new_feature_forwards','reference_gradient_recomputations','detail_gradient_tensor_reads','outer_supervision_reads','target_domain_access','lolv2_access','official_test_access','inference_reference_leakage']},completed_utc=utc())
 write(out/'result.json',r);print(json.dumps({k:r[k] for k in ['classification','consensus_huber','margins','tie_statistics']}))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['maps','predict','evaluate']);a.add_argument('--out',type=pathlib.Path,required=True);a=a.parse_args();dict(maps=maps,predict=predict,evaluate=evaluate)[a.stage](a.out)
