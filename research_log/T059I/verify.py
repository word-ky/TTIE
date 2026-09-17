import pathlib,json,sys,hashlib,numpy as np,torch
from research_log.T059I.core import tensor_only
out=pathlib.Path(sys.argv[1]);r=json.loads((out/'result.json').read_text());G=pathlib.Path('/home/wenchang/asdasdsad/wjq/TTIE/runs/20260918-050302-ttie-t059g-nn/artifacts/T059G');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();torch.set_num_threads(1)
for n,h in r['inputs_before'].items():assert sha(G/n)==h==r['inputs_after'][n]
assert sha(out/'maps.pt')==r['map_marker']['sha256'] and sha(out/'predictions.pt')==r['prediction_marker']['sha256']
m=torch.load(G/'maps.pt',map_location='cpu',weights_only=True);new=torch.load(out/'maps.pt',map_location='cpu',weights_only=True);pred=torch.load(out/'predictions.pt',map_location='cpu',weights_only=True)
tr=m['train']['x'].double().numpy();ti=m['train']['image'].numpy();global_ids=m['train']['global_index'].numpy();images=np.unique(ti);groups=[np.where(ti==i)[0] for i in images];train_target,_=tensor_only(G/'evaluation_rows.pt','train','target');metrics={};error={}
for s in ['train','heldout']:
 q=m[s]['x'].double().numpy();qi=m[s]['image'].numpy();saved=new[s];err=0.
 for start in range(0,len(q),64):
  d=np.square(q[start:start+64,None,:]-tr[None,:,:]).sum(-1);ix=np.stack([ids[d[:,ids].argmin(1)] for ids in groups],axis=1);ds=np.take_along_axis(d,ix,axis=1)
  for j in range(len(d)):
   eligible=np.where(images!=qi[start+j])[0] if s=='train' else np.arange(len(images));order=eligible[np.lexsort((images[eligible],global_ids[ix[j,eligible]],ds[j,eligible]))[:5]];expected=ix[j,order]
   assert np.array_equal(expected,saved['index'][start+j].numpy());assert np.array_equal(images[order],saved['image'][start+j].numpy());assert len(np.unique(images[order]))==5
   actual=saved['distance'][start+j].numpy();assert np.allclose(ds[j,order],actual,rtol=1e-12,atol=1e-12);err=max(err,float(np.abs(ds[j,order]-actual).max()))
   counts=[int((d[j,groups[k]]==ds[j,k]).sum()) for k in order];assert counts==saved['within_image_tie_count'][start+j].tolist();assert int((ds[j,eligible]==ds[j,order[-1]]).sum())==int(saved['fifth_distance_tie_count'][start+j])
 assert torch.equal(saved['global_index'],m['train']['global_index'][saved['index']]);target,_=tensor_only(G/'evaluation_rows.pt',s,'target');expected=train_target[saved['index']].mean(1);assert torch.equal(expected,pred[s]);e=(pred[s]-target).abs();h=float(torch.where(e<1,.5*e*e,e-.5).mean());assert h==r['consensus_huber'][s];metrics[s]=h;error[s]=err
limit=.07650849781930447
classification='fixed five-image consensus does not preserve inner-train scalar consistency; estimator smoothing is not a supported rescue' if metrics['train']>limit else 'T059-G scalar failure is consistent with single-donor estimator instability; 28-D scalar support transfers under fixed five-image consensus' if metrics['heldout']<=limit else 'conditional scalar mismatch persists under fixed five-image consensus'
assert classification==r['classification'];assert r['map_marker']['persisted_utc']<r['prediction_marker']['training_target_opened_utc']<r['prediction_marker']['persisted_utc']<r['held_scalar_opened_utc'];assert all(v==0 for v in r['counters'].values())
z=dict(verification='PASS',independent_classification=classification,consensus_huber=metrics,all_5886_five_donor_maps_exact=True,all_29430_donor_rows_and_tie_counts_verified=True,maximum_distance_replay_error=error,predictions_exact=True,chronology_verified=True,result_sha256=sha(out/'result.json'));(out/'verification.json').write_text(json.dumps(z,indent=2));print(json.dumps(z))
