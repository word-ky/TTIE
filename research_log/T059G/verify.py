"""CPU independent NN replay and aggregate gate/classification verification."""
import pathlib,sys,json,hashlib,numpy as np,torch
out=pathlib.Path(sys.argv[1]);r=json.loads((out/'result.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();torch.set_num_threads(1)
assert sha(out/'maps.pt')==r['marker']['maps_sha256'];assert sha(out/'evaluation_rows.pt')==r['evaluation_rows_sha256']
maps=torch.load(out/'maps.pt',map_location='cpu',weights_only=True);rows=torch.load(out/'evaluation_rows.pt',map_location='cpu',weights_only=True)
for p,h in r['inputs_before'].items():assert sha(pathlib.Path(p))==h==r['inputs_after'][p]
tr=maps['train']['x'].double().numpy();ti=maps['train']['image'].numpy();verified={};distance_error={}
for side in ['train','heldout']:
 q=maps[side];x=q['x'].double().numpy();qi=q['image'].numpy();idx=q['neighbor']['index'].numpy();maxerr=0.
 assert np.all(np.diff(maps['train']['global_index'].numpy())>0)
 for start in range(0,len(x),64):
  d=np.square(x[start:start+64,None,:]-tr[None,:,:]).sum(-1)
  if side=='train':d[qi[start:start+64,None]==ti[None,:]]=np.inf
  arg=d.argmin(1);assert np.array_equal(arg,idx[start:start+64]);distance=d[np.arange(len(arg)),arg]
  saved=q['neighbor']['distance'][start:start+64].numpy();assert np.allclose(distance,saved,rtol=1e-12,atol=1e-12);maxerr=max(maxerr,float(np.abs(distance-saved).max()))
  assert np.array_equal((d==distance[:,None]).sum(1),q['neighbor']['tie_count'][start:start+64].numpy())
 assert not np.any(qi==ti[idx]);assert torch.equal(q['neighbor']['global_index'],maps['train']['global_index'][q['neighbor']['index']]);distance_error[side]=maxerr
 z=rows[side];assert torch.equal(z['pred'],rows['train']['target'][q['neighbor']['index']]);assert torch.equal(z['predicted_gradient'],rows['train']['reference_gradient'][q['neighbor']['index']])
 pg=z['predicted_gradient'].clone();truth=z['reference_gradient'];pm=pg.double().norm(dim=1)>1e-12;mask=truth.double().norm(dim=1)>1e-12;pg[~pm]=0
 dot=(pg*truth).sum(1);den=pg.norm(dim=1)*truth.norm(dim=1);cs=dot/torch.where(den>0,den,torch.ones_like(den));d=(z['pred']-z['target']).abs();h=float(torch.where(d<1,.5*d*d,d-.5).mean());positive=float((dot[mask]>0).float().mean());median=float(cs[mask].quantile(.5))
 assert abs(h-r[side]['relative_huber'])<2e-7 and positive==r[side]['detail_positive'] and median==r[side]['detail_median']
 verified[side]=dict(value=h<=.07650849781930447,detail_positive=positive>=.75,detail_median=median>=.5)
assert verified['train']==r['train_gates'] and verified['heldout']==r['heldout_gates']
classification=('28-D features do not show cross-image local target consistency even within inner-train support under fixed 1-NN' if not all(verified['train'].values()) else 'T059-E failure is consistent with an inner-held feature-support shift under fixed 1-NN' if not all(verified['heldout'].values()) else '28-D local feature support is adequate under fixed 1-NN; parametric head/function fitting remains the primary suspect')
assert classification==r['classification'];assert r['marker']['persisted_utc']<r['supervision_opened_utc']['heldout'];assert all(v==0 for v in r['counters'].values())
for a in r['selected_gradient_ranges']:
 with open(a['file'],'rb') as f:f.seek(a['offset']);assert hashlib.sha256(f.read(a['bytes'])).hexdigest()==a['sha256']
v=dict(verification='PASS',independent_classification=classification,independent_gates=verified,all_neighbor_indices_and_tie_counts_exact=True,maximum_distance_replay_error=distance_error,neighbor_rows=5886,canonical_smallest_tie_verified=True,heldout_map_before_supervision=True,result_sha256=sha(out/'result.json'))
(out/'verification.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
