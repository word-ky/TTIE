import argparse,pathlib,json,hashlib,os,datetime,torch
from research_log.T059H.core import boundaries,reweight,summary
ROOT=pathlib.Path('/home/wenchang/asdasdsad/wjq/TTIE/runs/20260918-050302-ttie-t059g-nn/artifacts/T059G')
EXPECTED={'maps.pt':'a40c269baed073499defcfb651600cc7ff852cc47134cc2cd4e5c2283bb807c4','evaluation_rows.pt':'671a9b3afc74f2ffecab4c0359dbec902942a43d166f31af1cfd3d7cf763cf1b','result.json':'46177cfda13e3c00323b50c837981996aa313237de328aa4f5c632b3c95814d4'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def thash(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):
 with p.open('w') as f:json.dump(v,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def main():
 a=argparse.ArgumentParser();a.add_argument('--out',required=True,type=pathlib.Path);a=a.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
 binding=json.loads(pathlib.Path('research_log/T059H/source_binding.json').read_text());assert all(sha(pathlib.Path(n))==h for n,h in binding.items())
 assert sha(ROOT/'maps.pt')==EXPECTED['maps.pt'];m=torch.load(ROOT/'maps.pt',map_location='cpu',weights_only=True);td=m['train']['neighbor']['distance'];hd=m['heldout']['neighbor']['distance'];assert len(td)==4357 and len(hd)==1529
 b=boundaries(td);marker=dict(boundaries=b.tolist(),quantile='empirical type7 linear interpolation: position=(n-1)*p',probabilities=[i/10 for i in range(1,10)],tie_semantics='left closed/right open; exact internal boundary to higher bin; outer endpoints -infinity,+infinity include all rows; last bin closed',persisted_utc=utc(),maps_sha256=EXPECTED['maps.pt'],subgroup_errors_read=False);write(a.out/'bins_frozen.json',marker)
 pairs_opened=utc();assert pairs_opened>=marker['persisted_utc'];before={n:sha(ROOT/n) for n in EXPECTED};assert before==EXPECTED
 g=json.loads((ROOT/'result.json').read_text());v=torch.load(ROOT/'evaluation_rows.pt',map_location='cpu',weights_only=True);losses={};replay={};tensors={}
 assert g['marker']['maps_sha256']==EXPECTED['maps.pt'] and g['evaluation_rows_sha256']==EXPECTED['evaluation_rows.pt']
 for side,n,expected in [('train',4357,.06445551663637161),('heldout',1529,.23172274231910706)]:
  q=m[side];idx=q['neighbor']['index'];z=v[side];assert len(z['pred'])==len(z['target'])==n;assert q['global_index'].tolist()==g['split']['row_indices'][side]
  assert torch.equal(q['neighbor']['global_index'],m['train']['global_index'][idx]);assert torch.equal(z['pred'],v['train']['target'][idx]);assert not (q['image']==m['train']['image'][idx]).any()
  losses[side]=torch.nn.functional.huber_loss(z['pred'],z['target'],reduction='none');replay[side]=float(losses[side].mean());assert replay[side]==g[side]['relative_huber']==expected
  tensors[side]={k:thash(t) for k,t in dict(global_index=q['global_index'],neighbor_index=idx,neighbor_global=q['neighbor']['global_index'],distance=q['neighbor']['distance'],pred=z['pred'],target=z['target']).items()}
 r=reweight(td,hd,losses['train'],losses['heldout'],b);assert replay['heldout']>r['threshold'];after={n:sha(ROOT/n) for n in EXPECTED};assert before==after;assert all(sha(pathlib.Path(n))==h for n,h in binding.items())
 r.update(status='DONE',input_root=str(ROOT),inputs_before=before,inputs_after=after,tensor_replay_hashes=tensors,original_huber=replay,train_distance=summary(td),held_distance=summary(hd),held_fraction_above_train_p90=float((hd>b[-1]).double().mean()),held_fraction_above_train_maximum=float((hd>td.max()).double().mean()),bins_frozen=marker,scalar_pairs_opened_utc=pairs_opened,source_binding=binding,numeric_convention='Frozen float32 per-row Huber; bin means and reweighted sum float64; train-only float64 type7 quantiles',counters={k:0 for k in ['training_runs','optimizer_steps','model_forwards','new_neighbor_searches','new_source_image_opens','new_feature_forwards','reference_gradient_recomputations','outer_supervision_reads','target_domain_access','lolv2_access','official_test_access','inference_reference_leakage']},completed_utc=utc())
 write(a.out/'result.json',r);print(json.dumps({k:r[k] for k in ['classification','distance_reweighted_train_huber','original_huber','held_fraction_above_train_p90','held_fraction_above_train_maximum']}))
if __name__=='__main__':main()
