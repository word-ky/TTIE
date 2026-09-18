import pathlib,json,hashlib,datetime,os,argparse,torch
from research_log.T059L.core import displacement,nearest,classify,LIMIT
from research_log.T059I.core import tensor_only
ROOT=pathlib.Path('/home/wenchang/asdasdsad/wjq/TTIE');G=ROOT/'runs/20260918-050302-ttie-t059g-nn/artifacts/T059G';K=ROOT/'runs/20260918-105152-ttie-t059k-global/artifacts/T059K';E=ROOT/'runs/20260918-023558-ttie-t059e-relative/artifacts/T059E'
INPUTS={str(G/'maps.pt'):'a40c269baed073499defcfb651600cc7ff852cc47134cc2cd4e5c2283bb807c4',str(G/'result.json'):'46177cfda13e3c00323b50c837981996aa313237de328aa4f5c632b3c95814d4',str(K/'result.json'):'b0cdbf1d630daaab08edf1f0e12e9358b3aeeaa2fdff2e9cd6a284b5174de910',str(E/'split.json'):'d44f86b7d1920f8f5efc386888ded4c65f543717165b030978f40ad595956621'}
SCALAR='671a9b3afc74f2ffecab4c0359dbec902942a43d166f31af1cfd3d7cf763cf1b'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def thash(x):return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):
 with p.open('w') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
def save(p,v):
 torch.save(v,p)
 with p.open('rb') as f:os.fsync(f.fileno())
def setup():
 torch.set_num_threads(1);binding=json.loads(pathlib.Path('research_log/T059L/source_binding.json').read_text());assert all(sha(n)==h for n,h in binding.items());assert all(sha(n)==h for n,h in INPUTS.items())
 g=json.loads((G/'result.json').read_text());k=json.loads((K/'result.json').read_text());s=json.loads((E/'split.json').read_text());m=torch.load(G/'maps.pt',weights_only=True,map_location='cpu')
 for side,expected in [('train',.06445551663637161),('heldout',.23172274231910706)]:
  assert g[side]['relative_huber']==expected==k['exact_baseline_replay'][side]['G'];q=m[side];rec=[s['canonical'][i] for i in s['row_indices'][side]];assert q['global_index'].tolist()==s['row_indices'][side];assert q['image'].tolist()==[v['image_id'] for v in rec];assert q['bank'].tolist()==[v['bank_index'] for v in rec];assert not set(q['global_index'].tolist())&set(s['row_indices']['outer']);q['state']=torch.tensor([v['state_index'] for v in rec])
 assert [len(m[v]['x']) for v in ['train','heldout']]==[4357,1529];assert [len(m[v]['image'].unique()) for v in ['train','heldout']]==[48,16]
 # Replay original G/K source bindings without opening their scalar artifacts.
 source_files={}
 for task,release,b in [('G','20260918-050258-ttie-t059g-nn',g['source_binding']),('K','20260918-092440-ttie-t059k-global',k['source_binding'])]:
  for n,h in b.items():
   path=ROOT/'releases'/release/n;assert sha(path)==h;source_files[str(path)]=h
 return m,binding,source_files
def maps(out):
 m,b,src=setup();out.mkdir(parents=True,exist_ok=False);anchors={};started=utc()
 for side,q in m.items():
  a,dx=displacement(q['x'],q['bank'],q['state']);anchors[side]=dict(anchor=a,anchor_global=q['global_index'][a],dx=dx,global_index=q['global_index'],image=q['image'],bank=q['bank'],state=q['state']);assert len(q['bank'].unique())==(240 if side=='train' else 80)
 save(out/'anchors_dx.pt',anchors);anchor_time=utc();assert torch.cuda.is_available();nn={};tr=anchors['train']
 for side,q in anchors.items():
  nn[side]=nearest(q['dx'],tr['dx'],q['image'],tr['image'],tr['global_index'],loo=side=='train');assert not (q['image']==tr['image'][nn[side]['index']]).any();nn[side]['candidate_count']=(q['image'][:,None]!=tr['image'][None,:]).sum(1) if side=='train' else torch.full((len(q['image']),),4357)
 save(out/'maps.pt',nn);write(out/'maps_persisted.json',dict(started_utc=started,anchors_persisted_utc=anchor_time,persisted_utc=utc(),anchors_sha256=sha(out/'anchors_dx.pt'),maps_sha256=sha(out/'maps.pt'),tensor_hashes={s:{n:thash(q[n]) for n in ['anchor','anchor_global','dx']} for s,q in anchors.items()},bank_counts={'train':240,'heldout':80},one_state0_anchor_per_bank=True,source_scalar_reads=0,inputs=INPUTS,source_binding=b,accepted_source_files=src,gpu=torch.cuda.get_device_name(0),physical_gpu=1));print('MAPS_FROZEN',utc())
def evaluate(out):
 m,b,src=setup();mark=json.loads((out/'maps_persisted.json').read_text());assert sha(out/'maps.pt')==mark['maps_sha256'] and sha(out/'anchors_dx.pt')==mark['anchors_sha256'];opened=utc();assert mark['persisted_utc']<opened;before={**INPUTS,str(G/'evaluation_rows.pt'):SCALAR};assert all(sha(n)==h for n,h in before.items());nn=torch.load(out/'maps.pt',weights_only=True,map_location='cpu');targets={};receipts=[];baseline={}
 for s in ['train','heldout']:
  targets[s],rec=tensor_only(G/'evaluation_rows.pt',s,'target');receipts.append(rec);pred,rec=tensor_only(G/'evaluation_rows.pt',s,'pred');receipts.append(rec);assert torch.equal(pred,targets['train'][m[s]['neighbor']['index']]);baseline[s]=float(torch.nn.functional.huber_loss(pred,targets[s]));assert baseline[s]==(.06445551663637161 if s=='train' else .23172274231910706)
 rows={};stats={}
 for s,q in nn.items():
  pred=targets['train'][q['index']];loss=torch.nn.functional.huber_loss(pred,targets[s],reduction='none');h=float(loss.mean());rows[s]=dict(pred=pred,target=targets[s],loss=loss);stats[s]=dict(rows=len(pred),huber=h,margin=LIMIT-h,absolute_baseline=baseline[s],change_from_absolute=h-baseline[s],queries_with_ties=int((q['tie_count']>1).sum()),maximum_ties=int(q['tie_count'].max()),candidate_count_min=int(q['candidate_count'].min()),candidate_count_max=int(q['candidate_count'].max()))
 after={n:sha(n) for n in before};assert before==after;assert all(sha(n)==h for n,h in {**b,**src}.items());assert sha(out/'maps.pt')==mark['maps_sha256'] and sha(out/'anchors_dx.pt')==mark['anchors_sha256'];save(out/'evaluation_rows.pt',rows)
 result=dict(status='DONE',classification=classify(stats['train']['huber'],stats['heldout']['huber']),statistics=stats,threshold=LIMIT,marker=mark,scalar_opened_utc=opened,scalar_receipts=receipts,inputs_before=before,inputs_after=after,evaluation_rows_sha256=sha(out/'evaluation_rows.pt'),source_binding=b,source_files_unchanged=True,counters={n:0 for n in ['training_runs','optimizer_steps','model_forwards','new_source_image_opens','new_feature_forwards','reference_gradient_recomputations','detail_gradient_tensor_reads','outer_supervision_reads','target_domain_access','lolv2_access','official_test_access','inference_reference_leakage']},completed_utc=utc());write(out/'result.json',result);print(json.dumps(dict(classification=result['classification'],statistics=stats)))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['maps','evaluate']);a.add_argument('--out',type=pathlib.Path,required=True);v=a.parse_args();(maps if v.stage=='maps' else evaluate)(v.out)
