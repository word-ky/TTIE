import argparse,pathlib,json,hashlib,os,datetime,zipfile,torch
from research_log.T059G.core import nearest,summarize,classify
ROOT=pathlib.Path('/home/wenchang/asdasdsad/wjq/TTIE');E=ROOT/'runs/20260918-023558-ttie-t059e-relative/artifacts/T059E';BANK=ROOT/'runs/20260912-181047-ttie-t014-stage-a-repaired/artifacts/audit'
HASHES={'head.pt':'e15e91c4e9be401bcac6d3de039ec2d1ae7141fb6cec40c388d22549652d43d0','split.json':'d44f86b7d1920f8f5efc386888ded4c65f543717165b030978f40ad595956621','train_access.json':'501e8594ef1443cfa7bc8b2455992f6ae5c114a508b5f45f94b7327e6072cc94','heldout_access.json':'75e4ce2b3ca3da31e179caa40293c2b3c0d62546099673ddfd9340412f537731','train_row_values.pt':'77618dc7b5b4c6b5b2f6b8ad180521a1ab845867d17b4ffec7a04a2815efbc85','heldout_row_values.pt':'976d8b68e4988623c954230a87577acf5e4d46b5c4583e35bb530fdf1a37b61d'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def thash(t):return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):
 with p.open('w') as f:json.dump(v,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def save(p,v):
 torch.save(v,p)
 with p.open('rb') as f:os.fsync(f.fileno())
def checked(name):
 p=E/name;assert sha(p)==HASHES[name],name;return p
def base():
 torch.set_num_threads(1);binding=json.loads(pathlib.Path('research_log/T059G/source_binding.json').read_text());assert all(sha(pathlib.Path(n))==h for n,h in binding.items())
 s=json.loads(checked('split.json').read_text());assert [len(s['row_indices'][k]) for k in ['train','heldout','outer']]==[4357,1529,1460];assert [len(s['image_ids'][k]) for k in ['train','heldout','outer']]==[48,16,16]
 assert all(not set(s['image_ids'][a])&set(s['image_ids'][b]) for a,b in [('train','heldout'),('train','outer'),('heldout','outer')]);assert len(set(sum(s['row_indices'].values(),[])))==7346
 return s,binding

def maps(out):
 s,binding=base();out.mkdir(parents=True,exist_ok=False);manifest=BANK/'training_manifest.json';assert sha(manifest)=='92fd60d01ca0780880d724464c4d0a76ba1721ab5b8a2d054d4035a141673125';banks=json.loads(manifest.read_text());state=torch.load(checked('head.pt'),map_location='cpu',weights_only=True)['state_dict'];mean=state['x_mean'];scale=state['x_scale'];data={};files={str(manifest):sha(manifest)}
 for side in ['train','heldout']:
  access=json.loads(checked(side+'_access.json').read_text());xs=[]
  for bi in s['bank_indices'][side]:
   path=BANK/banks[bi]['directory']/'bank.pt';h=sha(path);assert h==access['bank_file_hashes'][str(path)];files[str(path)]=h
   # bank.pt contains only states/grids/scores/features, no images or supervision.
   xs.append(torch.load(path,map_location='cpu',weights_only=True)['features'])
  x=torch.cat(xs);assert thash(x)==access['tensor_hashes']['x'];assert x.shape==(len(s['row_indices'][side]),28)
  if side=='train':
   assert torch.equal(x.double().mean(0).float(),mean);std=x.double().std(0,unbiased=False);assert torch.equal(torch.where(std==0,torch.ones_like(std),std).float(),scale)
  records=[s['canonical'][i] for i in s['row_indices'][side]];data[side]=dict(x=(x-mean)/scale,raw_x_hash=thash(x),global_index=torch.tensor(s['row_indices'][side]),image=torch.tensor([a['image_id'] for a in records]),bank=torch.tensor([a['bank_index'] for a in records]))
 tr=data['train'];held=data['heldout'];assert torch.cuda.is_available()
 for side in ['train','heldout']:
  q=data[side];q['neighbor']=nearest(q['x'],tr['x'],q['image'],tr['image'],tr['global_index'],loo=side=='train')
  assert not (q['image']==tr['image'][q['neighbor']['index']]).any()
 save(out/'maps.pt',data);marker=dict(persisted_utc=utc(),maps_sha256=sha(out/'maps.pt'),gpu=torch.cuda.get_device_name(0),physical_gpu='CUDA_VISIBLE_DEVICES=1',distance='float32 frozen standardization then float64 direct squared differences summed over28; no matmul/TF32; canonical first argmin ties',normalization_hashes={k:thash(state[k]) for k in ['x_mean','x_scale']},feature_file_hashes=files,metadata_hashes={str(E/n):HASHES[n] for n in ['head.pt','split.json','train_access.json','heldout_access.json']},source_binding=binding,heldout_supervision_reads=0,model_forwards=0)
 write(out/'maps_persisted.json',marker);print(json.dumps({k:marker[k] for k in ['persisted_utc','maps_sha256','gpu','heldout_supervision_reads']}))

def reference(access,indices):
 selected={};ranges=[];wanted=set(indices)
 for c in access['cache_reads']:
  if c['key']!='g_R':continue
  with zipfile.ZipFile(c['file']) as z:
   n=next(n for n in z.namelist() if n.endswith('/data.pkl'));assert hashlib.sha256(z.read(n)).hexdigest()==c['metadata_sha256']
  with open(c['file'],'rb') as f:
   for r in c['ranges']:
    assert r['index'] in wanted;f.seek(r['offset']);raw=f.read(r['bytes']);assert hashlib.sha256(raw).hexdigest()==r['sha256'];selected[r['index']]=torch.frombuffer(bytearray(raw),dtype=torch.float32).clone();ranges.append(dict(file=c['file'],**r))
 assert set(selected)==wanted;g=torch.stack([selected[i] for i in indices]);assert g.shape==(len(indices),64) and thash(g)==access['tensor_hashes']['detail_reference_gradient'];assert thash(g.double().norm(dim=1)>1e-12)==access['tensor_hashes']['detail_direction_mask']
 return g,ranges

def evaluate(out):
 s,binding=base();marker=json.loads((out/'maps_persisted.json').read_text());assert sha(out/'maps.pt')==marker['maps_sha256'];data=torch.load(out/'maps.pt',map_location='cpu',weights_only=True);supervision={};opens={};read_ranges=[];inputfiles=dict(marker['metadata_hashes']);inputfiles.update(marker['feature_file_hashes'])
 for side in ['train','heldout']:
  opens[side]=utc();assert marker['persisted_utc']<opens[side]
  path=checked(side+'_row_values.pt');inputfiles[str(path)]=HASHES[path.name];rows=torch.load(path,map_location='cpu',weights_only=True);assert rows['global_indices'].tolist()==s['row_indices'][side]
  access=json.loads(checked(side+'_access.json').read_text());g,ranges=reference(access,s['row_indices'][side]);read_ranges+=ranges;supervision[side]=dict(target=rows['delta_t'],gradient=g)
 result={};evalrows={}
 for side in ['train','heldout']:
  q=data[side];idx=q['neighbor']['index'];pred=supervision['train']['target'][idx];pg=supervision['train']['gradient'][idx];target=supervision[side]['target'];truth=supervision[side]['gradient']
  result[side]=summarize(pred,target,pg,truth,q['bank'],q['image'],q['neighbor']['distance']);evalrows[side]=dict(pred=pred,target=target,predicted_gradient=pg,reference_gradient=truth)
 for n,h in inputfiles.items():assert sha(pathlib.Path(n))==h
 for r in read_ranges:
  with open(r['file'],'rb') as f:f.seek(r['offset']);assert hashlib.sha256(f.read(r['bytes'])).hexdigest()==r['sha256']
 assert all(sha(pathlib.Path(n))==h for n,h in binding.items());assert sha(out/'maps.pt')==marker['maps_sha256']
 save(out/'evaluation_rows.pt',evalrows)
 result.update(classify(result['train'],result['heldout']));result.update(status='DONE',split=s,marker=marker,supervision_opened_utc=opens,inputs_before=inputfiles,inputs_after=inputfiles,selected_gradient_ranges=read_ranges,gradient_numeric_range_before_after_verified=True,full_mixed_supervision_chunk_hashes_recomputed=False,source_binding=binding,evaluation_rows_sha256=sha(out/'evaluation_rows.pt'),counters={k:0 for k in ['training_runs','optimizer_steps','model_forwards','new_source_image_opens','reference_gradient_recomputations','new_feature_forwards','outer_supervision_reads','target_domain_access','lolv2_access','official_test_access','inference_reference_leakage']},completed_utc=utc())
 write(out/'result.json',result);print(json.dumps({k:result[k] for k in ['classification','train_gates','heldout_gates']}))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['maps','evaluate']);a.add_argument('--out',type=pathlib.Path,required=True);a=a.parse_args();(maps if a.stage=='maps' else evaluate)(a.out)
