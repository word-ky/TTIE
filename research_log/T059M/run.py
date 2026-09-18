import pathlib,json,hashlib,datetime,os,argparse,torch
from ttie.energy_model import EnergyHead,load_energy,save_energy,RECIPE
from research_log.T059M.fit import train_scalar,thash
ROOT=pathlib.Path('/home/wenchang/asdasdsad/wjq/TTIE');E=ROOT/'runs/20260918-023558-ttie-t059e-relative/artifacts/T059E';BANK=ROOT/'runs/20260912-181047-ttie-t014-stage-a-repaired/artifacts/audit';ESRC=ROOT/'releases/20260918-023553-ttie-t059e-relative'
HASHES={'split.json':'d44f86b7d1920f8f5efc386888ded4c65f543717165b030978f40ad595956621','train_statistics.json':'c282def30eb914b936f707f603d336cf6b655441fb8c3ff41d41f2ddc6bd7666','heldout_statistics.json':'3673efcfee8872947c53450afc7af948f50d5ad51708768dc3ef2d8150a0122c','train_receipt.json':'6d754e2580ec2b872d5514371e8c2415bfdefd733bb138a2fb4ff5bd547bd375','train_access.json':'501e8594ef1443cfa7bc8b2455992f6ae5c114a508b5f45f94b7327e6072cc94','heldout_access.json':'75e4ce2b3ca3da31e179caa40293c2b3c0d62546099673ddfd9340412f537731'}
LIMIT=.07650849781930447
LABELS=['scalar-only fixed recipe does not fit the inner-train scalar objective; stop as capacity/optimization inconclusive','scalar-only transfer passes; joint Sobolev objective interference is supported as the cause of T059-E scalar failure','removing joint Sobolev losses does not rescue unseen-image scalar transfer; scalar conditioning/generalization remains unsupported under the current 28-D EnergyHead']
def classify(t,h):return LABELS[0] if t>LIMIT else LABELS[1] if h<=LIMIT else LABELS[2]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):
 with p.open('w') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
def save(p,v):
 torch.save(v,p)
 with p.open('rb') as f:os.fsync(f.fileno())
def setup():
 torch.set_num_threads(1);binding=json.loads(pathlib.Path('research_log/T059M/source_binding.json').read_text());assert all(sha(n)==h for n,h in binding.items());inputs={str(E/n):h for n,h in HASHES.items()};assert all(sha(n)==h for n,h in inputs.items());manifest=BANK/'training_manifest.json';assert sha(manifest)=='92fd60d01ca0780880d724464c4d0a76ba1721ab5b8a2d054d4035a141673125';inputs[str(manifest)]=sha(manifest)
 source=json.loads((ESRC/'research_log/T059E_source_binding.json').read_text());assert all(sha(ESRC/n)==h for n,h in source.items());inputs[str(ESRC/'research_log/T059E_source_binding.json')]=sha(ESRC/'research_log/T059E_source_binding.json');s=json.loads((E/'split.json').read_text());assert [len(s['row_indices'][k]) for k in ['train','heldout','outer']]==[4357,1529,1460];baseline={k:json.loads((E/(k+'_statistics.json')).read_text())['bank_relative_huber'] for k in ['train','heldout']};assert baseline==dict(train=.056902974843978882,heldout=.22107574343681335);return s,json.loads(manifest.read_text()),binding,inputs,baseline

def load_side(s,banks,side):
 assert side in ['train','heldout'];ids=s['row_indices'][side];assert not set(ids)&set(s['row_indices']['outer']);access=json.loads((E/(side+'_access.json')).read_text());xs=[];ms=[];files={};opened=utc()
 for bi in s['bank_indices'][side]:
  directory=BANK/banks[bi]['directory']
  for n in ['bank.pt','targets.json']:
   path=directory/n;h=sha(path);assert h==access['bank_file_hashes'][str(path)];files[str(path)]=h
  xs.append(torch.load(directory/'bank.pt',map_location='cpu',weights_only=True)['features']);ms.append(torch.tensor([v['mse'] for v in json.loads((directory/'targets.json').read_text())],dtype=torch.float64))
 x=torch.cat(xs);mse=torch.cat(ms);assert thash(x)==access['tensor_hashes']['x'] and thash(mse)==access['tensor_hashes']['mse'];records=[s['canonical'][i] for i in ids];bank=torch.tensor([v['bank_index'] for v in records]);state=torch.tensor([v['state_index'] for v in records]);anchors=torch.empty(len(ids),dtype=torch.long)
 for bi in bank.unique():
  z=torch.where((bank==bi)&(state==0))[0];assert len(z)==1;anchors[bank==bi]=z[0]
 assert thash(anchors)==access['tensor_hashes']['anchors'];return x,mse,anchors,dict(side=side,opened_utc=opened,global_indices=ids,image_ids=s['image_ids'][side],bank_indices=s['bank_indices'][side],bank_files=files,tensor_hashes={n:thash(v) for n,v in [('x',x),('mse',mse),('anchors',anchors)]},legacy_gradient_tensor_reads=0,detail_gradient_tensor_reads=0,outer_supervision_reads=0)
def stats(head,x,mse,a):
 with torch.no_grad():p=head.standardized(x);t=((mse.double()+1e-6).log().float()-head.y_mean)/head.y_scale;dp=p-p[a];dt=t-t[a];loss=torch.nn.functional.huber_loss(dp,dt,reduction='none')
 return float(loss.mean()),dict(p=p,t=t,delta_p=dp,delta_t=dt,loss=loss,anchors=a)
def train(out):
 s,banks,b,inputs,baseline=setup();out.mkdir(parents=True,exist_ok=False);x,mse,a,access=load_side(s,banks,'train');write(out/'train_access.json',access);head,history,initial,opt,generator=train_scalar(x,mse,a);old=json.loads((E/'train_receipt.json').read_text());assert initial==old['initial_head_hashes'];assert head.normalization()==old['normalization'];assert RECIPE==old['recipe'];h,rows=stats(head,x,mse,a)
 save_energy(head,out/'head.pt')
 with (out/'head.pt').open('rb') as f:os.fsync(f.fileno())
 save(out/'optimizer.pt',opt);save(out/'generator.pt',generator);save(out/'train_rows.pt',rows);write(out/'history.json',history);write(out/'train_statistics.json',dict(huber=h,margin=LIMIT-h,baseline=baseline['train'],change=h-baseline['train']));write(out/'initial_final_hashes.json',dict(initial=initial,final={n:thash(v) for n,v in head.state_dict().items()},normalization=head.normalization()));assert len(history)==100;assert {int(v['step']) for v in opt['state'].values()}=={1800}
 inputs.update(access['bank_files']);assert all(sha(n)==v for n,v in inputs.items());assert all(sha(n)==v for n,v in b.items());names=['head.pt','optimizer.pt','generator.pt','train_rows.pt','history.json','train_statistics.json','initial_final_hashes.json','train_access.json'];write(out/'checkpoint_persisted.json',dict(persisted_utc=utc(),files={n:sha(out/n) for n in names},recipe=RECIPE,training_runs=1,optimizer_steps=1800,held_scalar_reads=0,outer_supervision_reads=0,inputs=inputs,source_binding=b,baseline=baseline));print('CHECKPOINT_FROZEN',h,utc())
def evaluate(out):
 s,banks,b,inputs,baseline=setup();marker=json.loads((out/'checkpoint_persisted.json').read_text());assert all(sha(out/n)==h for n,h in marker['files'].items());x,mse,a,access=load_side(s,banks,'heldout');assert marker['persisted_utc']<access['opened_utc'];head=load_energy(out/'head.pt');h,rows=stats(head,x,mse,a);save(out/'heldout_rows.pt',rows);write(out/'heldout_access.json',access);tr=json.loads((out/'train_statistics.json').read_text());before={**marker['inputs'],**access['bank_files']};after={n:sha(n) for n in before};assert before==after and all(sha(n)==v for n,v in b.items());assert all(sha(out/n)==v for n,v in marker['files'].items());result=dict(status='DONE',classification=classify(tr['huber'],h),threshold=LIMIT,train=tr,heldout=dict(huber=h,margin=LIMIT-h,baseline=baseline['heldout'],change=h-baseline['heldout']),marker=marker,held_opened_utc=access['opened_utc'],inputs_before=before,inputs_after=after,source_binding=b,held_rows_sha256=sha(out/'heldout_rows.pt'),counters=dict(training_runs=1,optimizer_steps=1800,new_source_image_opens=0,new_feature_forwards=0,reference_gradient_recomputations=0,legacy_gradient_tensor_reads=0,detail_gradient_tensor_reads=0,outer_supervision_reads=0,target_domain_access=0,lolv2_access=0,official_test_access=0,inference_reference_leakage=0),completed_utc=utc());write(out/'result.json',result);print(json.dumps({k:result[k] for k in ['classification','train','heldout']}))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['train','evaluate']);a.add_argument('--out',type=pathlib.Path,required=True);v=a.parse_args();(train if v.stage=='train' else evaluate)(v.out)
