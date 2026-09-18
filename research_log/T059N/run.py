import pathlib,json,hashlib,datetime,os,argparse,torch
from ttie.energy_model import load_energy,save_energy,RECIPE
from research_log.T059N.fit import train_scalar,thash
from research_log.T059N.core import partition,select,LIMIT
ROOT=pathlib.Path('/home/wenchang/asdasdsad/wjq/TTIE');E=ROOT/'runs/20260918-023558-ttie-t059e-relative/artifacts/T059E';BANK=ROOT/'runs/20260912-181047-ttie-t014-stage-a-repaired/artifacts/audit'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):
 with p.open('w') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
def save(p,v):
 torch.save(v,p)
 with p.open('rb') as f:os.fsync(f.fileno())
def setup():
 torch.set_num_threads(1);binding=json.loads(pathlib.Path('research_log/T059N/source_binding.json').read_text());assert all(sha(n)==v for n,v in binding.items());original=json.loads(pathlib.Path('research_log/T059N/accepted_source_binding.json').read_text());assert all(sha(n)==v for n,v in original.items());inputs={str(E/'split.json'):'d44f86b7d1920f8f5efc386888ded4c65f543717165b030978f40ad595956621',str(E/'train_access.json'):'501e8594ef1443cfa7bc8b2455992f6ae5c114a508b5f45f94b7327e6072cc94',str(BANK/'training_manifest.json'):'92fd60d01ca0780880d724464c4d0a76ba1721ab5b8a2d054d4035a141673125'};assert all(sha(n)==v for n,v in inputs.items());parent=json.loads((E/'split.json').read_text());return parent,partition(parent),json.loads((BANK/'training_manifest.json').read_text()),binding,original,inputs

def load_side(parent,split,banks,side):
 assert side in ['fit','selector'];ids=split['row_indices'][side];assert set(ids)<=set(parent['row_indices']['train']);assert not set(ids)&set(parent['row_indices']['heldout']+parent['row_indices']['outer']);accepted=json.loads((E/'train_access.json').read_text());xs=[];ys=[];files={};opened=utc()
 for bi in split['bank_indices'][side]:
  directory=BANK/banks[bi]['directory']
  for n in ['bank.pt','targets.json']:
   path=directory/n;h=sha(path);assert h==accepted['bank_file_hashes'][str(path)];files[str(path)]=h
  xs.append(torch.load(directory/'bank.pt',map_location='cpu',weights_only=True)['features']);ys.extend(v['mse'] for v in json.loads((directory/'targets.json').read_text()))
 x=torch.cat(xs);mse=torch.tensor(ys,dtype=torch.float64);records=[parent['canonical'][i] for i in ids];bank=torch.tensor([v['bank_index'] for v in records]);state=torch.tensor([v['state_index'] for v in records]);a=torch.empty(len(ids),dtype=torch.long)
 for bi in bank.unique():
  z=torch.where((bank==bi)&(state==0))[0];assert len(z)==1;a[bank==bi]=z[0]
 assert len(x)==len(ids);access=dict(side=side,opened_utc=opened,global_indices=ids,image_ids=split['image_ids'][side],bank_indices=split['bank_indices'][side],bank_files=files,tensor_hashes={n:thash(v) for n,v in [('x',x),('mse',mse),('anchors',a)]});return x,mse,a,access

def values(head,x,mse,a):
 with torch.no_grad():p=head.standardized(x);t=((mse.double()+1e-6).log().float()-head.y_mean)/head.y_scale;dp=p-p[a];dt=t-t[a];loss=torch.nn.functional.huber_loss(dp,dt,reduction='none')
 return dict(p=p,t=t,delta_p=dp,delta_t=dt,loss=loss)
def train(out):
 parent,split,banks,b,orig,inputs=setup();out.mkdir(parents=True,exist_ok=False);(out/'checkpoints').mkdir();write(out/'split.json',split);x,mse,a,access=load_side(parent,split,banks,'fit');write(out/'fit_access.json',access);allrows=[];checkpoint_hashes={}
 def callback(epoch,head):
  v=values(head,x,mse,a);allrows.append(v);path=out/'checkpoints'/('epoch_%03d.pt'%epoch);save_energy(head,path)
  with path.open('rb') as f:os.fsync(f.fileno())
  checkpoint_hashes[str(path.relative_to(out))]=sha(path);return float(v['loss'].mean())
 head,history,initial,opt,gen=train_scalar(x,mse,a,callback);steps=100*((len(x)+255)//256);assert {int(v['step']) for v in opt['state'].values()}=={steps};save(out/'optimizer.pt',opt);save(out/'generator.pt',gen);save(out/'fit_rows.pt',{k:torch.stack([v[k] for v in allrows]) for k in allrows[0]});write(out/'history.json',history);write(out/'normalization.json',dict(values=head.normalization(),hashes={k:thash(getattr(head,k)) for k in ['x_mean','x_scale','y_mean','y_scale']},fit_rows=split['row_indices']['fit'],fit_images=split['image_ids']['fit']));write(out/'initial_final_hashes.json',dict(initial=initial,final={n:thash(v) for n,v in head.state_dict().items()}));inputs.update(access['bank_files']);assert all(sha(n)==v for n,v in {**inputs,**b,**orig}.items());assert len(checkpoint_hashes)==100
 names=['optimizer.pt','generator.pt','fit_rows.pt','history.json','normalization.json','initial_final_hashes.json','fit_access.json','split.json'];write(out/'checkpoints_persisted.json',dict(persisted_utc=utc(),files={**checkpoint_hashes,**{n:sha(out/n) for n in names}},recipe=RECIPE,training_runs=1,optimizer_steps=steps,fit_rows=len(x),selector_scalar_reads_before_checkpoint_freeze=0,inputs=inputs,source_binding=b,accepted_sources=orig));print('ALL100_FROZEN',utc(),steps)
def evaluate(out):
 parent,split,banks,b,orig,inputs=setup();marker=json.loads((out/'checkpoints_persisted.json').read_text());assert all(sha(out/n)==v for n,v in marker['files'].items());assert split==json.loads((out/'split.json').read_text());x,mse,a,access=load_side(parent,split,banks,'selector');assert marker['persisted_utc']<access['opened_utc'];allrows=[];curve=[]
 for epoch in range(1,101):
  head=load_energy(out/'checkpoints'/('epoch_%03d.pt'%epoch));v=values(head,x,mse,a);allrows.append(v);curve.append(float(v['loss'].mean()))
 save(out/'selector_rows.pt',{k:torch.stack([v[k] for v in allrows]) for k in allrows[0]});write(out/'selector_access.json',access);history=json.loads((out/'history.json').read_text());fit=[v['fit_huber'] for v in history];selection=select(fit,curve);before={**marker['inputs'],**access['bank_files']};after={n:sha(n) for n in before};assert before==after and all(sha(n)==v for n,v in {**b,**orig}.items());assert all(sha(out/n)==v for n,v in marker['files'].items());result=dict(status='DONE',**selection,threshold=LIMIT,fit_curve=fit,selector_curve=curve,split=split,row_counts={s:len(v) for s,v in split['row_indices'].items()},bank_counts={s:len(v) for s,v in split['bank_indices'].items()},marker=marker,selector_opened_utc=access['opened_utc'],inputs_before=before,inputs_after=after,selector_rows_sha256=sha(out/'selector_rows.pt'),counters=dict(training_runs=1,optimizer_steps=marker['optimizer_steps'],selector_scalar_reads_before_checkpoint_freeze=0,inner_held_supervision_reads=0,outer_supervision_reads=0,new_source_image_opens=0,new_feature_forwards=0,reference_gradient_recomputations=0,legacy_gradient_tensor_reads=0,detail_gradient_tensor_reads=0,target_domain_access=0,lolv2_access=0,official_test_access=0,inference_reference_leakage=0),completed_utc=utc());write(out/'result.json',result);print(json.dumps({k:result[k] for k in ['classification','selected_epoch','row_counts','bank_counts']},ensure_ascii=False));print(json.dumps(selection))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['train','evaluate']);a.add_argument('--out',type=pathlib.Path,required=True);v=a.parse_args();(train if v.stage=='train' else evaluate)(v.out)
