import argparse,os,sys,time,json
from pathlib import Path
import torch
from ttie.common_gain import CommonRegion2
from research_log.T063A.common import *

def render(low,trace):
 model=CommonRegion2(trace['active']).to(low).eval().requires_grad_(False);images=[]
 assert trace['states'].shape==(28,1,3,2,2)
 with torch.no_grad():
  for state in trace['states']:
   model.raw.copy_(state.to(low));images.append(model(low).cpu().clone())
 return torch.stack(images)

def main(out):
 torch.set_num_threads(1);torch.manual_seed(7);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.is_available() and sha(MANIFEST)==COHORT and sha(RAW/'freeze.json')==FREEZE
 prior=json.loads((RAW/'freeze.json').read_bytes());assert sha(RAW/'config.json')==prior['config_sha256'];manifest=json.loads(MANIFEST.read_bytes())['selected']
 binding=json.loads((HERE/'binding.json').read_bytes())
 for n,h in binding.items():assert sha(n)==h,n
 out.mkdir(parents=True,exist_ok=False);allowed=set();opens=[];enabled=True
 for r in prior['rows']:
  d=RAW/f"{r['index']:03d}";allowed.update([(d/'outputs.pt').resolve(),(d/'T062_trace.pt').resolve()])
 def guard(event,args):
  if not enabled or event!='open':return
  name,mode,flags=args
  if isinstance(name,int):return
  p=Path(os.fsdecode(name)).resolve()
  if p==out or out in p.parents:return
  if p in allowed:opens.append(str(p));return
  if p.suffix in ['.py','.pyc','.so']:return
  raise PermissionError('T063-A reconstruction reads only frozen low/output/state files: '+str(p))
 sys.addaudithook(guard);begin=time.perf_counter();records=[]
 try:
  write(out/'config.json',dict(task='T063-A',label='REFERENCE_ORACLE_ONLY',source_commit=os.environ['TTIE_SOURCE_COMMIT'],source_binding=binding,cohort_sha256=COHORT,input_freeze_sha256=FREEZE,states=list(range(28)),optimizer_runs=0,renderer='exact accepted CommonRegion2',physical_gpu=1,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,started_utc=utc()))
  for i,(r,m) in enumerate(zip(prior['rows'],manifest)):
   assert i==r['index'] and r['low']==m['low'];d=RAW/f'{i:03d}'
   assert sha(d/'outputs.pt')==r['outputs_sha256'] and sha(d/'T062_trace.pt')==r['methods']['T062']['trace_sha256']
   saved=torch.load(d/'outputs.pt',weights_only=True,map_location='cpu');trace=torch.load(d/'T062_trace.pt',weights_only=True,map_location='cpu');low=saved['identity'];assert thash(low)==r['identity_hash']
   images=render(low.cuda(),trace);assert torch.equal(images[27],saved['T062']) and thash(images[27])==r['methods']['T062']['output_hash']
   target=out/f'{i:03d}';target.mkdir()
   with (target/'images.pt').open('xb') as f:torch.save(images,f);f.flush();os.fsync(f.fileno())
   record=dict(index=i,low=m['low'],image_file_sha256=sha(target/'images.pt'),rendered_hashes=[thash(im) for im in images],step27_bit_exact=True,input_outputs_sha256=r['outputs_sha256'],input_trace_sha256=r['methods']['T062']['trace_sha256'])
   write(target/'receipt.json',record);records.append(record);print('reconstructed',i+1,flush=True)
   del images,saved,trace,low
  assert len(records)==100
  write(out/'freeze.json',dict(label='REFERENCE_ORACLE_ONLY',rows=records,outputs=2800,step27_exact=100,cohort_sha256=COHORT,input_freeze_sha256=FREEZE,config_sha256=sha(out/'config.json'),completed_utc=utc(),seconds=time.perf_counter()-begin,reads=opens,normal_image_opens=0,optimizer_runs=0))
  print('ALL_2800_FROZEN',sha(out/'freeze.json'),flush=True)
 finally:enabled=False
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
