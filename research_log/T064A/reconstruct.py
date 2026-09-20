import argparse,os,sys,time,json
from pathlib import Path
import torch
from ttie.common_gain import CommonRegion2
from ttie.lolv2_gamma_core import native_rgb
from research_log.T064A.common import *

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
 accepted=json.loads((RAW/'config.json').read_bytes())
 for n,h in accepted['source_binding'].items():assert sha(n)==h,n
 binding=json.loads((HERE/'binding.json').read_bytes())
 for n,h in binding.items():assert sha(n)==h,n
 out.mkdir(parents=True,exist_ok=False);allowed=set();opens=[];enabled=True
 for r in prior['rows']:
  d=RAW/f"{r['index']:03d}";allowed.update([(LOWROOT/r['low']).resolve(),(d/'T063_trace.pt').resolve()])
 def guard(event,args):
  if not enabled or event!='open':return
  name,mode,flags=args
  if isinstance(name,int):return
  p=Path(os.fsdecode(name)).resolve()
  if p==out or out in p.parents:return
  if p in allowed:opens.append(str(p));return
  if p.suffix in ['.py','.pyc','.so']:return
  raise PermissionError('T064-A reconstruction reads only frozen low/output/state files: '+str(p))
 sys.addaudithook(guard);begin=time.perf_counter();records=[]
 try:
  write(out/'config.json',dict(task='T064-A',label='REFERENCE_ORACLE_ONLY',source_commit=os.environ['TTIE_SOURCE_COMMIT'],source_binding=binding,cohort_sha256=COHORT,input_freeze_sha256=FREEZE,states=list(range(28)),optimizer_runs=0,renderer='exact accepted CommonRegion2',physical_gpu=1,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,started_utc=utc()))
  for i,(r,m) in enumerate(zip(prior['rows'],manifest)):
   assert i==r['index'] and r['low']==m['low'];d=RAW/f'{i:03d}'
   assert sha(LOWROOT/r['low'])==r['low_sha256'] and sha(d/'T063_trace.pt')==r['methods']['T063']['trace_sha256']
   trace=torch.load(d/'T063_trace.pt',weights_only=True,map_location='cpu');low=native_rgb(LOWROOT/r['low']);assert thash(low)==r['identity_hash']
   selected=r['methods']['T063']['selected_step'];assert selected==trace['selected_step']
   images=render(low.cuda(),trace);assert thash(images[selected])==r['methods']['T063']['output_hash']
   target=out/f'{i:03d}';target.mkdir()
   with (target/'images.pt').open('xb') as f:torch.save(images,f);f.flush();os.fsync(f.fileno())
   record=dict(index=i,low=m['low'],selected_step=selected,accepted_selected_hash=r['methods']['T063']['output_hash'],low_sha256=r['low_sha256'],image_file_sha256=sha(target/'images.pt'),rendered_hashes=[thash(im) for im in images],selected_bit_exact=True,input_outputs_sha256=r['outputs_sha256'],input_trace_sha256=r['methods']['T063']['trace_sha256'])
   write(target/'receipt.json',record);records.append(record);print('reconstructed',i+1,flush=True)
   del images,trace,low
  assert len(records)==100
  write(out/'freeze.json',dict(label='REFERENCE_ORACLE_ONLY',rows=records,outputs=2800,selected_exact=100,cohort_sha256=COHORT,input_freeze_sha256=FREEZE,config_sha256=sha(out/'config.json'),completed_utc=utc(),seconds=time.perf_counter()-begin,reads=opens,normal_image_opens=0,optimizer_runs=0))
  print('ALL_2800_FROZEN',sha(out/'freeze.json'),flush=True)
 finally:enabled=False
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
