import argparse,json,statistics,math,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from research_log.T063A.common import *
from research_log.T063A.analyze import score
import torch
from ttie.common_gain import CommonRegion2

def main(out):
 f=json.loads((out/'freeze.json').read_bytes());m=json.loads(MANIFEST.read_bytes())['selected'];stored=json.loads((out/'per_image.json').read_bytes());s=json.loads((out/'result.json').read_bytes());marker=json.loads((out/'reference_open.json').read_bytes())
 assert sha(MANIFEST)==COHORT and sha(RAW/'freeze.json')==FREEZE and f['outputs']==2800 and f['step27_exact']==100
 assert f['config_sha256']==sha(out/'config.json') and sha(out/'freeze.json')==marker['freeze_sha256']==s['freeze_sha256']
 assert f['completed_utc']<marker['first_reference_or_quality_read_utc']<=min(r['reference_read_utc'] for r in stored)
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 for rec in f['rows']:
  directory=RAW/f"{rec['index']:03d}";assert sha(directory/'T062_trace.pt')==rec['input_trace_sha256']
  trace=torch.load(directory/'T062_trace.pt',weights_only=True,map_location='cpu');saved=torch.load(directory/'outputs.pt',weights_only=True,map_location='cpu');low=saved['identity'].cuda();model=CommonRegion2(trace['active']).to(low).requires_grad_(False)
  with torch.no_grad():
   for k in range(28):
    model.raw.copy_(trace['states'][k].to(low));actual=model(low).cpu()
    assert thash(actual)==rec['rendered_hashes'][k]
  del trace,saved,low,model
 refs=json.loads(Path('research_log/T062CR2/reference_inputs.json').read_bytes());assert sha(RAW/'paired.json')==PAIRED;prior=json.loads((RAW/'paired.json').read_bytes())
 jobs=[(str(out),r,item,ref,b,True) for r,item,ref,b in zip(f['rows'],m,refs,prior)]
 with ProcessPoolExecutor(max_workers=8) as pool:recomputed=list(pool.map(score,jobs))
 errors=[];choices=[];all_reachable=True
 for r,old,item in zip(recomputed,stored,m):
  assert r['low']==old['low']==item['low'] and r['normal']==old['normal']==item['normal']
  for actual,expected in zip(r['states'],old['states']):errors.extend(abs(actual[k]-expected[k]) for k in ['psnr','ssim'])
  reachable=[k for k in range(28) if r['states'][k]['psnr']>=r['controls']['T026']['psnr']-5.614]
  chosen=sorted(reachable or list(range(28)),key=lambda k:(-r['states'][k]['psnr'],k))[0]
  assert reachable==old['reachable_steps'] and chosen==old['oracle_step'] and bool(not reachable)==old['safety_unreachable']
  all_reachable &= bool(reachable);choices.append(r['states'][chosen])
 assert max(errors)<1e-10
 d=[v['psnr']-r['controls']['T036']['psnr'] for v,r in zip(choices,recomputed)];b=[v['psnr']-r['controls']['T026']['psnr'] for v,r in zip(choices,recomputed)];ds=[v['ssim']-r['controls']['T036']['ssim'] for v,r in zip(choices,recomputed)]
 values=dict(mean_delta_psnr=math.fsum(d)/100,median_delta_psnr=statistics.median(d),regressions_t026=sum(v<0 for v in b),worst_delta_t026=min(b),mean_delta_ssim=math.fsum(ds)/100,mean_psnr=statistics.fmean(v['psnr'] for v in choices),mean_ssim=statistics.fmean(v['ssim'] for v in choices))
 for k,v in values.items():assert abs(v-s[k])<1e-10
 gates=dict(mean_psnr=values['mean_delta_psnr']>=2,median_psnr=values['median_delta_psnr']>0,regressions=values['regressions_t026']<=29,worst=values['worst_delta_t026']>=-5.614,mean_ssim=values['mean_delta_ssim']>=-.001);assert gates==s['gates']
 classification='SELECTION_HEADROOM_PRESENT' if all_reachable and all(gates.values()) else 'TRAJECTORY_LIMITED';assert classification==s['classification']
 write(out/'verification.json',dict(status='PASS',classification=classification,output_hashes=2800,step27_exact=100,max_metric_error=max(errors),gates=gates,verified_utc=utc()));print(classification,'VERIFIED',max(errors),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
