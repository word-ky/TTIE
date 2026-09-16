"""Only after all100selected outputs are frozen: accepted RGB metrics."""
from core import *
import argparse,csv,collections,numpy as np
p=argparse.ArgumentParser()
for k in ['split','normal-root','accepted','preflight','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();torch.set_num_threads(1);assert sha(a.accepted/'pairs.json')==PRIOR_PAIRS;assert sha(a.split)==SPLIT_SHA and sha(a.accepted/'freeze.json')==PRIOR_FREEZE
f=json.loads((a.out/'freeze.json').read_bytes());pre=json.loads((a.preflight/'preflight.json').read_bytes());assert sha(a.preflight/'preflight.json')==f['preflight_sha256'] and sha(a.out/'config.json')==f['config_sha256']
split=json.loads(a.split.read_bytes())['selected'];old=json.loads((a.accepted/'pairs.json').read_bytes());assert len(old)==len(f['rows'])==len(split)==100
allowed={str((a.normal_root/r['normal']).resolve()) for r in split};opened=[];original=o.Image.open
def normal_only(path,*args,**kwargs):
    name=str(Path(path).resolve());assert name in allowed;opened.append(dict(path=name,utc=utc()));return original(path,*args,**kwargs)
o.Image.open=normal_only;rows=[];field_values=[];starts=torch.load(a.preflight/'starts.pt',weights_only=True,map_location='cpu');initial_lifts=torch.load(a.preflight/'lifts.pt',weights_only=True,map_location='cpu')
for i,(item,row,baseline) in enumerate(zip(split,f['rows'],old)):
    assert item['low']==row['low']==baseline['low'];root=a.out/f'{i:03d}'
    for name,h in row['files'].items():assert sha(root/name)==h
    saved=torch.load(root/'output.pt',weights_only=True,map_location='cpu');hist=torch.load(root/'history.pt',weights_only=True,map_location='cpu')
    assert hist['u'].shape==(501,1,1,8,8) and hist['mse'].shape==(501,)
    index=int(hist['mse'].argmin());assert index==row['best_step'] and torch.equal(saved['u'],hist['u'][index]) and thash(saved['q'])==pre['rows'][i]['q_sha256'] and torch.equal(saved['lift'],initial_lifts[i]) and thash(saved['raw'])==pre['rows'][i]['raw_sha256'] and thash(saved['lift'])==row['lift_sha256'] and thash(saved['q'])==row['q_sha256'] and thash(saved['knots'])==row['knots_sha256']
    assert thash(saved['raw'])==row['raw_sha256'] and thash(saved['image'])==row['output_sha256']
    assert sha(a.normal_root/item['normal'])==item['normal_sha256'];normal=native(a.normal_root/item['normal']);v=o.score(saved['image'],normal)
    assert abs(v['mse']-float(hist['mse'][index]))<=1e-10
    field_values.extend(saved['ev'][0,0][torch.tensor(pre['rows'][i]['gate']['active']).reshape(2,2)[(torch.arange(saved['ev'].shape[-2])>=saved['ev'].shape[-2]//2).long()[:,None],(torch.arange(saved['ev'].shape[-1])>=saved['ev'].shape[-1]//2).long()[None,:]]].flatten().tolist())
    rows.append(dict(index=i,low=item['low'],best_step=index,psnr=v['psnr'],ssim=v['ssim'],t050_psnr=float(baseline['psnr']),t050_ssim=float(baseline['ssim']),delta_psnr=v['psnr']-float(baseline['psnr']),delta_ssim=v['ssim']-float(baseline['ssim']),seconds=row['seconds']))
def stats(v):return dict(mean=float(np.mean(v)),median=float(np.median(v)))
s=dict(label=LABEL,metrics={k:stats([r[k] for r in rows]) for k in ['psnr','ssim','t050_psnr','t050_ssim','delta_psnr','delta_ssim']},best_step_histogram=dict(collections.Counter(str(r['best_step']) for r in rows)),at_step500=sum(r['best_step']==500 for r in rows),win_equal_loss={k:dict(win=sum(r['delta_'+k]>0 for r in rows),equal=sum(r['delta_'+k]==0 for r in rows),loss=sum(r['delta_'+k]<0 for r in rows)) for k in ['psnr','ssim']},runtime_seconds=sum(r['seconds'] for r in rows))
s['field_distribution']=dict(active_pixels=len(field_values),min=min(field_values),max=max(field_values),mean=float(np.mean(field_values)),median=float(np.median(field_values)),lower_hits=sum(v<=-2 for v in field_values),upper_hits=sum(v>=2 for v in field_values))
s['classification']=verdict(s['metrics']['delta_psnr']['mean'],s['metrics']['delta_psnr']['median'],s['metrics']['delta_ssim']['mean'])
write(a.out/'pairs.json',rows);write(a.out/'summary.json',s);write(a.out/'evaluation_receipt.json',dict(completed_utc=utc(),freeze_sha256=sha(a.out/'freeze.json'),metric_start_after_freeze=True,normal_opens=opened,prior_csv_sha256=sha(a.accepted/'pairs.json'),official_test_access=False));print(json.dumps(s),flush=True)
