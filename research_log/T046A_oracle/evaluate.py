"""Only after all100selected outputs are frozen: accepted RGB metrics."""
from core import *
import argparse,csv,collections,numpy as np
p=argparse.ArgumentParser()
for k in ['split','normal-root','accepted','preflight','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();torch.set_num_threads(1);assert sha(a.accepted/'per_image.csv')==PRIOR_CSV;assert sha(a.split)==SPLIT_SHA and sha(a.accepted/'freeze.json')==PRIOR_FREEZE
f=json.loads((a.out/'freeze.json').read_bytes());pre=json.loads((a.preflight/'preflight.json').read_bytes());assert sha(a.preflight/'preflight.json')==f['preflight_sha256'] and sha(a.out/'config.json')==f['config_sha256']
split=json.loads(a.split.read_bytes())['selected'];old=list(csv.DictReader((a.accepted/'per_image.csv').open()));assert len(old)==len(f['rows'])==len(split)==100
allowed={str((a.normal_root/r['normal']).resolve()) for r in split};opened=[];original=o.Image.open
def normal_only(path,*args,**kwargs):
    name=str(Path(path).resolve());assert name in allowed;opened.append(dict(path=name,utc=utc()));return original(path,*args,**kwargs)
o.Image.open=normal_only;rows=[]
for i,(item,row,baseline) in enumerate(zip(split,f['rows'],old)):
    assert item['low']==row['low']==baseline['low'];root=a.out/f'{i:03d}'
    for name,h in row['files'].items():assert sha(root/name)==h
    saved=torch.load(root/'output.pt',weights_only=True,map_location='cpu');hist=torch.load(root/'history.pt',weights_only=True,map_location='cpu')
    assert hist['raw'].shape==(1001,1,3,2,2) and hist['mse'].shape==(1001,)
    index=int(hist['mse'].argmin());assert index==row['best_step'] and torch.equal(saved['raw'],hist['raw'][index])
    assert thash(saved['raw'])==row['raw_sha256'] and thash(saved['image'])==row['output_sha256']
    assert sha(a.normal_root/item['normal'])==item['normal_sha256'];normal=native(a.normal_root/item['normal']);v=o.score(saved['image'],normal)
    assert abs(v['mse']-float(hist['mse'][index]))<=1e-10
    rows.append(dict(index=i,low=item['low'],best_step=index,psnr=v['psnr'],ssim=v['ssim'],t035_psnr=float(baseline['oracle_psnr']),t035_ssim=float(baseline['oracle_ssim']),delta_psnr=v['psnr']-float(baseline['oracle_psnr']),delta_ssim=v['ssim']-float(baseline['oracle_ssim']),seconds=row['seconds']))
def stats(v):return dict(mean=float(np.mean(v)),median=float(np.median(v)))
s=dict(label=LABEL,metrics={k:stats([r[k] for r in rows]) for k in ['psnr','ssim','t035_psnr','t035_ssim','delta_psnr','delta_ssim']},best_step_histogram=dict(collections.Counter(str(r['best_step']) for r in rows)),at_step1000=sum(r['best_step']==1000 for r in rows),win_equal_loss={k:dict(win=sum(r['delta_'+k]>0 for r in rows),equal=sum(r['delta_'+k]==0 for r in rows),loss=sum(r['delta_'+k]<0 for r in rows)) for k in ['psnr','ssim']},runtime_seconds=sum(r['seconds'] for r in rows))
s['classification']=verdict(s['metrics']['delta_psnr']['mean'],s['metrics']['delta_psnr']['median'])
write(a.out/'pairs.json',rows);write(a.out/'summary.json',s);write(a.out/'evaluation_receipt.json',dict(completed_utc=utc(),freeze_sha256=sha(a.out/'freeze.json'),metric_start_after_freeze=True,normal_opens=opened,prior_csv_sha256=sha(a.accepted/'per_image.csv'),official_test_access=False));print(json.dumps(s),flush=True)
