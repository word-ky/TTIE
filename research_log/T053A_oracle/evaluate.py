"""Frozen T053 outputs versus accepted T052, reference-only evaluation."""
from core import *
import argparse,collections,numpy as np
p=argparse.ArgumentParser()
for k in ['split','normal-root','accepted','preflight','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();torch.set_num_threads(1)
assert sha(a.split)==SPLIT_SHA and sha(a.accepted/'freeze.json')==PRIOR_FREEZE and sha(a.accepted/'pairs.json')==PRIOR_PAIRS
f=json.loads((a.out/'freeze.json').read_bytes());pre=json.loads((a.preflight/'preflight.json').read_bytes());assert sha(a.preflight/'preflight.json')==f['preflight_sha256'] and sha(a.out/'config.json')==f['config_sha256']
split=json.loads(a.split.read_bytes())['selected'];old=json.loads((a.accepted/'pairs.json').read_bytes());assert len(split)==len(f['rows'])==len(old)==100
allowed={str((a.normal_root/r['normal']).resolve()) for r in split};opened=[];original=o.Image.open
def normal_only(path,*args,**kwargs):
    name=str(Path(path).resolve());assert name in allowed;opened.append(dict(path=name,utc=utc()));return original(path,*args,**kwargs)
o.Image.open=normal_only;rows=[];fields={k:[] for k in ['starting_controls','selected_controls','starting_field','selected_field']}
for i,(item,row,baseline) in enumerate(zip(split,f['rows'],old)):
    assert item['low']==row['low']==baseline['low'];root=a.out/f'{i:03d}';accepted=a.accepted/f'{i:03d}'
    for name,h in row['files'].items():assert sha(root/name)==h
    for name,h in pre['rows'][i]['prior_files'].items():assert sha(accepted/name)==h
    v=torch.load(root/'output.pt',weights_only=True,map_location='cpu');hist=torch.load(root/'history.pt',weights_only=True,map_location='cpu');start=torch.load(accepted/'output.pt',weights_only=True,map_location='cpu')
    assert hist['b'].shape==(501,1,1,8,8) and hist['mse'].shape==(501,) and torch.isfinite(hist['b']).all() and hist['b'].min()>=-.4 and hist['b'].max()<=.4
    step=int(hist['mse'].argmin());assert step==row['best_step'] and torch.equal(v['b'],hist['b'][step]) and torch.equal(hist['b'][0],start['b'])
    for k in ['raw','lift','q','u','knots','ev']:assert torch.equal(v[k],start[k]) and thash(v[k])==row[k+'_sha256']
    assert thash(v['image'])==row['output_sha256'] and thash(v['b'])==row['b_sha256'] and thash(v['offset'])==row['offset_sha256']
    assert sha(a.normal_root/item['normal'])==item['normal_sha256'];normal=native(a.normal_root/item['normal']);metrics=o.score(v['image'],normal);assert abs(metrics['mse']-float(hist['mse'][step]))<=1e-10
    h,w=v['offset'].shape[-2:];mask=torch.tensor(pre['rows'][i]['gate']['active']).reshape(2,2)[(torch.arange(h)>=h//2).long()[:,None],(torch.arange(w)>=w//2).long()[None,:]]
    fields['starting_controls'].extend(start['b'].flatten().tolist());fields['selected_controls'].extend(v['b'].flatten().tolist());fields['starting_field'].extend(start['offset'][0,0][mask].tolist());fields['selected_field'].extend(v['offset'][0,0][mask].tolist())
    rows.append(dict(index=i,low=item['low'],best_step=step,psnr=metrics['psnr'],ssim=metrics['ssim'],t052_psnr=baseline['psnr'],t052_ssim=baseline['ssim'],delta_psnr=metrics['psnr']-baseline['psnr'],delta_ssim=metrics['ssim']-baseline['ssim'],seconds=row['seconds']))
def stats(v):return dict(mean=float(np.mean(v)),median=float(np.median(v)))
def distribution(v):
    values=np.asarray(v,dtype=np.float64);d=dict(count=len(v),min=float(values.min()),max=float(values.max()),**stats(values));d['exact_hits']={str(bound):int(np.count_nonzero(values==float(np.float32(bound)))) for bound in [-.4,-.2,.2,.4]};return d
s=dict(label=LABEL,metrics={k:stats([r[k] for r in rows]) for k in ['psnr','ssim','t052_psnr','t052_ssim','delta_psnr','delta_ssim']},best_step_histogram=dict(collections.Counter(str(r['best_step']) for r in rows)),at_step500=sum(r['best_step']==500 for r in rows),win_equal_loss={k:dict(win=sum(r['delta_'+k]>0 for r in rows),equal=sum(r['delta_'+k]==0 for r in rows),loss=sum(r['delta_'+k]<0 for r in rows)) for k in ['psnr','ssim']},runtime_seconds=sum(r['seconds'] for r in rows),b_distributions={k:distribution(v) for k,v in fields.items()})
s['classification']=verdict(s['metrics']['delta_psnr']['mean'],s['metrics']['delta_psnr']['median'],s['metrics']['delta_ssim']['mean'])
write(a.out/'pairs.json',rows);write(a.out/'summary.json',s);write(a.out/'evaluation_receipt.json',dict(completed_utc=utc(),freeze_sha256=sha(a.out/'freeze.json'),metric_start_after_freeze=True,normal_opens=opened,prior_csv_sha256=sha(a.accepted/'pairs.json'),official_test_access=False));print(json.dumps(s),flush=True)
