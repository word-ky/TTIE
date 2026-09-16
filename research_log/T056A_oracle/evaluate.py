"""Frozen T056 outputs versus accepted T055, reference-only evaluation."""
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
o.Image.open=normal_only;rows=[];fields={k:[] for k in ['selected_controls','selected_field']}
for i,(item,row,baseline) in enumerate(zip(split,f['rows'],old)):
    assert item['low']==row['low']==baseline['low'];root=a.out/f'{i:03d}';accepted=a.accepted/f'{i:03d}'
    for name,h in row['files'].items():assert sha(root/name)==h
    for name,h in pre['rows'][i]['prior_files'].items():assert sha(accepted/name)==h
    v=torch.load(root/'output.pt',weights_only=True,map_location='cpu');hist=torch.load(root/'history.pt',weights_only=True,map_location='cpu');oldstate=torch.load(accepted/'output.pt',weights_only=True,map_location='cpu')
    assert hist['w'].shape==(501,1,1,8,8) and hist['mse'].shape==(501,) and torch.isfinite(hist['w']).all() and torch.count_nonzero(hist['w'][0])==0
    step=int(hist['mse'].argmin());assert step==row['best_step'] and torch.equal(v['w'],hist['w'][step])
    for name in OLD:assert torch.equal(v[name],oldstate[name]),name
    for name,h in row['hashes'].items():assert thash(v[name])==h
    assert torch.equal(v['one'],oldstate['image']) and thash(v['d2'])==pre['rows'][i]['d2_sha256']
    assert sha(a.normal_root/item['normal'])==item['normal_sha256'];normal=native(a.normal_root/item['normal']);metrics=o.score(v['image'],normal);assert abs(metrics['mse']-float(hist['mse'][step]))<=1e-10
    fields['selected_controls'].extend(v['c2_controls'].flatten().tolist());fields['selected_field'].extend(v['c2'].flatten().tolist())
    rows.append(dict(index=i,low=item['low'],best_step=step,psnr=metrics['psnr'],ssim=metrics['ssim'],t055_psnr=baseline['psnr'],t055_ssim=baseline['ssim'],delta_psnr=metrics['psnr']-baseline['psnr'],delta_ssim=metrics['ssim']-baseline['ssim'],seconds=row['seconds']))
def stats(v):return dict(mean=float(np.mean(v)),median=float(np.median(v)))
def distribution(v):
    values=np.asarray(v,dtype=np.float64);d=dict(count=len(v),min=float(values.min()),max=float(values.max()),**stats(values));d['exact_hits']={str(bound):int(np.count_nonzero(values==float(np.float32(bound)))) for bound in [-1.,1.]};d['fraction_le_neg099']=float(np.mean(values<=float(np.float32(-.99))));d['fraction_ge_pos099']=float(np.mean(values>=float(np.float32(.99))));return d
s=dict(label=LABEL,metrics={k:stats([r[k] for r in rows]) for k in ['psnr','ssim','t055_psnr','t055_ssim','delta_psnr','delta_ssim']},best_step_histogram=dict(collections.Counter(str(r['best_step']) for r in rows)),at_step500=sum(r['best_step']==500 for r in rows),win_equal_loss={k:dict(win=sum(r['delta_'+k]>0 for r in rows),equal=sum(r['delta_'+k]==0 for r in rows),loss=sum(r['delta_'+k]<0 for r in rows)) for k in ['psnr','ssim']},runtime_seconds=sum(r['seconds'] for r in rows),c2_distributions={k:distribution(v) for k,v in fields.items()})
s['classification']=verdict(s['metrics']['delta_psnr']['mean'],s['metrics']['delta_psnr']['median'],s['metrics']['delta_ssim']['mean'])
write(a.out/'pairs.json',rows);write(a.out/'summary.json',s);write(a.out/'evaluation_receipt.json',dict(completed_utc=utc(),freeze_sha256=sha(a.out/'freeze.json'),metric_start_after_freeze=True,normal_opens=opened,prior_csv_sha256=sha(a.accepted/'pairs.json'),official_test_access=False));print(json.dumps(s),flush=True)
