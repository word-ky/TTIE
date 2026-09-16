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
o.Image.open=normal_only;rows=[];segments=[];knot_values=[];starts=torch.load(a.preflight/'starts.pt',weights_only=True,map_location='cpu');initial_lifts=torch.load(a.preflight/'lifts.pt',weights_only=True,map_location='cpu')
for i,(item,row,baseline) in enumerate(zip(split,f['rows'],old)):
    assert item['low']==row['low']==baseline['low'];root=a.out/f'{i:03d}'
    for name,h in row['files'].items():assert sha(root/name)==h
    saved=torch.load(root/'output.pt',weights_only=True,map_location='cpu');hist=torch.load(root/'history.pt',weights_only=True,map_location='cpu')
    assert hist['q'].shape==(1001,4,8) and hist['mse'].shape==(1001,)
    index=int(hist['mse'].argmin());assert index==row['best_step'] and torch.equal(saved['q'],hist['q'][index]) and torch.equal(saved['lift'],initial_lifts[i]) and thash(saved['raw'])==pre['rows'][i]['raw_sha256'] and thash(saved['lift'])==row['lift_sha256'] and thash(saved['q'])==row['q_sha256'] and thash(saved['knots'])==row['knots_sha256']
    assert thash(saved['raw'])==row['raw_sha256'] and thash(saved['image'])==row['output_sha256']
    assert sha(a.normal_root/item['normal'])==item['normal_sha256'];normal=native(a.normal_root/item['normal']);v=o.score(saved['image'],normal)
    assert abs(v['mse']-float(hist['mse'][index]))<=1e-10
    active=torch.tensor(pre['rows'][i]['gate']['active']);y=saved['knots'][active];assert (y.diff(dim=-1)>=0).all();segments.extend(y.diff(dim=-1).flatten().tolist());knot_values.extend(y[:,1:-1].flatten().tolist())
    rows.append(dict(index=i,low=item['low'],best_step=index,psnr=v['psnr'],ssim=v['ssim'],t049_psnr=float(baseline['psnr']),t049_ssim=float(baseline['ssim']),delta_psnr=v['psnr']-float(baseline['psnr']),delta_ssim=v['ssim']-float(baseline['ssim']),total_vs_t048_psnr=v['psnr']-float(baseline['t048_psnr']),total_vs_t048_ssim=v['ssim']-float(baseline['t048_ssim']),seconds=row['seconds']))
def stats(v):return dict(mean=float(np.mean(v)),median=float(np.median(v)))
s=dict(label=LABEL,metrics={k:stats([r[k] for r in rows]) for k in ['psnr','ssim','t049_psnr','t049_ssim','delta_psnr','delta_ssim','total_vs_t048_psnr','total_vs_t048_ssim']},best_step_histogram=dict(collections.Counter(str(r['best_step']) for r in rows)),at_step1000=sum(r['best_step']==1000 for r in rows),win_equal_loss={k:dict(win=sum(r['delta_'+k]>0 for r in rows),equal=sum(r['delta_'+k]==0 for r in rows),loss=sum(r['delta_'+k]<0 for r in rows)) for k in ['psnr','ssim']},runtime_seconds=sum(r['seconds'] for r in rows))
s['tone_distribution']=dict(active_curves=len(segments)//8,segment_count=len(segments),segment_min=min(segments),segment_max=max(segments),segment_mean=float(np.mean(segments)),segment_median=float(np.median(segments)),zero_width_segments=sum(v==0 for v in segments),interior_knot_min=min(knot_values),interior_knot_max=max(knot_values),interior_knot_mean=float(np.mean(knot_values)),interior_knot_median=float(np.median(knot_values)),fixed_endpoints=[0,1])
s['classification']=verdict(s['metrics']['delta_psnr']['mean'],s['metrics']['delta_psnr']['median'])
write(a.out/'pairs.json',rows);write(a.out/'summary.json',s);write(a.out/'evaluation_receipt.json',dict(completed_utc=utc(),freeze_sha256=sha(a.out/'freeze.json'),metric_start_after_freeze=True,normal_opens=opened,prior_csv_sha256=sha(a.accepted/'pairs.json'),official_test_access=False));print(json.dumps(s),flush=True)
