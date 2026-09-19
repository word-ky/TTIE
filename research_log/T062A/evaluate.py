"""Separate post-freeze reference evaluation and independent objective verification."""
import argparse,csv,json,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from scripts.evaluate_t026a import pixels,independent_ssim
from ttie.ssim_transfer import rgb_ssim
from research_log.T062A.infer import ROOT,MANIFEST,COHORT,sha,utc,write,tensor_hash

def independent_loss(x,y):
    def pool(a,k):
        h,w=a.shape[-2:];h=h//k*k;w=w//k*k
        return a[...,:h,:w].reshape(1,3,h//k,k,w//k,k).mean(axis=(3,5))
    a=pool(x,4);b=pool(y,4)
    dh=np.diff(b,axis=-1)-np.diff(a,axis=-1);dv=np.diff(b,axis=-2)-np.diff(a,axis=-2)
    spa=(np.abs(dh).sum()+np.abs(dv).sum())/(dh.size+dv.size)
    exp=np.mean((pool(y,16).mean(axis=1)-.6)**2)
    mu=y.mean(axis=(-2,-1))[0];col=sum((mu[i]-mu[j])**2 for i,j in [(0,1),(0,2),(1,2)])
    return float(spa+10*exp+5*col)

def main(out):
    torch.set_num_threads(1);start=time.perf_counter()
    f=json.loads((out/'freeze.json').read_bytes());assert len(f['rows'])==100 and sha(MANIFEST)==COHORT
    assert f['config_sha256']==sha(out/'config.json')
    for row in f['rows']:
        for n,h in row['files'].items():assert sha(out/f"{row['index']:03d}"/n)==h
    first=utc();assert f['completed_utc']<first
    write(out/'reference_open.json',dict(first_reference_or_baseline_read_utc=first,freeze_sha256=sha(out/'freeze.json')))
    per=Path('research_log/T037A_result/per_image.csv')
    assert sha(per)=='5676541d245fdb41c54a543cf88a79ca37640eea94432af0665cca2ddce2277c'
    baseline=list(csv.DictReader(per.open()));manifest=json.loads(MANIFEST.read_bytes())['selected']
    rows=[];errors=[];loss_errors=[];checks=[]
    for rec,b,m in zip(f['rows'],baseline,manifest):
        i=rec['index'];assert rec['low']==b['low']==m['low'] and int(b['index'])==i
        d=out/f'{i:03d}';t=torch.load(d/'trace.pt',map_location='cpu',weights_only=True)
        output=torch.load(d/'output.pt',map_location='cpu',weights_only=True);y=output['image'];x=output['low']
        assert t['states'].shape==(41,1,3,2,2) and torch.count_nonzero(t['states'][0])==0
        assert torch.isfinite(t['states']).all() and len(t['values'])==41
        selected=int(np.argmin(np.asarray(t['values'],dtype=np.float64)));assert selected==rec['selected_step']==t['selected_step']
        assert tensor_hash(y)==rec['rendered_hashes'][selected]
        loss=independent_loss(x.numpy().astype(np.float64),y.numpy().astype(np.float64))
        error=abs(loss-t['values'][selected]);assert error<1e-5;loss_errors.append(error)
        target=ROOT/'shared/t036a/normal'/m['normal'];assert sha(target)==m['normal_sha256']
        normal=pixels(target).astype(np.float32).astype(np.float64);image=y[0].permute(1,2,0).numpy().astype(np.float64)
        delta=image-normal;psnr=float(-10*np.log10(np.mean(delta**2)));ssim=rgb_ssim(image,normal)
        other=float(-10*np.log10(np.dot(delta.ravel(),delta.ravel())/delta.size));ssim2=independent_ssim(image,normal)
        metric_error=max(abs(psnr-other),abs(ssim-ssim2));assert metric_error<1e-11;errors.append(metric_error)
        r=dict(index=i,low=m['low'],selected_step=selected,psnr=psnr,ssim=ssim,t036_psnr=float(b['common_selected_psnr']),t036_ssim=float(b['common_selected_ssim']),t026_psnr=float(b['baseline_selected_psnr']))
        r.update(delta_psnr=psnr-r['t036_psnr'],delta_ssim=ssim-r['t036_ssim'],delta_t026=psnr-r['t026_psnr']);rows.append(r)
        checks.append(dict(index=i,selected_step=selected,independent_objective=loss,objective_error=error,metric_error=metric_error))
    assert len(rows)==100
    p=np.array([r['delta_psnr'] for r in rows]);s=np.array([r['delta_ssim'] for r in rows]);b=np.array([r['delta_t026'] for r in rows])
    gates=dict(mean_psnr=float(p.mean())>=.5,regressions=int((b<0).sum())<=29,worst=float(b.min())>=-5.614,mean_ssim=float(s.mean())>=-.001)
    result=dict(task='T062-A',verdict='PASS' if all(gates.values()) else 'NEGATIVE',classification='a target-time zero-reference objective materially improves the T036 action-space control' if all(gates.values()) else 'the fixed three-term zero-reference objective is insufficient',gates=gates,
        mean_psnr=float(np.mean([r['psnr'] for r in rows])),mean_ssim=float(np.mean([r['ssim'] for r in rows])),
        mean_delta_psnr=float(p.mean()),median_delta_psnr=float(np.median(p)),mean_delta_ssim=float(s.mean()),
        improve=int((p>0).sum()),regress=int((p<0).sum()),tie=int((p==0).sum()),regressions_t026=int((b<0).sum()),worst_delta_t026=float(b.min()),
        selected_step_histogram=dict(Counter(str(r['selected_step']) for r in rows)),inference_seconds=sum(r['seconds'] for r in f['rows']),evaluation_seconds=time.perf_counter()-start,
        freeze_sha256=sha(out/'freeze.json'),freeze_utc=f['completed_utc'],first_reference_read_utc=first,completed_utc=utc())
    # Independent aggregate replay uses scalar Python arithmetic, not NumPy reductions.
    independent_mean=sum(r['psnr']-r['t036_psnr'] for r in rows)/100
    independent_ssim_mean=sum(r['ssim']-r['t036_ssim'] for r in rows)/100
    independent_gates=dict(mean_psnr=independent_mean>=.5,regressions=sum(r['psnr']<r['t026_psnr'] for r in rows)<=29,worst=min(r['psnr']-r['t026_psnr'] for r in rows)>=-5.614,mean_ssim=independent_ssim_mean>=-.001)
    assert independent_gates==gates and abs(independent_mean-result['mean_delta_psnr'])<1e-12
    write(out/'paired.json',rows);write(out/'verification.json',dict(status='PASS',checks=checks,max_objective_error=max(loss_errors),max_metric_error=max(errors),aggregate_error=abs(independent_mean-result['mean_delta_psnr']),gates=independent_gates))
    write(out/'result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
