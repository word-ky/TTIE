"""Separate postfreeze reference evaluator with the exact T026 pixel convention."""
import argparse,json,csv
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from PIL import Image
from scripts.run_t036a import sha,write,utc
from scripts.evaluate_t026a import pixels,independent_ssim
from ttie.ssim_transfer import rgb_ssim
p=argparse.ArgumentParser()
for k in ['audit','manifest','normal-root','deployment']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();torch.set_num_threads(1)
f=json.loads((a.audit/'freeze.json').read_bytes());m=json.loads(a.manifest.read_bytes());deployment=json.loads(a.deployment.read_bytes())
assert sha(a.audit/'freeze.json')==deployment['freeze_sha256'] and f['completed_utc']<deployment['started_utc']
assert sha(a.manifest)==f['manifest_sha256']==deployment['manifest_sha256'] and sha(a.audit/'config.json')==f['config_sha256']
assert json.loads((a.audit/'independent_replay.json').read_bytes())['status']=='PASS'
allowed={str((a.normal_root/r['normal']).resolve()) for r in m['selected']};opened=[];original=Image.open
def normal_only(path,*args,**kw):
    name=str(Path(path).resolve());assert name in allowed;opened.append(name);return original(path,*args,**kw)
Image.open=normal_only;rows=[];bounds=[];error=0.
for r,s in zip(f['rows'],m['selected']):
    assert r['low']==s['low'] and sha(a.normal_root/s['normal'])==s['normal_sha256']
    normal=pixels(a.normal_root/s['normal']).astype(np.float32).astype(np.float64);row=dict(index=r['index'],low=r['low'])
    for name,record in r['methods'].items():
        d=a.audit/f'{r["index"]:03d}'/name
        for n,h in record['files'].items():assert sha(d/n)==h['sha256']
        image=torch.load(d/'output.pt',weights_only=True,map_location='cpu')['image'][0].permute(1,2,0).numpy().astype(np.float64)
        mse=float(np.mean((image-normal)**2));psnr=float(-10*np.log10(mse));ssim=rgb_ssim(image,normal)
        error=max(error,abs(psnr-float(-10*np.log10(np.dot((image-normal).ravel(),(image-normal).ravel())/image.size))),abs(ssim-independent_ssim(image,normal)))
        assert error<1e-11
        row.update({name+'_psnr':psnr,name+'_ssim':ssim,name+'_step':record['selected_step'],name+'_seconds':record['seconds']})
        dec=json.loads((d/'decision.json').read_bytes());fields=torch.load(d/'fast_fields.pt',weights_only=True,map_location='cpu')['selected']
        lo=torch.tensor(dec['diagnostics']['action_box']['lower'])[0];hi=torch.tensor(dec['diagnostics']['action_box']['upper'])[0]
        for ch,channel in enumerate(['ev','gamma','gain']):
            for y in range(2):
                for x in range(2):
                    val=float(fields[ch,y,x]);lower=float(lo[ch,y,x]) if ch<2 else .5;upper=float(hi[ch,y,x]) if ch<2 else 2.
                    bounds.append(dict(index=r['index'],method=name,channel=channel,region=f'{y}{x}',active=dec['gate']['active'][2*y+x],value=val,lower_hit=abs(val-lower)<=1e-6,upper_hit=abs(val-upper)<=1e-6))
    row.update(delta_psnr=row['common_psnr']-row['baseline_psnr'],delta_ssim=row['common_ssim']-row['baseline_ssim']);rows.append(row)
assert len(rows)==len(opened)==100
with (a.audit/'metrics.csv').open('w',newline='') as file:
    w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stats(v):return dict(count=len(v),mean=float(np.mean(v)),median=float(np.median(v)),min=float(min(v)),max=float(max(v)),p05=float(np.quantile(v,.05)),p95=float(np.quantile(v,.95))) if v else dict(count=0)
metrics={k:stats([r[k] for r in rows]) for k in rows[0] if k not in ['index','low','baseline_step','common_step']}
distributions={}
for method in ['baseline','common']:
    distributions[method]={}
    for channel in ['ev','gamma','gain']:
        distributions[method][channel]={}
        for region in ['00','01','10','11','all']:
            distributions[method][channel][region]={}
            for group in ['active','inactive','all']:
                b=[r for r in bounds if r['method']==method and r['channel']==channel and (region=='all' or r['region']==region) and (group=='all' or r['active']==(group=='active'))]
                distributions[method][channel][region][group]=dict(**stats([r['value'] for r in b]),lower_hits=sum(r['lower_hit'] for r in b),upper_hits=sum(r['upper_hit'] for r in b))
summary=dict(metrics=metrics,classification='materially positive' if metrics['delta_psnr']['mean']>=.3 and metrics['delta_ssim']['mean']>=0 else 'negative/insufficient',
    win_equal_loss={m:dict(win=sum(r['delta_'+m]>0 for r in rows),equal=sum(r['delta_'+m]==0 for r in rows),loss=sum(r['delta_'+m]<0 for r in rows)) for m in ['psnr','ssim']},
    selected_step_histograms={m:dict(Counter(str(r[m+'_step']) for r in rows)) for m in ['baseline','common']},bounds_and_gains=distributions)
write(a.audit/'summary.json',summary);write(a.audit/'bound_values.json',bounds)
write(a.audit/'evaluation_receipt.json',dict(completed_utc=utc(),normal_opens=opened,freeze_sha256=sha(a.audit/'freeze.json'),deployment_sha256=sha(a.deployment),
    independent_metric_max_abs_error=error,metric='exact T026: native RGB float32/255 then float64 PSNR/RGB-SSIM, no crop/resize/quantization',official_test_access=False))
print(summary['classification'],metrics['delta_psnr'],metrics['delta_ssim'],flush=True)
