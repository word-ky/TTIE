"""Attach qualification references only after both complete output freezes."""
import argparse,json,csv,hashlib
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import numpy as np
import torch
from PIL import Image
from scripts.evaluate_t026a import pixels,independent_ssim
from ttie.ssim_transfer import rgb_ssim
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def main():
    p=argparse.ArgumentParser()
    for k in ['audit','manifest','normal-root','deployment']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();f=json.loads((a.audit/'freeze.json').read_bytes());manifest=json.loads(a.manifest.read_bytes())
    deployment=json.loads(a.deployment.read_bytes());assert f['completed_utc']<deployment['started_utc']
    assert sha(a.audit/'freeze.json')==deployment['freeze_sha256'], 'audit freeze does not match reference deployment'
    assert sha(a.manifest)==f['manifest_sha256'] and sha(a.audit/'config.json')==f['config_sha256']
    replay=json.loads((a.audit/'independent_replay.json').read_bytes());assert replay['status']=='PASS' and replay['count']==100
    for r,s in zip(f['rows'],manifest['selected']):
        assert r['low']==s['low']
        for n,h in r['files'].items():assert sha(a.audit/f'{r["index"]:03d}'/n)==h['sha256']
        assert sha(a.normal_root/s['normal'])==s['normal_sha256']
    allowed={str((a.normal_root/r['normal']).resolve()) for r in manifest['selected']};opened=[];original_open=Image.open
    def reference_only(path,*args,**kwargs):
        name=str(Path(path).resolve());assert name in allowed;opened.append(name);return original_open(path,*args,**kwargs)
    Image.open=reference_only;rows=[];max_error=0.
    for r,s in zip(f['rows'],manifest['selected']):
        normal=pixels(a.normal_root/s['normal']);row=dict(index=r['index'],low=r['low'])
        for name in ['original','guarded']:
            saved=torch.load(a.audit/f'{r["index"]:03d}'/(name+'.pt'),map_location='cpu',weights_only=True)
            image=saved['image'].squeeze(0).permute(1,2,0).numpy().astype(np.float64)
            assert image.shape==normal.shape==(400,600,3) and np.isfinite(image).all()
            mse=float(np.mean((image-normal)**2));psnr=float(-10*np.log10(mse));ssim=rgb_ssim(image,normal)
            independent=float(-10*np.log10(np.dot((image-normal).ravel(),(image-normal).ravel())/image.size))
            error=max(abs(psnr-independent),abs(ssim-independent_ssim(image,normal)));max_error=max(max_error,error)
            assert error<1e-10
            row.update({name+'_psnr':psnr,name+'_ssim':ssim})
        row.update(delta_psnr=row['guarded_psnr']-row['original_psnr'],delta_ssim=row['guarded_ssim']-row['original_ssim'])
        rows.append(row)
    with (a.audit/'metrics.csv').open('w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def stats(values):return dict(mean=float(np.mean(values)),median=float(np.median(values)),p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9)))
    summary={k:stats([r[k] for r in rows]) for k in rows[0] if k not in ['index','low']}
    summary.update(classification='materially positive' if summary['delta_psnr']['mean']>=.3 and summary['delta_ssim']['mean']>=0 else 'negative/insufficient',
        win_equal_loss={m:dict(win=sum(r['delta_'+m]>0 for r in rows),equal=sum(r['delta_'+m]==0 for r in rows),loss=sum(r['delta_'+m]<0 for r in rows)) for m in ['psnr','ssim']},
        histograms={k:dict(Counter(str(r[k]) for r in f['rows'])) for k in ['cutoff','crossing','original_step','selected_step','fallback']},
        fallback_count=sum(r['fallback'] is not None for r in f['rows']),updates=sum(r['updates'] for r in f['rows']),
        runtime={k:stats([r[k] for r in f['rows']]) for k in ['trajectory_seconds','guard_seconds']})
    write(a.audit/'summary.json',summary)
    write(a.audit/'evaluation_receipt.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),pairs=100,
        normal_opens=opened,freeze_sha256=sha(a.audit/'freeze.json'),manifest_sha256=sha(a.manifest),
        independent_metric_max_abs_error=max_error,all_finite=True,official_test_access=False))
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
