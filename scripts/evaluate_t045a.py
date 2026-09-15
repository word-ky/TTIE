"""Post-freeze T026-convention metrics; accepted Ours CSV remains read-only."""
import argparse,json,csv,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from PIL import Image
from scripts.evaluate_t026a import pixels,independent_ssim
from ttie.ssim_transfer import rgb_ssim,METRIC
from scripts.bind_t033a import verify,sha

def main():
    p=argparse.ArgumentParser()
    for k in ['audit','split','normal-root','deployment','ours','binding']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();f=verify(a.audit,json.loads(a.deployment.read_bytes()));b=json.loads(a.binding.read_bytes())
    assert sha(a.split)==f['split_sha256'] and sha(a.binding)==f['binding_sha256'] and sha(a.audit/'config.json')==f['config_sha256']
    assert sha(a.ours)==b['ours_csv_sha256']
    split=json.loads(a.split.read_bytes())['selected'];ours=list(csv.DictReader(a.ours.open()));assert len(ours)==len(split)==100
    for r,s,o in zip(f['rows'],split,ours):
        assert r['low']==s['low']==o['low'] and s['normal']==o['normal']
        assert sha(a.audit/r['output'])==r['file_sha256'] and sha(a.normal_root/s['normal'])==s['normal_sha256']
    original=Image.open;allowed={str((a.normal_root/s['normal']).resolve()) for s in split};opened=[]
    def normal_only(path,*args,**kwargs):
        name=str(Path(path).resolve());assert name in allowed;opened.append(dict(path=name,utc=datetime.now(timezone.utc).isoformat()));return original(path,*args,**kwargs)
    Image.open=normal_only;rows=[];error=0.
    for r,s,o in zip(f['rows'],split,ours):
        # Accepted T026 evaluator rounds normal RGB pixels to float32 before float64 arithmetic.
        normal=pixels(a.normal_root/s['normal']).astype(np.float32).astype(np.float64)
        image=np.load(a.audit/r['output']).astype(np.float64);assert image.shape==normal.shape==(400,600,3)
        mse=float(np.mean((image-normal)**2));psnr=float(-10*np.log10(mse));ssim=rgb_ssim(image,normal)
        independent=-10*np.log10(np.dot((image-normal).ravel(),(image-normal).ravel())/image.size)
        error=max(error,abs(psnr-independent),abs(ssim-independent_ssim(image,normal)));assert error<1e-11
        rows.append(dict(index=r['index'],low=s['low'],snr_psnr=psnr,snr_ssim=ssim,ours_psnr=float(o['ours_psnr']),ours_ssim=float(o['ours_ssim']),
            delta_psnr=psnr-float(o['ours_psnr']),delta_ssim=ssim-float(o['ours_ssim']),seconds=r['runtime_s']))
    with (a.audit/'metrics.csv').open('w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def stats(v):return dict(mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.quantile(v,.95)))
    summary=dict(classification='SNR-Aware development benchmark complete',metrics={k:stats([r[k] for r in rows]) for k in ['snr_psnr','snr_ssim','ours_psnr','ours_ssim','delta_psnr','delta_ssim']},
        runtime=stats([r['seconds'] for r in rows]),peak_memory_bytes=max(r['peak_memory_bytes'] for r in f['rows']),
        win_equal_loss={m:dict(win=sum(r['delta_'+m]>0 for r in rows),equal=sum(r['delta_'+m]==0 for r in rows),loss=sum(r['delta_'+m]<0 for r in rows)) for m in ['psnr','ssim']},
        metric=METRIC,pixel_convention='native full RGB float32 [0,1] pixels promoted tofloat64; no crop/resize/Y/quantization',GT_mean=False,self_ensemble=False)
    (a.audit/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    assert sha(a.ours)==b['ours_csv_sha256']
    (a.audit/'evaluation_receipt.json').write_text(json.dumps(dict(completed_utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=sha(a.audit/'freeze.json'),deployment_sha256=sha(a.deployment),
        normal_opens=opened,ours_csv_sha256=sha(a.ours),independent_metric_max_abs_error=error,official_test_access=False),indent=2)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
