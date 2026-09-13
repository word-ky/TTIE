"""Reference-only evaluation and independent audit after all100 output freezes."""
import argparse
import csv
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import convolve1d
from ttie.ssim_transfer import rgb_ssim, METRIC


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixels(path):
    with Image.open(path) as im:return np.asarray(im.convert('RGB'),dtype=np.float64)/255.


def independent_ssim(x,y):
    weights=np.exp(-np.arange(-5,6,dtype=np.float64)**2/4.5);weights/=weights.sum()
    def smooth(a):return convolve1d(convolve1d(a,weights,axis=0,mode='reflect'),weights,axis=1,mode='reflect')
    mx,my=smooth(x),smooth(y)
    return float(np.mean(((2*mx*my+.0001)*(2*(smooth(x*y)-mx*my)+.0009))/
                         ((mx*mx+my*my+.0001)*(smooth(x*x)-mx*mx+smooth(y*y)-my*my+.0009))))


def main():
    p=argparse.ArgumentParser()
    for k in ['audit','low-root','normal-root','split','deployment']:
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();frozen=json.loads((a.audit/'freeze.json').read_bytes());split=json.loads(a.split.read_bytes())
    deployment=json.loads(a.deployment.read_bytes())
    assert len(frozen['rows'])==len(split['selected'])==100
    assert sha(a.split)==frozen['split_sha256']
    assert sha(a.audit/'config.json')==frozen['config_sha256']
    assert datetime.fromisoformat(frozen['completed_utc'])<datetime.fromisoformat(deployment['reference_deployment_started_utc'])
    assert frozen['opened_images']==[str((a.low_root/r['low']).resolve()) for r in split['selected']]
    # Recheck every output, decision, trajectory and input identity before any reference decode.
    for r,s in zip(frozen['rows'],split['selected']):
        assert r['low']==s['low']
        for name,receipt in r['files'].items():assert sha(a.audit/f'{r["index"]:03d}'/name)==receipt['sha256']
        for root,key in [(a.low_root,'low'),(a.normal_root,'normal')]:assert sha(root/s[key])==s[key+'_sha256']
        d=json.loads((a.audit/f'{r["index"]:03d}'/'decision.json').read_bytes())
        scores=d['selection']['scores']
        assert d['selection']['selected_step']==min(range(len(scores)),key=lambda i:(scores[i],i))
        assert r['updates']==(40 if any(d['gate']['active']) else 0)
    rows=[];errors=[]
    for r,s in zip(frozen['rows'],split['selected']):
        normal=pixels(a.normal_root/s['normal']);raw=pixels(a.low_root/s['low'])
        # Raw baseline uses the exact float32 decoding used by adaptation.
        raw=raw.astype(np.float32).astype(np.float64)
        saved=torch.load(a.audit/f'{r["index"]:03d}'/'output.pt',map_location='cpu',weights_only=True)
        ours=saved['image'].squeeze(0).permute(1,2,0).numpy().astype(np.float64)
        normal=normal.astype(np.float32).astype(np.float64)
        assert ours.shape==raw.shape==normal.shape
        row=dict(index=r['index'],low=s['low'],normal=s['normal'],seconds=r['seconds'],active=r['active'],
                 updates=r['updates'],selected_step=r['selected_step'])
        for name,image in [('raw',raw),('ours',ours)]:
            assert np.isfinite(image).all()
            mse=np.mean((image-normal)**2)
            row[name+'_psnr']=float(-10*np.log10(mse));row[name+'_ssim']=rgb_ssim(image,normal)
            independent_psnr=float(-10*torch.log10(torch.from_numpy(image-normal).square().mean()))
            errors.extend([abs(row[name+'_psnr']-independent_psnr),abs(row[name+'_ssim']-independent_ssim(image,normal))])
        assert all(np.isfinite(row[k]) for k in ['raw_psnr','ours_psnr','raw_ssim','ours_ssim'])
        rows.append(row)
    assert max(errors)<1e-11
    with (a.audit/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(rows=100,ssim=METRIC,psnr='Per-image -10log10 RGB MSE, float64 arithmetic over float32 [0,1] pixels, no crop/resize',
                 metrics={k:dict(mean=float(np.mean([r[k] for r in rows])),median=float(np.median([r[k] for r in rows])))
                          for k in ['raw_psnr','ours_psnr','raw_ssim','ours_ssim']},
                 runtime_seconds=dict(mean=float(np.mean([r['seconds'] for r in rows])),median=float(np.median([r['seconds'] for r in rows])),
                                      p95=float(np.quantile([r['seconds'] for r in rows],.95,method='linear'))),
                 active=sum(r['active'] for r in rows),abstained=sum(not r['active'] for r in rows),
                 total_updates=sum(r['updates'] for r in rows),official_test_evaluated=False,verdict='benchmark-ready')
    (a.audit/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    audit=dict(passed=True,all100_pre_reference_hashes_verified=True,all100_native_shapes_and_metrics_finite=True,
               all_selected_steps_minimum_label_free_energy=True,reference_isolation='100 low-only opens; references deployed after all-output freeze',
               independent_metric_max_abs_error=max(errors),completed_utc=datetime.now(timezone.utc).isoformat())
    (a.audit/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(summary));print(json.dumps(audit))


if __name__=='__main__':main()
