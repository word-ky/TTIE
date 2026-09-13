"""T022-B: render frozen states only, then stream offline validation metrics."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .lolv2_core import native_rgb, write
from .semantic_ttt import Region2
from .ssim_transfer import rgb_ssim, METRIC
from .stop_receipt import sha

SPLIT_SHA='b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b'


def render(model,image,state):
    with torch.no_grad():
        model.raw.copy_(state.to(image))
        return model(image)


def saturation(grid,lower,upper):
    result={}
    for channel,name in enumerate(['ev','gamma']):
        low=(grid[0,channel]-lower[0,channel]).abs()<=1e-6
        high=(grid[0,channel]-upper[0,channel]).abs()<=1e-6
        result[name+'_lower']=float(low.float().mean())
        result[name+'_upper']=float(high.float().mean())
        result[name+'_either']=float((low|high).float().mean())
    return result


def stats(values):
    a=np.asarray(values,dtype=np.float64)
    return dict(mean=float(a.mean()),median=float(np.median(a)),p95=float(np.quantile(a,.95,method='linear')))


def summarize(rows,selections):
    by_image={i:[r for r in rows if r['index']==i] for i in range(100)}
    fixed=[]; per_image=[]
    for k in range(41):
        values=[r for r in rows if r['step']==k]
        fixed.append(dict(step=k,**{m:stats([r[m] for r in values]) for m in ['psnr','ssim']}))
    groups={name:[] for name in ['selected','psnr_oracle','ssim_oracle']}
    for i,values in by_image.items():
        indices=dict(selected=selections[i],psnr_oracle=max(range(41),key=lambda k:(values[k]['psnr'],-k)),
                     ssim_oracle=max(range(41),key=lambda k:(values[k]['ssim'],-k)))
        item=dict(index=i,low=values[0]['low'])
        for name,k in indices.items():
            r=values[k]; groups[name].append(r);item[name]=r
        per_image.append(item)
    aggregates={name:{m:stats([r[m] for r in values]) for m in ['psnr','ssim']} for name,values in groups.items()}
    headroom={}
    for name in ['psnr_oracle','ssim_oracle']:
        headroom[name]={m:stats([a[m]-b[m] for a,b in zip(groups[name],groups['selected'])]) for m in ['psnr','ssim']}
        headroom[name]['selected_equals_oracle_fraction']=sum(a['step']==b['step'] for a,b in zip(groups[name],groups['selected']))/100
    histograms={name:np.bincount([r['step'] for r in values],minlength=41).tolist() for name,values in groups.items()}
    bounds={name:{k:float(np.mean([r[k] for r in values])) for k in ['ev_lower','ev_upper','ev_either','gamma_lower','gamma_upper','gamma_either']}
            for name,values in groups.items()}
    summary=dict(status='audit-complete',images=100,checkpoint_rows=4100,metrics=METRIC,
                 aggregates=aggregates,headroom=headroom,step_histograms=histograms,saturation=bounds,
                 saturation_definition='Fraction of four coordinates per channel within1e-6 of saved physical bounds; either counts union, includes collapsed inactive bounds',
                 best_global_fixed_step={m:max(range(41),key=lambda k:(fixed[k][m]['mean'],-k)) for m in ['psnr','ssim']},
                 oracle_scope='Offline validation-only reference oracle, not a deployable selector',official_test_used=False)
    return summary,fixed,per_image


def main():
    p=argparse.ArgumentParser()
    for k in ['original','low-root','normal-root','split','freeze','out']:
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert sha(a.split)==SPLIT_SHA
    split=json.loads(a.split.read_bytes());frozen=json.loads(a.freeze.read_bytes())
    assert len(split['selected'])==len(frozen['rows'])==100
    assert sha(a.original/'freeze.json')==sha(a.freeze)
    assert sha(a.original/'config.json')==frozen['config_sha256']
    for row,chosen in zip(frozen['rows'],split['selected']):
        assert row['low']==chosen['low'] and row['updates']==40
        for name,receipt in row['files'].items():
            path=a.original/f'{row["index"]:03d}'/name
            assert path.stat().st_size==receipt['bytes'] and sha(path)==receipt['sha256']
        assert sha(a.low_root/chosen['low'])==chosen['low_sha256']
        assert sha(a.normal_root/chosen['normal'])==chosen['normal_sha256']
    write(a.out/'provenance.json',dict(freeze_sha256=sha(a.freeze),split_sha256=SPLIT_SHA,
          original_root=str(a.original),rows=frozen['rows'],all_hashes_exact=True,
          completed_before_reference_pixels_utc=datetime.now(timezone.utc).isoformat(),
          source_sha256=sha(Path(__file__)),gpu=torch.cuda.get_device_name(0),torch=torch.__version__))
    # Complete selected-state reconstruction for ALL images before any normal decode.
    reconstruction=[]
    for row,chosen in zip(frozen['rows'],split['selected']):
        directory=a.original/f'{row["index"]:03d}'
        t=torch.load(directory/'trajectory.pt',map_location='cpu',weights_only=True)
        d=json.loads((directory/'decision.json').read_bytes())
        original=torch.load(directory/'output.pt',map_location='cpu',weights_only=True)['image']
        assert t['states'].shape==(41,1,2,2,2)
        image=native_rgb(a.low_root/chosen['low']).cuda()
        model=Region2(torch.tensor(d['gate']['active'])).to(image).requires_grad_(False)
        recovered=render(model,image,t['states'][row['selected_step']]).cpu()
        error=(recovered-original).abs()
        reconstruction.append(dict(index=row['index'],max_abs=float(error.max()),mean_abs=float(error.double().mean())))
        assert reconstruction[-1]['max_abs']<=1e-6
        del original,recovered,error
    write(a.out/'reconstruction.json',dict(rows=reconstruction,max_abs=max(r['max_abs'] for r in reconstruction),
          mean_abs=float(np.mean([r['mean_abs'] for r in reconstruction])),passed=True,
          completed_before_reference_pixels_utc=datetime.now(timezone.utc).isoformat()))
    print('All100 selected-state reconstructions pass before references',flush=True)
    rows=[]
    with (a.out/'checkpoint_metrics.csv').open('w',newline='') as f:
        writer=None
        for row,chosen in zip(frozen['rows'],split['selected']):
            directory=a.original/f'{row["index"]:03d}'
            t=torch.load(directory/'trajectory.pt',map_location='cpu',weights_only=True)
            d=json.loads((directory/'decision.json').read_bytes())
            image=native_rgb(a.low_root/chosen['low']).cuda()
            reference=native_rgb(a.normal_root/chosen['normal'])[0].permute(1,2,0).numpy().astype(np.float64)
            model=Region2(torch.tensor(d['gate']['active'])).to(image).requires_grad_(False)
            lower=torch.tensor(d['diagnostics']['action_box']['lower']);upper=torch.tensor(d['diagnostics']['action_box']['upper'])
            for k in range(41):
                output=render(model,image,t['states'][k])[0].permute(1,2,0).cpu().numpy().astype(np.float64)
                r=dict(index=row['index'],low=chosen['low'],step=k,psnr=float(-10*np.log10(np.mean((output-reference)**2))),
                       ssim=rgb_ssim(output,reference),**saturation(t['grids'][k],lower,upper))
                assert np.isfinite(r['psnr']) and np.isfinite(r['ssim'])
                if writer is None:writer=csv.DictWriter(f,fieldnames=list(r));writer.writeheader()
                writer.writerow(r);rows.append(r)
            f.flush();print(f'{row["index"]+1}/100 trajectories evaluated',flush=True)
    summary,fixed,per_image=summarize(rows,[r['selected_step'] for r in frozen['rows']])
    write(a.out/'summary.json',summary);write(a.out/'global_fixed_steps.json',fixed);write(a.out/'per_image_oracles.json',per_image)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
