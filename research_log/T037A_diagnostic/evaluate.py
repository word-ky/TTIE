"""Score all pre-frozen states; read only already-used T036 references."""
from core import *
import argparse,csv,time,os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import torch
from PIL import Image
from scripts.evaluate_t026a import pixels,independent_ssim
from ttie.ssim_transfer import rgb_ssim

def score_image(args):
    row,item,normal_root,out=args;torch.set_num_threads(1)
    target=Path(normal_root)/item['normal'];assert sha(target)==item['normal_sha256']
    # Hash both complete trajectories before this worker opens its reference.
    for record in row['methods'].values():assert sha(Path(out)/record['file'])==record['sha256']
    opened_utc=utc();normal=pixels(target).astype(np.float32).astype(np.float64)
    quality={};error=0.
    for method,record in row['methods'].items():
        frames=torch.load(Path(out)/record['file'],weights_only=True,map_location='cpu')['images'];assert frames.shape[0]==41
        quality[method]=dict(psnr=[],ssim=[])
        for image in frames:
            x=image[0].permute(1,2,0).numpy().astype(np.float64);delta=x-normal
            psnr=float(-10*np.log10(np.mean(delta*delta)));ssim=rgb_ssim(x,normal)
            independent=float(-10*np.log10(np.dot(delta.ravel(),delta.ravel())/delta.size))
            error=max(error,abs(psnr-independent),abs(ssim-independent_ssim(x,normal)))
            assert error<1e-11
            quality[method]['psnr'].append(psnr);quality[method]['ssim'].append(ssim)
        del frames
    return dict(index=row['index'],low=row['low'],quality=quality,normal=str(target),normal_open_utc=opened_utc,independent_error=error)

def main():
    p=argparse.ArgumentParser()
    for k in ['out','manifest','accepted','normal-root']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);start=time.perf_counter()
    assert sha(a.manifest)==COHORT and sha(a.accepted/'freeze.json')==PRIOR_FREEZE and sha(a.accepted/'metrics.csv')==PRIOR_METRICS
    f=json.loads((a.out/'freeze.json').read_bytes());manifest=json.loads(a.manifest.read_bytes())
    assert f['pairs']==100 and f['states']==8200 and f['normal_decodes']==f['optimizer_updates']==0
    assert f['cohort_sha256']==COHORT and f['prior_freeze_sha256']==PRIOR_FREEZE
    prior=list(csv.DictReader((a.accepted/'metrics.csv').open()))
    began=utc();write(a.out/'evaluation_start.json',dict(label=LABEL,utc=began,freeze_sha256=sha(a.out/'freeze.json'),workers=8))
    args=[]
    for row,item,old in zip(f['rows'],manifest['selected'],prior):
        assert row['low']==item['low']==old['low'];args.append((row,item,str(a.normal_root),str(a.out)))
    assert len(args)==100
    with ProcessPoolExecutor(max_workers=8) as pool:scored=list(pool.map(score_image,args))
    per_step=[];per_image=[];selected_error=0.
    for result,row,old in zip(scored,f['rows'],prior):
        selected={k:v['selected_step'] for k,v in row['methods'].items()};q=result['quality']
        for method,record in row['methods'].items():
            for metric in ['psnr','ssim']:
                selected_error=max(selected_error,abs(q[method][metric][selected[method]]-float(old[method+'_'+metric])))
            for step in range(41):per_step.append(dict(index=row['index'],low=row['low'],method=method,step=step,psnr=q[method]['psnr'][step],ssim=q[method]['ssim'][step],learned_energy=record['energies'][step]))
        per_image.append(image_diagnostic(row['index'],row['low'],q,selected))
    assert selected_error<1e-10
    for name,rows in [('per_step.csv',per_step),('per_image.csv',per_image)]:
        with (a.out/name).open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    curves={method:[dict(step=step,**{metric:stats([r[metric] for r in per_step if r['method']==method and r['step']==step]) for metric in ['psnr','ssim','learned_energy']}) for step in range(41)] for method in ['baseline','common']}
    headroom={m:stats([r[m+'_headroom'] for r in per_image]) for m in ['psnr','ssim']}
    losses={m:[r for r in per_image if r['prior_'+m+'_loss']] for m in ['psnr','ssim']}
    # Preserve precisely the accepted T036 loss cases, not a rounded substitute.
    assert {r['low'] for r in losses['psnr']}=={r['low'] for r in prior if float(r['delta_psnr'])<0} and len(losses['psnr'])==29
    assert {r['low'] for r in losses['ssim']}=={r['low'] for r in prior if float(r['delta_ssim'])<0} and len(losses['ssim'])==40
    rescues={m:sum(r['earlier_'+m+'_reaches_baseline'] for r in losses[m]) for m in ['psnr','ssim']}
    summary=dict(label=LABEL,classification=verdict(headroom['psnr']['mean'],rescues['psnr']),headroom=headroom,
        positive_headroom={m:sum(r[m+'_headroom']>0 for r in per_image) for m in ['psnr','ssim']},
        best_step_histograms={m:dict(Counter(str(r['common_reference_best_'+m+'_step']) for r in per_image)) for m in ['psnr','ssim']},
        prior_loss_counts={m:len(v) for m,v in losses.items()},earlier_rescued_loss_counts=rescues,
        earlier_reaches_baseline_all={m:sum(r['earlier_'+m+'_reaches_baseline'] for r in per_image) for m in ['psnr','ssim']},
        selected_metric_max_abs_error=selected_error,worst_case=next(r for r in per_image if r['low']=='Train/Low/low00559.png'))
    write(a.out/'summary.json',summary);write(a.out/'curves.json',curves);write(a.out/'prior_loss_cases.json',losses)
    write(a.out/'worst_case_trajectory.json',[r for r in per_step if r['low']=='Train/Low/low00559.png'])
    write(a.out/'evaluation_receipt.json',dict(label=LABEL,started_utc=began,completed_utc=utc(),freeze_sha256=sha(a.out/'freeze.json'),
        normal_opens=[dict(path=r['normal'],utc=r['normal_open_utc']) for r in scored],independent_metric_max_abs_error=max(r['independent_error'] for r in scored),
        selected_metric_max_abs_error=selected_error,scored_states=8200,workers=8,seconds=time.perf_counter()-start,optimizer_updates=0,deployable_state_changes=0,new_cohorts=0,official_test_access=False))
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
