"""Independent scalar objective progress/grid/selection and offline metric audit; no feedback to fitting."""
import argparse, json, math, statistics
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from scripts.evaluate_t026a import pixels, independent_ssim
from research_log.T063C.run import (ROOT, DEV, HERE, DEV_MANIFEST, TARGET_MANIFEST, TARGET,
    load_inputs, state_input, render_components, development_rows, score_transfer)
from research_log.T063A.common import sha, thash, write, utc


def independent_progress(values):
    best=min(values);reduction=values[0]-best
    q=[min(1.,max(0.,(values[0]-v)/max(reduction,1e-12))) for v in values]
    return best,reduction,q


def independent_steps(trajectories,rho):
    choices=[]
    for values in trajectories:
        _,d,_=independent_progress(values);selected=0
        if math.isfinite(d) and d>1e-12:
            bound=values[0]-rho*d
            for k,v in enumerate(values):
                if math.isfinite(v) and v<=bound:
                    selected=k;break
        choices.append(selected)
    return choices


def independent_summary(rows,steps):
    d=[];b=[];s=[]
    for row,k in zip(rows,steps):
        d.append(row['psnr'][k]-row['t036_psnr']);b.append(row['psnr'][k]-row['t026_psnr']);s.append(row['ssim'][k]-row['t036_ssim'])
    result=dict(mean_delta_psnr=math.fsum(d)/len(d),median_delta_psnr=statistics.median(d),
                regressions_t026=sum(x<0 for x in b),worst_delta_t026=min(b),mean_delta_ssim=math.fsum(s)/len(s))
    gates=dict(mean_psnr=result['mean_delta_psnr']>=2,median_psnr=result['median_delta_psnr']>0,
        regressions=result['regressions_t026']<=29,worst=result['worst_delta_t026']>=-5.614,mean_ssim=result['mean_delta_ssim']>=-.001)
    return result,gates


def independent_development(job):
    rec,item,prior=job;torch.set_num_threads(1);p=DEV/f"{rec['index']:03d}"/'images.pt'
    assert sha(p)==rec['files']['images.pt'];images=torch.load(p,weights_only=True,map_location='cpu')[:28]
    normalpath=ROOT/'shared/t036a/normal'/item['normal'];assert sha(normalpath)==item['normal_sha256']
    normal=pixels(normalpath).astype(np.float32).astype(np.float64);psnr=[];ssim=[]
    for k,image in enumerate(images):
        assert thash(image)==rec['rendered_hashes'][k]
        x=image[0].permute(1,2,0).numpy().astype(np.float64);d=x-normal
        psnr.append(-10*math.log10(float(d.ravel()@d.ravel())/d.size));ssim.append(independent_ssim(x,normal))
    error=max(max(abs(a-b) for a,b in zip(psnr,prior['psnr'])),max(abs(a-b) for a,b in zip(ssim,prior['ssim'])))
    assert error<1e-10
    return dict(prior,psnr=psnr,ssim=ssim,independent_error=error)


def check_render(transfer,records):
    raw,items,recs=load_inputs(transfer);allvalues=[]
    for i,(rec,reported) in enumerate(zip(recs,records)):
        low,trace=state_input(raw,rec,transfer);images,parts,totals=render_components(low,trace)
        assert [thash(v) for v in images]==reported['hashes']
        assert parts==reported['components']
        assert totals==reported['totals']
        np.testing.assert_allclose(totals,[v[0]+10*v[1]+5*v[2] for v in parts],rtol=1e-6,atol=1e-8)
        best,d,q=independent_progress(totals)
        assert best==reported['best'] and d==reported['reduction']
        np.testing.assert_array_equal(q,reported['progress']);allvalues.append(totals)
        print('verified-transfer' if transfer else 'verified-development',i+1,flush=True)
    return allvalues


def main(out):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rule=json.loads((out/'selector_manifest.json').read_bytes());result=json.loads((out/'result.json').read_bytes())
    for p,h in rule['source_binding'].items():assert sha(p)==h,p
    assert sha(out/'development_components.json')==rule['components_sha256'] and sha(out/'threshold_candidates.json')==rule['table_sha256']
    dev=json.loads((out/'development_components.json').read_bytes());r=check_render(False,dev['rows'])
    raw,items,recs=load_inputs(False);prior=development_rows()
    with ProcessPoolExecutor(max_workers=8) as pool:rows=list(pool.map(independent_development,zip(recs,items,prior)))
    grid=sorted({0.,1.}|{q for values in r for q in independent_progress(values)[2] if math.isfinite(q)})
    table=json.loads((out/'threshold_candidates.json').read_bytes());assert grid==[row['rho'] for row in table]
    best=None;maxerror=0.
    for rho,reported in zip(grid,table):
        steps=independent_steps(r,rho);m,g=independent_summary(rows,steps)
        error=max(abs(m[k]-reported[k]) for k in m);maxerror=max(maxerror,error);assert error<1e-10
        assert g==reported['gates']
        if all(g[k] for k in ['regressions','worst','mean_ssim']):
            if best is None or m['mean_delta_psnr']>best['mean_delta_psnr']:
                best=dict(rho=rho,**m,gates=g)
    passed=best is not None and all(best['gates'].values())
    assert passed==rule['calibration_pass'] and (best['rho'] if best else None)==rule['rho']
    assert (independent_steps(r,best['rho']) if best else [])==rule['development_selected_steps']
    verification=dict(status='PASS',candidate_count=len(grid),rho=rule['rho'],calibration_pass=passed,
        development_renders=2800,development_metric_max_error=max(v['independent_error'] for v in rows),candidate_max_error=maxerror)
    if not passed:
        assert result['classification']=='CALIBRATION_NEGATIVE' and not (out/'transfer_freeze.json').exists()
        verification.update(classification='CALIBRATION_NEGATIVE',transfer_renders=0,verified_utc=utc())
        write(out/'verification.json',verification);print(json.dumps(verification),flush=True);return
    freeze=json.loads((out/'transfer_freeze.json').read_bytes());marker=json.loads((out/'reference_open.json').read_bytes())
    assert freeze['selector_sha256']==sha(out/'selector_manifest.json')==result['selector_sha256']
    assert marker['transfer_freeze_sha256']==sha(out/'transfer_freeze.json')==result['transfer_freeze_sha256']
    assert rule['frozen_utc']<freeze['completed_utc']<marker['first_reference_or_quality_read_utc']
    allowed={str((TARGET/f'{i:03d}'/n).resolve()) for i in range(100) for n in ['outputs.pt','T062_trace.pt']}
    assert set(freeze['data_reads'])==allowed and freeze['reference_reads']==0
    rr=check_render(True,freeze['rows']);steps=independent_steps(rr,rule['rho'])
    assert steps==[v['selected_step'] for v in freeze['rows']]
    for v,k in zip(freeze['rows'],steps):assert v['hashes'][k]==v['output_hash']
    refs=json.loads(Path('research_log/T062CR2/reference_inputs.json').read_bytes());items=json.loads(TARGET_MANIFEST.read_bytes())['selected']
    with ProcessPoolExecutor(max_workers=8) as pool:scored=list(pool.map(score_transfer,[(str(out),r,i,n,True) for r,i,n in zip(freeze['rows'],items,refs)]))
    original=json.loads((out/'per_image.json').read_bytes());metric_error=0.
    for a,b in zip(scored,original):
        assert a['index']==b['index'] and b['reference_read_utc']>freeze['completed_utc']
        for name in ['selected','T026','T036']:
            for key in ['psnr','ssim']:metric_error=max(metric_error,abs(a['metrics'][name][key]-b['metrics'][name][key]))
    assert metric_error<1e-10
    mapped=[dict(psnr=[v['metrics']['selected']['psnr']],ssim=[v['metrics']['selected']['ssim']],
        t026_psnr=v['metrics']['T026']['psnr'],t036_psnr=v['metrics']['T036']['psnr'],t036_ssim=v['metrics']['T036']['ssim']) for v in scored]
    m,g=independent_summary(mapped,[0]*100)
    classification='TARGET_FREE_TRANSFER_PASS' if all(g.values()) else 'TRANSFER_NEGATIVE'
    assert classification==result['classification'] and g==result['gates']
    for k in m:assert abs(m[k]-result[k])<1e-10
    verification.update(classification=classification,transfer_renders=2800,selected_outputs=100,metric_max_error=metric_error,gates=g,verified_utc=utc())
    write(out/'verification.json',verification);print(json.dumps(verification),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
