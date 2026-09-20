"""Independent features/ridge/selection and offline metric audit; no feedback to fitting."""
import argparse, json, math, statistics
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from scripts.evaluate_t026a import pixels, independent_ssim
from research_log.T065A.run import (ROOT, DEV, HERE, DEV_MANIFEST, TARGET_MANIFEST, TARGET,
    load_inputs, state_input, development_rows, score_transfer, TARGET_LOW)
from ttie.common_gain import CommonRegion2
from research_log.T062A.core import losses
from research_log.T063A.common import sha, thash, write, utc


def independent_base(totals):
    improvement=totals[0]-min(totals[:28])
    if not math.isfinite(improvement) or improvement<=1e-12:return 0
    target=totals[0]-0.9857470621423519*improvement
    return next((k for k,v in enumerate(totals[:28]) if math.isfinite(v) and v<=target),0)


def independent_features(low,images,states,totals):
    x=low.numpy().astype(np.float64)
    def luma(v):return .299*v[:,0]+.587*v[:,1]+.114*v[:,2]
    def grad(v):
        a=np.abs(np.diff(v,axis=1));b=np.abs(np.diff(v,axis=2))
        return float((a.sum()+b.sum())/(a.size+b.size))
    denom=max(grad(luma(x)),1e-8);reduction=max(totals[0]-min(totals),1e-8);rows=[]
    for k,(image,state) in enumerate(zip(images,states)):
        y=image.numpy().astype(np.float64);Y=luma(y);mean=float(Y.mean());raw=state.numpy().astype(np.float64).ravel()
        rows.append([k/27,(totals[0]-totals[k])/reduction,0. if k==0 else (totals[k-1]-totals[k])/reduction,
            mean,float(np.sqrt(np.mean((Y-mean)**2))),float(np.count_nonzero(Y>=.98)/Y.size),float(np.count_nonzero(Y<=.02)/Y.size),
            grad(Y)/denom,float(np.abs(y-x).mean()),float(np.sqrt(raw@raw)),float(np.max(np.abs(raw)))])
    return np.asarray(rows,dtype=np.float64)


def independent_fit(x,y):
    from scipy.linalg import solve
    mean=np.sum(x,axis=0)/len(x);centered=x-mean
    scale=np.maximum(np.sqrt(np.einsum('ij,ij->j',centered,centered)/len(x)),1e-8)
    z=centered/scale;a=np.concatenate((np.ones((len(x),1)),z),axis=1)
    gram=np.einsum('ni,nj->ij',a,a);gram[1:,1:]+=np.eye(11)*.001
    rhs=np.einsum('ni,n->i',a,y);beta=solve(gram,rhs,assume_a='pos')
    return dict(mean=mean,scale=scale,intercept=beta[0],coefficients=beta[1:])


def independent_predict(x,model):
    return np.einsum('ij,j->i',(np.asarray(x)-model['mean'])/model['scale'],model['coefficients'])+model['intercept']


def independent_steps(records,model):
    predictions=[independent_predict(row['features'],model).tolist() for row in records]
    steps=[max(range(row['base_step']+1),key=lambda k:(values[k],-k)) for row,values in zip(records,predictions)]
    return predictions,steps


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
    raw,items,recs=load_inputs(transfer);verified=[];maxerror=0.
    for i,(rec,reported) in enumerate(zip(recs,records)):
        low,trace=state_input(raw,rec,transfer);x=low.cuda()
        model=CommonRegion2(trace['active']).to(x).eval().requires_grad_(False)
        images=[];totals=[];parts=[];hashes=[]
        with torch.no_grad():
            for state in trace['states'][:28]:
                model.raw.copy_(state.to(x));y=model(x);c=losses(x,y)
                parts.append(c.cpu().double().tolist());totals.append(float(c@c.new_tensor([1.,10.,5.])))
                images.append(y.cpu().clone());hashes.append(thash(images[-1]))
        assert hashes==reported['hashes'] and parts==reported['components'] and totals==reported['totals']
        values=independent_features(low,images,trace['states'][:28],totals)
        error=float(np.abs(values-np.asarray(reported['features'])).max());maxerror=max(maxerror,error);assert error<2e-12
        k=independent_base(totals);assert k==reported['base_step']
        if transfer:assert k==rec['methods']['T063']['selected_step'] and hashes[k]==rec['methods']['T063']['output_hash']
        else:assert hashes==rec['rendered_hashes'][:28]
        verified.append(dict(features=values.tolist(),base_step=k))
        print('verified-transfer' if transfer else 'verified-development',i+1,flush=True)
    return verified,maxerror


def main(out):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rule=json.loads((out/'selector_manifest.json').read_bytes());result=json.loads((out/'result.json').read_bytes())
    for p,h in rule['source_binding'].items():assert sha(p)==h,p
    for name,key in [('model.json','model_sha256'),('training_table.json','training_table_sha256'),('development_features.json','features_sha256'),('development_fit.json','development_fit_sha256')]:assert sha(out/name)==rule[key]
    dev=json.loads((out/'development_features.json').read_bytes());r,feature_error=check_render(False,dev['rows'])
    raw,items,recs=load_inputs(False);prior=development_rows()
    with ProcessPoolExecutor(max_workers=8) as pool:quality=list(pool.map(independent_development,zip(recs,items,prior)))
    table=json.loads((out/'training_table.json').read_bytes());x=np.concatenate([row['features'] for row in r]);y=np.asarray([q['psnr'][k]-q['t026_psnr'] for q in quality for k in range(28)])
    np.testing.assert_allclose(x,[row['features'] for row in table],rtol=0,atol=2e-12)
    np.testing.assert_allclose(y,[row['target'] for row in table],rtol=0,atol=1e-10)
    for n,row in enumerate(table):assert row['index']==n//28 and row['step']==n%28 and row['low']==items[n//28]['low']
    model=independent_fit(x,y);saved=json.loads((out/'model.json').read_bytes());coefficient_error=0.
    for key in ['mean','scale','intercept','coefficients']:
        error=float(np.max(np.abs(np.asarray(model[key])-np.asarray(saved[key]))));coefficient_error=max(coefficient_error,error);assert error<1e-7,(key,error)
    assert saved['ridge_lambda']==rule['ridge_lambda']==.001 and rule['rho']==0.9857470621423519
    predictions,steps=independent_steps(r,model);development=json.loads((out/'development_fit.json').read_bytes())
    np.testing.assert_allclose(predictions,development['predictions'],rtol=0,atol=1e-7)
    assert steps==development['selected_steps'] and [v['base_step'] for v in r]==development['base_steps']
    dm,dg=independent_summary(quality,steps)
    for key in dm:assert abs(dm[key]-development['metrics'][key])<1e-10
    assert dg==development['metrics']['gates']
    allowed_dev={str((DEV/f'{i:03d}'/'trace.pt').resolve()) for i in range(100)}|{str((ROOT/'shared/t036a/low'/v['low']).resolve()) for v in dev['rows']}
    assert set(dev['data_reads'])==allowed_dev
    verification=dict(status='PASS',development_renders=2800,development_feature_max_error=feature_error,coefficient_max_error=coefficient_error,
        development_metric_max_error=max(v['independent_error'] for v in quality),training_rows=2800)
    freeze=json.loads((out/'transfer_freeze.json').read_bytes());marker=json.loads((out/'reference_open.json').read_bytes())
    assert freeze['selector_sha256']==sha(out/'selector_manifest.json')==result['selector_sha256']
    assert marker['transfer_freeze_sha256']==sha(out/'transfer_freeze.json')==result['transfer_freeze_sha256']
    assert rule['frozen_utc']<freeze['completed_utc']<marker['first_reference_or_quality_read_utc']
    allowed={str((TARGET/f'{i:03d}'/'T063_trace.pt').resolve()) for i in range(100)}|{str((TARGET_LOW/v['low']).resolve()) for v in freeze['rows']}
    assert set(freeze['data_reads'])==allowed and freeze['reference_reads']==0
    rr,feature_error=check_render(True,freeze['rows']);predictions,steps=independent_steps(rr,model)
    np.testing.assert_allclose(predictions,[v['predictions'] for v in freeze['rows']],rtol=0,atol=1e-7)
    assert steps==[v['selected_step'] for v in freeze['rows']]
    for v,k in zip(freeze['rows'],steps):assert v['hashes'][k]==v['output_hash']
    refs=json.loads(Path('research_log/T063D/reference_inputs.json').read_bytes());items=json.loads(TARGET_MANIFEST.read_bytes())['selected']
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
    classification='LINEAR_QUALITY_HEAD_TRANSFER_PASS' if all(g.values()) else 'TRANSFER_NEGATIVE'
    assert classification==result['classification'] and g==result['gates']
    for k in m:assert abs(m[k]-result[k])<1e-10
    assert result['choices_changed']==sum(k!=v['base_step'] for k,v in zip(steps,rr))
    assert result['selected_step_histogram']=={str(k):steps.count(k) for k in range(28)}
    assert json.loads((out/'tail_outcomes.json').read_bytes())==[v for v in original if v['index'] in [16,86]]
    verification.update(classification=classification,transfer_feature_max_error=feature_error,transfer_renders=2800,selected_outputs=100,metric_max_error=metric_error,gates=g,verified_utc=utc())
    write(out/'verification.json',verification);print(json.dumps(verification),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
