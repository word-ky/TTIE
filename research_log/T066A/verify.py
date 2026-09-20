"""Independent temporal features/LOIO logistic/rollback and offline metric audit; no feedback to fitting."""
import argparse, json, math, statistics
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from scripts.evaluate_t026a import pixels, independent_ssim
from research_log.T066A.run import (ROOT, DEV, HERE, DEV_MANIFEST, TARGET_MANIFEST, TARGET,
    load_inputs, state_input, development_rows, score_transfer, TARGET_LOW, NORMALIZATION)
from ttie.common_gain import CommonRegion2
from research_log.T062A.core import losses
from research_log.T063A.common import sha, thash, write, utc


def independent_base(totals):
    improvement=totals[0]-min(totals[:28])
    if not math.isfinite(improvement) or improvement<=1e-12:return 0
    target=totals[0]-0.9857470621423519*improvement
    return next((k for k,v in enumerate(totals[:28]) if math.isfinite(v) and v<=target),0)


def independent_snapshot(low,images,states,totals):
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


def independent_features(low,images,states,totals):
    snapshot=independent_snapshot(low,images,states,totals);denom=max(abs(totals[0]-min(totals)),1e-8);out=[]
    y=[im.numpy().astype(np.float64) for im in images];raw=[s.numpy().astype(np.float64).ravel() for s in states]
    for k in range(len(y)):
        values=[(totals[max(0,k-3)]-totals[k])/denom,((totals[k-2]-totals[k-1])-(totals[k-1]-totals[k]))/denom if k>=2 else 0.]
        if k==0:values += [0.]*6
        else:
            dy=y[k]-y[k-1];ds=raw[k]-raw[k-1]
            values += [float(np.sqrt(np.mean(dy*dy))),float(np.sqrt(np.mean(((y[k]-y[k-1])-(y[k-1]-y[k-2]))**2))) if k>=2 else 0.,
                float(np.sqrt(ds@ds)),float(np.linalg.norm((raw[k]-raw[k-1])-(raw[k-1]-raw[k-2]))) if k>=2 else 0.,
                abs(snapshot[k,3]-snapshot[k-1,3]),abs(snapshot[k,7]-snapshot[k-1,7])]
        out.append(values)
    return np.column_stack([snapshot,np.asarray(out)])


def independent_normalization(x,old):
    d=x[:,11:];mean=np.sum(d,axis=0)/len(d);centered=d-mean
    scale=np.maximum(np.sqrt(np.einsum('ij,ij->j',centered,centered)/len(d)),1e-8)
    return dict(mean=list(old['mean'])+mean.tolist(),scale=list(old['scale'])+scale.tolist())


def independent_fit(x,y,normalization):
    from scipy.linalg import solve
    from scipy.special import expit
    mean=np.asarray(normalization['mean']);scale=np.asarray(normalization['scale'])
    a=np.concatenate((np.ones((len(x),1)),(x-mean)/scale),axis=1)
    ns=int(np.count_nonzero(y));nu=len(y)-ns;weights=np.where(y==1,len(y)/(2*ns),len(y)/(2*nu))
    penalty=np.diag([0.]+[.001]*19);beta=np.zeros(20)
    for iteration in range(50):
        probability=expit(np.clip(np.einsum('ij,j->i',a,beta),-30,30))
        g=np.einsum('ni,n->i',a,weights*(probability-y))+penalty@beta
        H=np.einsum('ni,n,nj->ij',a,weights*probability*(1-probability),a)+penalty
        step=solve(H,g,assume_a='pos');beta-=step
        if np.max(np.abs(step))<1e-12:break
    return dict(mean=mean,scale=scale,intercept=beta[0],coefficients=beta[1:])


def check_newton(trace,x,y,normalization,saved):
    from scipy.special import expit
    a=np.column_stack((np.ones(len(x)),(x-np.asarray(normalization['mean']))/np.asarray(normalization['scale'])))
    ns=int(y.sum());weights=np.where(y==1,len(y)/(2*ns),len(y)/(2*(len(y)-ns)));penalty=np.diag([0.]+[.001]*19)
    previous=np.zeros(20);max_residual=0.;assert 1<=len(trace)<=50
    for i,row in enumerate(trace):
        before=np.asarray(row['beta_before']);step=np.asarray(row['newton_step']);after=np.asarray(row['beta_after'])
        assert row['iteration']==i;np.testing.assert_array_equal(before,previous);np.testing.assert_array_equal(after,before-step)
        logits=np.clip(a@before,-30,30);pr=expit(logits)
        g=np.einsum('ni,n->i',a,weights*(pr-y))+penalty@before
        H=np.einsum('ni,n,nj->ij',a,weights*pr*(1-pr),a)+penalty
        residual=float(np.max(np.abs(H@step-g)));max_residual=max(max_residual,residual);assert residual<1e-8
        loss=float(np.sum(weights*(np.logaddexp(0,logits)-y*logits))+.0005*(before[1:]@before[1:]))
        assert abs(loss-row['loss'])<1e-8 and row['step_inf']==float(np.max(np.abs(step)))
        if i<len(trace)-1:assert row['step_inf']>=1e-12
        previous=after
    assert len(trace)==50 or trace[-1]['step_inf']<1e-12
    np.testing.assert_array_equal(previous,[saved['intercept'],*saved['coefficients']])
    return max_residual


def independent_predict(x,model):
    from scipy.special import expit
    logits=np.einsum('ij,j->i',(np.asarray(x)-model['mean'])/model['scale'],model['coefficients'])+model['intercept']
    return expit(np.clip(logits,-30,30))


def independent_steps(records,model):
    predictions=[independent_predict(row['features'],model).tolist() for row in records];steps=[]
    for row,values in zip(records,predictions):
        selected=0
        for k in range(row['base_step'],-1,-1):
            if values[k]>=.5:selected=k;break
        steps.append(selected)
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
    result=json.loads((out/'result.json').read_bytes());config=json.loads((out/'config.json').read_bytes())
    for name,h in config['source_binding'].items():assert sha(name)==h,name
    manifest=json.loads((out/'development_manifest.json').read_bytes())
    for name,key in [('training_table.json','training_table_sha256'),('development_features.json','features_sha256'),('development_fit.json','development_fit_sha256'),('class_counts.json','class_counts_sha256')]:assert sha(out/name)==manifest[key]
    assert sha(NORMALIZATION)==manifest['normalization_sha256'] and sha('research_log/T065A/core.py')==manifest['feature_source_sha256']
    dev=json.loads((out/'development_features.json').read_bytes());r,feature_error=check_render(False,dev['rows'])
    raw,items,recs=load_inputs(False);prior=development_rows()
    with ProcessPoolExecutor(max_workers=8) as pool:quality=list(pool.map(independent_development,zip(recs,items,prior)))
    table=json.loads((out/'training_table.json').read_bytes());x=np.concatenate([row['features'] for row in r]);y=np.asarray([q['psnr'][k]-q['t026_psnr'] for q in quality for k in range(28)])
    np.testing.assert_allclose(x,[row['features'] for row in table],rtol=0,atol=2e-12);np.testing.assert_allclose(y,[row['target'] for row in table],rtol=0,atol=1e-10)
    labels=(y>=-5.614).astype(float);ids=np.repeat(np.arange(100),28);np.testing.assert_array_equal(labels,[row['safe'] for row in table])
    old=json.loads(NORMALIZATION.read_bytes());predictions=[];steps=[];coefficient_error=0.;residual=0.;norm_error=0.
    for i in range(100):
        p=out/f'fold_{i:03d}.json';assert sha(p)==manifest['folds'][i]['sha256'];fold=json.loads(p.read_bytes());keep=ids!=i
        assert fold['held_out']==i and fold['training_images']==[j for j in range(100) if j!=i]
        norm=independent_normalization(x[keep],old);saved=fold['model']
        assert saved['mean'][:11]==old['mean'] and saved['scale'][:11]==old['scale']
        for key in ['mean','scale']:norm_error=max(norm_error,float(np.max(np.abs(np.asarray(norm[key])-saved[key]))))
        assert norm_error<1e-10
        model=independent_fit(x[keep],labels[keep],norm)
        for key in ['intercept','coefficients']:
            error=float(np.max(np.abs(np.asarray(model[key])-saved[key])));coefficient_error=max(coefficient_error,error);assert error<1e-6,(i,key,error)
        assert saved['class_info']['n']==2772 and saved['class_info']['safe']==int(labels[keep].sum())
        ns=int(labels[keep].sum());assert saved['class_info']['safe_weight']==2772/(2*ns) and saved['class_info']['unsafe_weight']==2772/(2*(2772-ns))
        residual=max(residual,check_newton(fold['trace'],x[keep],labels[keep],norm,saved))
        prob=independent_predict(r[i]['features'],model).tolist();predictions.append(prob)
        selected=next((k for k in range(r[i]['base_step'],-1,-1) if prob[k]>=.5),0);steps.append(selected)
        print('verified-loio',i+1,flush=True)
    development=json.loads((out/'development_fit.json').read_bytes());np.testing.assert_allclose(predictions,development['probabilities'],rtol=0,atol=1e-7)
    assert steps==development['selected_steps'] and [v['base_step'] for v in r]==development['base_steps']
    dm,dg=independent_summary(quality,steps)
    for key in dm:assert abs(dm[key]-development['metrics'][key])<1e-10
    assert dg==development['metrics']['gates']
    pr=np.asarray(predictions).reshape(-1)>=.5;truth=labels.astype(bool)
    cm=dict(safe_pred_safe=int(np.sum(truth&pr)),safe_pred_unsafe=int(np.sum(truth&~pr)),unsafe_pred_safe=int(np.sum(~truth&pr)),unsafe_pred_unsafe=int(np.sum(~truth&~pr)))
    recall=cm['unsafe_pred_unsafe']/(cm['unsafe_pred_unsafe']+cm['unsafe_pred_safe'])
    assert cm==development['confusion'] and recall==development['unsafe_recall']
    assert development['selected_step_histogram']=={str(k):steps.count(k) for k in range(28)}
    assert development['choices_changed']==sum(k!=v['base_step'] for k,v in zip(steps,r))
    allowed_dev={str((DEV/f'{i:03d}'/'trace.pt').resolve()) for i in range(100)}|{str((ROOT/'shared/t036a/low'/v['low']).resolve()) for v in dev['rows']}
    assert set(dev['data_reads'])==allowed_dev
    verification=dict(status='PASS',development_renders=2800,development_feature_max_error=feature_error,loio_fits=100,coefficient_max_error=coefficient_error,
        normalization_max_error=norm_error,newton_equation_max_residual=residual,development_metric_max_error=max(v['independent_error'] for v in quality),development_confusion=cm,unsafe_recall=recall)
    if recall<.5 or not all(dg.values()):
        assert result['classification']=='DEVELOPMENT_DYNAMICS_NEGATIVE' and not (out/'transfer_freeze.json').exists() and not (out/'reference_open.json').exists() and not (out/'model.json').exists()
        assert result['development_manifest_sha256']==sha(out/'development_manifest.json')
        verification.update(classification='DEVELOPMENT_DYNAMICS_NEGATIVE',transfer_renders=0,final_fits=0,verified_utc=utc());write(out/'verification.json',verification);print(json.dumps(verification),flush=True);return
    rule=json.loads((out/'selector_manifest.json').read_bytes());assert rule['development_manifest_sha256']==sha(out/'development_manifest.json')
    for name,key in [('model.json','model_sha256'),('normalization.json','normalization_sha256'),('newton_trace.json','newton_trace_sha256')]:assert sha(out/name)==rule[key]
    norm=independent_normalization(x,old);saved=json.loads((out/'model.json').read_bytes());model=independent_fit(x,labels,norm)
    for key in ['mean','scale','intercept','coefficients']:np.testing.assert_allclose(model[key],saved[key],rtol=0,atol=1e-6)
    check_newton(json.loads((out/'newton_trace.json').read_bytes()),x,labels,norm,saved)
    freeze=json.loads((out/'transfer_freeze.json').read_bytes());marker=json.loads((out/'reference_open.json').read_bytes())
    assert freeze['selector_sha256']==sha(out/'selector_manifest.json')==result['selector_sha256']
    assert marker['transfer_freeze_sha256']==sha(out/'transfer_freeze.json')==result['transfer_freeze_sha256']
    assert rule['frozen_utc']<freeze['completed_utc']<marker['first_reference_or_quality_read_utc']
    allowed={str((TARGET/f'{i:03d}'/'T063_trace.pt').resolve()) for i in range(100)}|{str((TARGET_LOW/v['low']).resolve()) for v in freeze['rows']}
    assert set(freeze['data_reads'])==allowed and freeze['reference_reads']==0
    rr,feature_error=check_render(True,freeze['rows']);predictions,steps=independent_steps(rr,model)
    np.testing.assert_allclose(predictions,[v['probabilities'] for v in freeze['rows']],rtol=0,atol=1e-7)
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
    classification='DYNAMICS_SAFETY_GUARD_TRANSFER_PASS' if all(g.values()) else 'TRANSFER_NEGATIVE'
    assert classification==result['classification'] and g==result['gates']
    for k in m:assert abs(m[k]-result[k])<1e-10
    assert result['choices_changed']==sum(k!=v['base_step'] for k,v in zip(steps,rr))
    assert result['selected_step_histogram']=={str(k):steps.count(k) for k in range(28)}
    assert json.loads((out/'tail_outcomes.json').read_bytes())==[v for v in original if v['index'] in [16,86]]
    verification.update(classification=classification,transfer_feature_max_error=feature_error,transfer_renders=2800,selected_outputs=100,metric_max_error=metric_error,gates=g,verified_utc=utc())
    write(out/'verification.json',verification);print(json.dumps(verification),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
