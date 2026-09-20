import argparse,json,math,statistics
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from research_log.T063A.common import sha,thash,write,utc
from research_log.T066A.run import load_inputs,state_input,development_rows,CommonRegion2
from research_log.T066A.verify import independent_development,independent_predict
from research_log.T067B.verify import independent_choices
from research_log.T068B.run import HERE,DEV_FEATURES,DEV_TABLE,MODEL,PRIOR,PRIOR_METRICS

def independent_curve(values,fs,kl,images):
    steps=list(range(fs,kl+1));d=[0.];s=[0.]
    for k in range(fs+1,kl+1):
        diff=images[k].numpy().astype(np.float64)-images[k-1].numpy().astype(np.float64)
        motion=math.sqrt(float(diff.ravel()@diff.ravel())/diff.size);d.append(motion);s.append(s[-1]+motion)
    dl=float(values[fs]-values[kl])
    u=[max(0.,min(1.,float((values[fs]-values[k])/dl))) for k in steps] if dl>1e-12 else [None]*len(steps)
    v=[x/s[-1] for x in s] if s[-1]>1e-12 else [None]*len(steps)
    a=[x-y for x,y in zip(u,v)] if dl>1e-12 and s[-1]>1e-12 else [None]*len(steps)
    if fs==kl:chosen=kl;reason='singleton'
    elif dl<=1e-12:chosen=kl;reason='objective_degenerate'
    elif s[-1]<=1e-12:chosen=kl;reason='motion_degenerate'
    else:
        chosen=steps[0];best=a[0]
        for k,adv in zip(steps,a):
            if adv>=best:chosen=k;best=adv
        reason='maximum_advantage'
    return dict(k_FS=fs,k_lambda=kl,k_knee=chosen,steps=steps,D_L=dl,u_k=u,d_k=d,s_k=s,v_k=v,a_k=a,reason=reason)

def metric_check(quality,steps,reported):
    d=[q['psnr'][k]-q['t036_psnr'] for q,k in zip(quality,steps)];b=[q['psnr'][k]-q['t026_psnr'] for q,k in zip(quality,steps)];s=[q['ssim'][k]-q['t036_ssim'] for q,k in zip(quality,steps)]
    m=dict(mean_delta_psnr=math.fsum(d)/100,median_delta_psnr=statistics.median(d),regressions_t026=sum(v<0 for v in b),worst_delta_t026=min(b),mean_delta_ssim=math.fsum(s)/100)
    for k,v in m.items():assert abs(reported[k]-v)<1e-10
    gates=dict(mean_psnr=m['mean_delta_psnr']>=2,median_psnr=m['median_delta_psnr']>0,regressions=m['regressions_t026']<=29,worst=m['worst_delta_t026']>=-5.614,mean_ssim=m['mean_delta_ssim']>=-.001)
    assert reported['gates']==gates
    return gates

def main(out):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    get=lambda n:json.loads((out/n).read_bytes());cfg=get('config.json')
    for p,h in cfg['source_binding'].items():assert sha(p)==h,p
    freeze=get('candidate_freeze.json');opened=get('reference_open.json')
    assert len(freeze['rows'])==100 and freeze['reference_reads']==0 and freeze['frozen_utc']<opened['first_development_quality_read_utc'] and sha(out/'candidate_freeze.json')==opened['freeze_sha256']
    for p,h in freeze['input_hashes'].items():assert sha(p)==h,p
    dev=json.loads(DEV_FEATURES.read_bytes())['rows'];pt=json.loads(DEV_TABLE.read_bytes());model=json.loads(MODEL.read_bytes());prior=json.loads(PRIOR.read_bytes());raw,items,recs=load_inputs(False);rendered=0;curve_error=0.
    with torch.no_grad():
        for i,(d,rec,row) in enumerate(zip(dev,recs,freeze['rows'])):
            probs=[r['p_safe'] for r in pt[i*28:(i+1)*28]];np.testing.assert_allclose(independent_predict(d['features'],model),probs,rtol=0,atol=1e-7)
            fs=min(k for k in range(d['base_step']+1) if probs[k]>=.5);kl=dict(independent_choices(d['totals'],probs,d['base_step']))[.875]
            old=next(r for r in prior['rows'] if r['index']==i and r['lambda_value']==.875);assert old['selected_step']==kl and old['k_FS']==fs
            low,trace=state_input(raw,rec,False);x=low.cuda();renderer=CommonRegion2(trace['active']).to(x).eval().requires_grad_(False);images={}
            for k in range(fs,kl+1):
                renderer.raw.copy_(trace['states'][k].to(x));images[k]=renderer(x).cpu();assert thash(images[k])==rec['rendered_hashes'][k];rendered+=1
            curve=independent_curve(d['totals'],fs,kl,images)
            for key,val in curve.items():
                if key in ['u_k','d_k','s_k','v_k','a_k'] and val[0] is not None:
                    error=max(abs(a-b) for a,b in zip(val,row[key]));curve_error=max(curve_error,error);assert error<1e-12
                else:assert row[key]==val,key
            k=curve['k_knee'];assert row['index']==i and row['low']==rec['low'] and row['low_sha256']==rec['low_sha256'] and row['trace_sha256']==rec['files']['trace.pt'] and row['images_sha256']==rec['files']['images.pt'] and row['reference_reads']==0
            assert row['state_hash']==thash(trace['states'][k]) and row['output_hash']==thash(images[k]) and row['interval_output_hashes']==rec['rendered_hashes'][fs:kl+1]
            assert row['model_sha256']==sha(MODEL) and row['rule_sha256']==sha(HERE/'core.py') and row['interpolation_rule_sha256']==sha('research_log/T067B/core.py')
            assert old['state_hash']==thash(trace['states'][kl]) and old['output_hash']==thash(images[kl])
    for p,h in json.loads((HERE/'evaluation_binding.json').read_bytes()).items():assert sha(p)==h,p
    previous=development_rows()
    with ProcessPoolExecutor(max_workers=8) as pool:quality=list(pool.map(independent_development,zip(recs,items,previous)))
    rows=freeze['rows'];steps=[r['k_knee'] for r in rows];control=[r['k_lambda'] for r in rows];result=get('result.json')
    gates=metric_check(quality,steps,result['metrics']);metric_check(quality,control,result['control_metrics'])
    old_control=next(r for r in json.loads(PRIOR_METRICS.read_bytes()) if r['lambda_value']==.875)
    assert dict(selected_steps=control,**result['control_metrics'])=={k:v for k,v in old_control.items() if k!='lambda_value'}
    paired=get('paired_control.json');assert len(paired)==100
    for i,(q,k,c,row) in enumerate(zip(quality,steps,control,paired)):
        assert row['index']==i and row['selected_step']==k and row['control_step']==c
        assert abs(row['delta_psnr']-(q['psnr'][k]-q['psnr'][c]))<1e-10 and abs(row['delta_ssim']-(q['ssim'][k]-q['ssim'][c]))<1e-10
    ps=[q['psnr'][k]-q['psnr'][c] for q,k,c in zip(quality,steps,control)];ss=[q['ssim'][k]-q['ssim'][c] for q,k,c in zip(quality,steps,control)]
    for key,val in dict(mean_delta_psnr=math.fsum(ps)/100,median_delta_psnr=statistics.median(ps),mean_delta_ssim=math.fsum(ss)/100).items():assert abs(result['deltas_vs_control'][key]-val)<1e-10
    changed=sum(k!=c for k,c in zip(steps,control));assert result['changed_choices_vs_control']==changed
    hist=lambda xs:{str(k):v for k,v in sorted(Counter(xs).items())}
    assert result['selected_step_histogram']==hist(steps) and result['changed_selected_histogram']==hist(k for k,c in zip(steps,control) if k!=c)
    verdict='OBJECTIVE_MOTION_KNEE_DEV_CANDIDATE_FROZEN' if changed and all(gates.values()) else 'OBJECTIVE_MOTION_KNEE_DEV_NO_GAIN';assert result['classification']==verdict and get('selected_choices.json')==rows
    rule=get('rule_manifest.json');assert rule['lambda_value']==.875 and rule['rho']==.9857470621423519 and rule['probability_threshold']==.5 and rule['epsilon']==1e-12 and rule['classification']==verdict and rule['transfer_authorized'] is False
    assert rule['candidate_freeze_sha256']==sha(out/'candidate_freeze.json') and rule['selected_choices_sha256']==sha(out/'selected_choices.json') and rule['probability_model_sha256']==sha(MODEL) and rule['rule_sha256']==sha(HERE/'core.py') and rule['interpolation_rule_sha256']==sha('research_log/T067B/core.py')
    write(out/'verification.json',dict(status='PASS',classification=verdict,candidate_choices=100,gpu_interval_renders=rendered,independent_quality_states=2800,curve_max_error=curve_error,metric_max_error=max(q['independent_error'] for q in quality),control_exact=True,optimizer_runs=0,model_fits=0,verified_utc=utc()));print(json.dumps(get('verification.json')),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
