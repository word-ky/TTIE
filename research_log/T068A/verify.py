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
from research_log.T068A.run import HERE,DEV_FEATURES,DEV_TABLE,MODEL,PRIOR,PRIOR_METRICS

def main(out):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    get=lambda n:json.loads((out/n).read_bytes());cfg=get('config.json')
    for p,h in cfg['source_binding'].items():assert sha(p)==h,p
    freeze=get('candidate_freeze.json');opened=get('reference_open.json')
    assert len(freeze['rows'])==2800 and freeze['reference_reads']==0 and freeze['frozen_utc']<opened['first_development_quality_read_utc'] and sha(out/'candidate_freeze.json')==opened['freeze_sha256']
    for p,h in freeze['input_hashes'].items():assert sha(p)==h,p
    dev=json.loads(DEV_FEATURES.read_bytes())['rows'];pt=json.loads(DEV_TABLE.read_bytes());model=json.loads(MODEL.read_bytes());prior=json.loads(PRIOR.read_bytes());raw,items,recs=load_inputs(False);rendered=0
    with torch.no_grad():
        for i,(d,rec) in enumerate(zip(dev,recs)):
            prob=independent_predict(d['features'],model);original=[r['p_safe'] for r in pt[i*28:(i+1)*28]]
            np.testing.assert_allclose(prob,original,rtol=0,atol=1e-7)
            fs=min(k for k in range(d['base_step']+1) if original[k]>=.5)
            interior=dict(independent_choices(d['totals'],original,d['base_step']))[.875]
            rows=freeze['rows'][i*28:(i+1)*28]
            assert [r['K'] for r in rows]==list(range(28))
            expected=[fs if cap<fs else interior if cap>interior else cap for cap in range(28)]
            assert [r['k_K'] for r in rows]==expected
            old=next(r for r in prior['rows'] if r['index']==i and r['lambda_value']==.875)
            assert old['selected_step']==rows[27]['k_K']==interior and old['k_FS']==fs
            low,trace=state_input(raw,rec,False);x=low.cuda();renderer=CommonRegion2(trace['active']).to(x).eval().requires_grad_(False);images={}
            for k in sorted(set(expected)):
                renderer.raw.copy_(trace['states'][k].to(x));images[k]=thash(renderer(x).cpu());rendered+=1
            assert old['state_hash']==thash(trace['states'][interior]) and old['output_hash']==images[interior]
            for row in rows:
                k=row['k_K'];assert row['index']==i and row['low']==rec['low'] and row['low_sha256']==rec['low_sha256'] and row['trace_sha256']==rec['files']['trace.pt'] and row['images_sha256']==rec['files']['images.pt']
                assert row['state_hash']==thash(trace['states'][k]) and row['output_hash']==images[k]==rec['rendered_hashes'][k]
                assert row['k_FS']==fs and row['k_lambda']==interior and row['model_sha256']==sha(MODEL) and row['rule_sha256']==sha(HERE/'core.py') and row['interpolation_rule_sha256']==sha('research_log/T067B/core.py')
    for p,h in json.loads((HERE/'evaluation_binding.json').read_bytes()).items():assert sha(p)==h,p
    previous=development_rows()
    with ProcessPoolExecutor(max_workers=8) as pool:quality=list(pool.map(independent_development,zip(recs,items,previous)))
    table=get('candidate_metrics.json');assert len(table)==28
    for cap,row in enumerate(table):
        assert row['K']==cap;steps=[r['k_K'] for r in freeze['rows'] if r['K']==cap];assert steps==row['selected_steps']
        d=[q['psnr'][k]-q['t036_psnr'] for q,k in zip(quality,steps)];b=[q['psnr'][k]-q['t026_psnr'] for q,k in zip(quality,steps)];ss=[q['ssim'][k]-q['t036_ssim'] for q,k in zip(quality,steps)]
        m=dict(mean_delta_psnr=math.fsum(d)/100,median_delta_psnr=statistics.median(d),regressions_t026=sum(v<0 for v in b),worst_delta_t026=min(b),mean_delta_ssim=math.fsum(ss)/100)
        for key,v in m.items():assert abs(row[key]-v)<1e-10
        gates=dict(mean_psnr=m['mean_delta_psnr']>=2,median_psnr=m['median_delta_psnr']>0,regressions=m['regressions_t026']<=29,worst=m['worst_delta_t026']>=-5.614,mean_ssim=m['mean_delta_ssim']>=-.001);assert row['gates']==gates
    control=next(r for r in json.loads(PRIOR_METRICS.read_bytes()) if r['lambda_value']==.875)
    assert {k:v for k,v in control.items() if k!='lambda_value'}=={k:v for k,v in table[27].items() if k!='K'}
    # Independently verified metrics; use exact primary values for exact ties.
    passing=[r for r in table if all(r['gates'].values())]
    order=sorted(passing,key=lambda r:(-r['worst_delta_t026'],-r['mean_delta_psnr'],-r['median_delta_psnr'],-r['K']))
    result=get('result.json');best=order[0]
    assert result['passing_candidates_ranked']==[r['K'] for r in order] and result['selected_metrics']==best and result['selected_K']==best['K']
    verdict='ABS_STEP_CAP_DEV_CANDIDATE_FROZEN' if best['K']<27 else 'ABS_STEP_CAP_DEV_NO_GAIN';assert result['classification']==verdict
    selected=[r for r in freeze['rows'] if r['K']==best['K']];assert selected==get('selected_choices.json')
    assert result['changed_choices_vs_uncapped']==sum(r['k_K']!=r['k_lambda'] for r in selected)
    assert result['selected_step_histogram']=={str(k):v for k,v in sorted(Counter(r['k_K'] for r in selected).items())}
    rule=get('rule_manifest.json')
    assert rule['K']==best['K'] and rule['lambda_value']==.875 and rule['rho']==.9857470621423519 and rule['probability_threshold']==.5 and rule['caps']==list(range(28))
    assert rule['classification']==verdict and rule['transfer_authorized'] is False and rule['candidate_freeze_sha256']==sha(out/'candidate_freeze.json') and rule['candidate_metrics_sha256']==sha(out/'candidate_metrics.json') and rule['selected_choices_sha256']==sha(out/'selected_choices.json') and rule['probability_model_sha256']==sha(MODEL) and rule['rule_sha256']==sha(HERE/'core.py') and rule['interpolation_rule_sha256']==sha('research_log/T067B/core.py')
    write(out/'verification.json',dict(status='PASS',classification=verdict,selected_K=best['K'],candidate_choices=2800,gpu_unique_renders=rendered,independent_quality_states=2800,metric_max_error=max(q['independent_error'] for q in quality),control_exact=True,optimizer_runs=0,model_fits=0,verified_utc=utc()));print(json.dumps(get('verification.json')),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
