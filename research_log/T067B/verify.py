import argparse,json,math,statistics
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from research_log.T063A.common import sha,thash,write,utc
from research_log.T066A.run import load_inputs,state_input,development_rows,CommonRegion2
from research_log.T066A.verify import independent_development,independent_predict
from research_log.T067B.run import HERE,DEV_FEATURES,DEV_TABLE,MODEL

def independent_choices(values,probabilities,base):
    reduction=float(values[0]-min(values));assert reduction>1e-12
    r=[min(1.,max(0.,(values[0]-v)/max(reduction,1e-12))) for v in values];fs=min(k for k in range(base+1) if probabilities[k]>=.5)
    return [(j/8,min(k for k in range(fs,base+1) if r[k]>=r[fs]+(j/8)*(.9857470621423519-r[fs]))) for j in range(9)]

def main(out):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;get=lambda n:json.loads((out/n).read_bytes());cfg=get('config.json')
    for p,h in cfg['source_binding'].items():assert sha(p)==h,p
    freeze=get('candidate_freeze.json');opened=get('reference_open.json');assert len(freeze['rows'])==900 and freeze['reference_reads']==0 and freeze['frozen_utc']<opened['first_development_quality_read_utc'] and sha(out/'candidate_freeze.json')==opened['freeze_sha256']
    for p,h in freeze['input_hashes'].items():assert sha(p)==h,p
    dev=json.loads(DEV_FEATURES.read_bytes())['rows'];pt=json.loads(DEV_TABLE.read_bytes());model=json.loads(MODEL.read_bytes());raw,items,recs=load_inputs(False);rendered=0
    with torch.no_grad():
        for i,(d,rec) in enumerate(zip(dev,recs)):
            prob=independent_predict(d['features'],model);np.testing.assert_allclose(prob,[r['p_safe'] for r in pt[i*28:(i+1)*28]],rtol=0,atol=1e-7)
            original_probs=[r['p_safe'] for r in pt[i*28:(i+1)*28]];expected=independent_choices(d['totals'],original_probs,d['base_step']);rows=freeze['rows'][i*9:(i+1)*9];assert [(r['lambda_value'],r['selected_step']) for r in rows]==expected
            low,trace=state_input(raw,rec,False);x=low.cuda();renderer=CommonRegion2(trace['active']).to(x).eval().requires_grad_(False);images={}
            for k in sorted({r['selected_step'] for r in rows}):renderer.raw.copy_(trace['states'][k].to(x));images[k]=thash(renderer(x).cpu());rendered+=1
            fs=rows[0]['selected_step'];base=d['base_step'];den=d['totals'][0]-min(d['totals']);rfs=(d['totals'][0]-d['totals'][fs])/max(den,1e-12)
            assert rows[-1]['selected_step']==base
            for row in rows:
                k=row['selected_step'];assert row['index']==i and row['low']==rec['low'] and row['low_sha256']==rec['low_sha256'] and row['trace_sha256']==rec['files']['trace.pt'] and row['images_sha256']==rec['files']['images.pt']
                assert row['state_hash']==thash(trace['states'][k]) and row['output_hash']==images[k]==rec['rendered_hashes'][k]
                assert row['k_FS']==fs and row['k_rho']==base and abs(row['r_FS']-rfs)<1e-12 and abs(row['r_target']-(rfs+row['lambda_value']*(.9857470621423519-rfs)))<1e-12
    for p,h in json.loads((HERE/'evaluation_binding.json').read_bytes()).items():assert sha(p)==h,p
    prior=development_rows()
    with ProcessPoolExecutor(max_workers=8) as pool:quality=list(pool.map(independent_development,zip(recs,items,prior)))
    table=get('candidate_metrics.json');computed=[]
    for j,row in enumerate(table):
        assert row['lambda_value']==j/8;steps=[r['selected_step'] for r in freeze['rows'] if r['lambda_value']==j/8];assert steps==row['selected_steps']
        d=[q['psnr'][k]-q['t036_psnr'] for q,k in zip(quality,steps)];b=[q['psnr'][k]-q['t026_psnr'] for q,k in zip(quality,steps)];s=[q['ssim'][k]-q['t036_ssim'] for q,k in zip(quality,steps)]
        m=dict(mean_delta_psnr=math.fsum(d)/100,median_delta_psnr=statistics.median(d),regressions_t026=sum(v<0 for v in b),worst_delta_t026=min(b),mean_delta_ssim=math.fsum(s)/100)
        for k,v in m.items():assert abs(row[k]-v)<1e-10
        gates=dict(mean_psnr=m['mean_delta_psnr']>=2,median_psnr=m['median_delta_psnr']>0,regressions=m['regressions_t026']<=29,worst=m['worst_delta_t026']>=-5.614,mean_ssim=m['mean_delta_ssim']>=-.001);assert row['gates']==gates
        computed.append(dict(lambda_value=j/8,**m,gates=gates))
    # Rank exact frozen primary values: tiny independent metric roundoff must not invent a non-exact tie.
    passing=[r for r in table if all(r['gates'].values())];order=sorted(passing,key=lambda r:(r['worst_delta_t026'],r['mean_delta_psnr'],-r['lambda_value']),reverse=True);result=get('result.json');best=order[0] if order else None
    assert result['passing_candidates_ranked']==[r['lambda_value'] for r in order] and result['selected_metrics']==best and result['selected_lambda']==(best['lambda_value'] if best else None)
    verdict='INTERIOR_PROGRESS_DEV_CANDIDATE_FROZEN' if best and 0<best['lambda_value']<1 else 'INTERIOR_PROGRESS_DEV_NEGATIVE';assert result['classification']==verdict
    rule=get('rule_manifest.json');assert rule['lambda_value']==result['selected_lambda'] and rule['classification']==verdict and rule['transfer_authorized'] is False and rule['candidate_freeze_sha256']==sha(out/'candidate_freeze.json') and rule['candidate_metrics_sha256']==sha(out/'candidate_metrics.json') and rule['probability_model_sha256']==sha(MODEL)
    write(out/'verification.json',dict(status='PASS',classification=verdict,selected_lambda=result['selected_lambda'],candidate_choices=900,gpu_unique_renders=rendered,independent_quality_states=2800,metric_max_error=max(q['independent_error'] for q in quality),optimizer_runs=0,model_fits=0,verified_utc=utc()));print(json.dumps(get('verification.json')),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
