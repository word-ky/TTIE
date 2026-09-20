import argparse,json,os,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from research_log.T063A.common import sha,thash,write,utc
from research_log.T066A.run import load_inputs,state_input,development_rows
from research_log.T066A.core import predict
from research_log.T063B.core import summarize
from research_log.T067B.core import choices,RHO
from research_log.T067B.run import DEV_FEATURES,DEV_TABLE,MODEL
from research_log.T068B.core import knee,gpu_motions
LAMBDA=.875
HERE=Path('research_log/T068B');PRIOR=Path('research_log/T067B/evidence/candidate_freeze.json');PRIOR_METRICS=Path('research_log/T067B/evidence/candidate_metrics.json')

def main(out):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    binding=json.loads((HERE/'binding.json').read_bytes())
    for p,h in binding.items():assert sha(p)==h,p
    out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    write(out/'config.json',dict(task='T068-B',source_commit=os.environ['TTIE_SOURCE_COMMIT'],source_binding=binding,started_utc=utc(),optimizer_runs=0,model_fits=0))
    dev=json.loads(DEV_FEATURES.read_bytes())['rows'];pt=json.loads(DEV_TABLE.read_bytes());model=json.loads(MODEL.read_bytes());prior=json.loads(PRIOR.read_bytes());raw,items,recs=load_inputs(False);rows=[]
    model_hash=sha(MODEL);rule_hash=sha(HERE/'core.py');interpolation_hash=sha('research_log/T067B/core.py')
    assert len(dev)==len(recs)==100 and prior['reference_reads']==0
    for i,(d,rec) in enumerate(zip(dev,recs)):
        low,trace=state_input(raw,rec,False);probs=predict(d['features'],model).tolist();oldpt=pt[i*28:(i+1)*28]
        np.testing.assert_array_equal(probs,[r['p_safe'] for r in oldpt]);assert d['hashes']==rec['rendered_hashes'][:28]
        np.testing.assert_allclose(d['totals'],trace['values'][:28],rtol=0,atol=1e-7)
        assert all(r['base_step']==d['base_step'] and r['index']==i and r['features']==d['features'][r['step']] for r in oldpt)
        interior=next(r for r in choices(d['totals'],probs,d['base_step']) if r['lambda_value']==LAMBDA)
        old=next(r for r in prior['rows'] if r['index']==i and r['lambda_value']==LAMBDA)
        assert all(old[k]==v for k,v in interior.items())
        assert old['low']==rec['low'] and old['low_sha256']==rec['low_sha256'] and old['trace_sha256']==rec['files']['trace.pt'] and old['images_sha256']==rec['files']['images.pt']
        imagepath=raw/f"{i:03d}"/'images.pt';assert sha(imagepath)==rec['files']['images.pt']
        images=torch.load(imagepath,weights_only=True,map_location='cpu')[:28]
        fs=interior['k_FS'];kl=interior['selected_step']
        assert [thash(images[k]) for k in range(fs,kl+1)]==rec['rendered_hashes'][fs:kl+1]
        assert thash(trace['states'][kl])==old['state_hash'] and thash(images[kl])==old['output_hash']
        curve=knee(d['totals'],fs,kl,gpu_motions(images,fs,kl));k=curve['k_knee']
        curve.update(index=i,low=rec['low'],low_sha256=rec['low_sha256'],trace_sha256=rec['files']['trace.pt'],images_sha256=rec['files']['images.pt'],state_hash=thash(trace['states'][k]),output_hash=thash(images[k]),interval_output_hashes=rec['rendered_hashes'][fs:kl+1],model_sha256=model_hash,rule_sha256=rule_hash,interpolation_rule_sha256=interpolation_hash,reference_reads=0)
        rows.append(curve)
    write(out/'candidate_freeze.json',dict(rows=rows,source_commit=os.environ['TTIE_SOURCE_COMMIT'],input_hashes={str(p):sha(p) for p in [DEV_FEATURES,DEV_TABLE,MODEL,PRIOR]},reference_reads=0,frozen_utc=utc()))
    write(out/'reference_open.json',dict(first_development_quality_read_utc=utc(),freeze_sha256=sha(out/'candidate_freeze.json')))
    for p,h in json.loads((HERE/'evaluation_binding.json').read_bytes()).items():assert sha(p)==h,p
    quality=development_rows()
    for q,d in zip(quality,dev):assert q['state_hashes'][:28]==d['hashes']
    selected=[r['k_knee'] for r in rows];control_steps=[r['k_lambda'] for r in rows]
    metrics=summarize(quality,selected);metrics.pop('eligible');control=summarize(quality,control_steps);control.pop('eligible')
    prior_control=next(r for r in json.loads(PRIOR_METRICS.read_bytes()) if r['lambda_value']==LAMBDA)
    assert dict(selected_steps=control_steps,**control)=={k:v for k,v in prior_control.items() if k!='lambda_value'}
    paired=[dict(index=i,selected_step=k,control_step=c,delta_psnr=q['psnr'][k]-q['psnr'][c],delta_ssim=q['ssim'][k]-q['ssim'][c]) for i,(q,k,c) in enumerate(zip(quality,selected,control_steps))]
    deltas=dict(mean_delta_psnr=float(np.mean([r['delta_psnr'] for r in paired])),median_delta_psnr=float(np.median([r['delta_psnr'] for r in paired])),mean_delta_ssim=float(np.mean([r['delta_ssim'] for r in paired])))
    changed=sum(k!=c for k,c in zip(selected,control_steps));verdict='OBJECTIVE_MOTION_KNEE_DEV_CANDIDATE_FROZEN' if changed and all(metrics['gates'].values()) else 'OBJECTIVE_MOTION_KNEE_DEV_NO_GAIN'
    write(out/'selected_choices.json',rows);write(out/'paired_control.json',paired)
    result=dict(classification=verdict,metrics=metrics,control_metrics=control,deltas_vs_control=deltas,changed_choices_vs_control=changed,selected_step_histogram=dict(sorted(Counter(selected).items())),changed_selected_histogram=dict(sorted(Counter(k for k,c in zip(selected,control_steps) if k!=c).items())),control_exact=True,optimizer_runs=0,model_fits=0,completed_utc=utc(),seconds=time.perf_counter()-start)
    write(out/'rule_manifest.json',dict(lambda_value=LAMBDA,rho=RHO,probability_threshold=.5,epsilon=1e-12,formula='argmax(u_k-v_k,k), endpoint on stated degeneracies',motion='float64 sqrt(mean((I_j-I_j-1)^2)); cumulative sum',probability_model_sha256=model_hash,rule_sha256=rule_hash,interpolation_rule_sha256=interpolation_hash,candidate_freeze_sha256=sha(out/'candidate_freeze.json'),selected_choices_sha256=sha(out/'selected_choices.json'),classification=verdict,transfer_authorized=False,frozen_utc=utc()))
    write(out/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
