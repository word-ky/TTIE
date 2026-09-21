import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from research_log.T063A.common import sha,thash,write,utc
from research_log.T066A.run import load_inputs,state_input,CommonRegion2
from research_log.T066A.core import predict
from research_log.T067B.core import choices
from research_log.T068B.core import gpu_motions
from research_log.T068C.core import score,threshold,diagnose
HERE=Path('research_log/T068C');MODEL=Path('research_log/T066A/evidence/model.json')
DEV=Path('research_log/T066A/evidence/development_features.json');DEVCURVES=Path('research_log/T068B/evidence/candidate_freeze.json')
TRANSFER=Path('research_log/T066A/evidence/transfer_freeze.json');CHOICES=Path('research_log/T067C/evidence/choice_freeze.json')
LABELS=Path('research_log/T067C/evidence/per_image.json');INDEPENDENT_LABELS=Path('research_log/T066B/evidence/transfer_labels.json')

def inputs(transfer):
    raw,items,recs=load_inputs(transfer);data=json.loads((TRANSFER if transfer else DEV).read_bytes())['rows'];ends=json.loads((CHOICES if transfer else DEVCURVES).read_bytes());assert ends['reference_reads']==0
    return raw,recs,data,ends['rows']

def render(low,trace,fs,kl,expected):
    x=low.cuda();renderer=CommonRegion2(trace['active']).to(x).eval().requires_grad_(False);images={}
    with torch.no_grad():
        for k in range(fs,kl+1):
            renderer.raw.copy_(trace['states'][k].to(x));images[k]=renderer(x).cpu();assert thash(images[k])==expected[k]
    return images

def identity(rec,trace,data,fs,kl,transfer):
    return dict(index=rec['index'],low=rec['low'],low_sha256=rec['low_sha256'],trace_sha256=rec['methods']['T063']['trace_sha256'] if transfer else rec['files']['trace.pt'],state_hash=thash(trace['states'][kl]),output_hash=data['hashes'][kl],interval_output_hashes=data['hashes'][fs:kl+1],model_sha256=sha(MODEL),rule_sha256=sha(HERE/'core.py'),interpolation_rule_sha256=sha('research_log/T067B/core.py'),reference_reads=0)

def endpoint_scores(transfer):
    raw,recs,data,ends=inputs(transfer);model=json.loads(MODEL.read_bytes());rows=[]
    for rec,d,end in zip(recs,data,ends):
        low,trace=state_input(raw,rec,transfer);np.testing.assert_allclose(d['totals'],trace['values'][:28],rtol=0,atol=1e-7)
        probs=predict(d['features'],model).tolist();choice=next(r for r in choices(d['totals'],probs,d['base_step']) if r['lambda_value']==.875);fs=choice['k_FS'];kl=choice['selected_step']
        assert end['index']==rec['index'] and end['low']==rec['low'] and end['low_sha256']==rec['low_sha256'] and end['k_FS']==fs
        assert (end['selected_step'] if transfer else end['k_lambda'])==kl
        if transfer:
            assert end['selected_state_hash']==thash(trace['states'][kl]) and end['output_hash']==d['hashes'][kl]
            images=render(low,trace,fs,kl,d['hashes']);motions=gpu_motions(images,fs,kl)
        else:
            assert end['trace_sha256']==rec['files']['trace.pt'] and end['images_sha256']==rec['files']['images.pt'] and end['interval_output_hashes']==d['hashes'][fs:kl+1] and end['model_sha256']==sha(MODEL)
            motions=end['d_k']
        rows.append(dict(**score(d['totals'],fs,kl,motions),**identity(rec,trace,d,fs,kl,transfer)))
    assert len(rows)==100
    return rows

def add_ranks(rows):
    return [dict(r,descending_rank=1+sum(x['R']>r['R'] for x in rows),empirical_percentile=sum(x['R']<=r['R'] for x in rows)/len(rows)) for r in rows]

def main(out):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    binding=json.loads((HERE/'binding.json').read_bytes())
    for p,h in binding.items():assert sha(p)==h,p
    out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();write(out/'config.json',dict(task='T068-C',source_commit=os.environ['TTIE_SOURCE_COMMIT'],source_binding=binding,started_utc=utc(),optimizer_runs=0,model_fits=0))
    dev=endpoint_scores(False);t99=threshold(dev);write(out/'development_freeze.json',dict(rows=add_ranks(dev),T99=t99,percentile='nearest rank sorted_R[98]',reference_reads=0,frozen_utc=utc()))
    transfer=endpoint_scores(True)
    for r in transfer:r.update(T99=t99,above_T99=r['R']>t99)
    write(out/'transfer_freeze.json',dict(rows=add_ranks(transfer),T99=t99,development_freeze_sha256=sha(out/'development_freeze.json'),reference_reads=0,frozen_utc=utc(),source_commit=os.environ['TTIE_SOURCE_COMMIT']))
    write(out/'reference_open.json',dict(first_transfer_quality_read_utc=utc(),freeze_sha256=sha(out/'transfer_freeze.json')))
    for p,h in json.loads((HERE/'evaluation_binding.json').read_bytes()).items():assert sha(p)==h,p
    labels=json.loads(LABELS.read_bytes());joined=[]
    for row,q in zip(add_ranks(transfer),labels):
        assert q['index']==row['index'] and q['selected_step']==row['k_lambda'] and q['low']==row['low']
        margin=q['metrics']['selected']['psnr']-q['metrics']['T026']['psnr'];joined.append(dict(row,margin=margin,unsafe=margin < -5.614))
    result=dict(**diagnose(joined),T99=t99,development_above_T99=sum(r['R']>t99 for r in dev),unsafe_rows=[r for r in joined if r['unsafe']],optimizer_runs=0,model_fits=0,completed_utc=utc(),seconds=time.perf_counter()-start)
    write(out/'joined_endpoints.json',joined);write(out/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
