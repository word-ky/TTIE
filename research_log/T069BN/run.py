import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from research_log.T063A.common import sha,thash,write,utc
from research_log.T066A.run import load_inputs,state_input,CommonRegion2
from research_log.T066A.core import predict
from research_log.T067B.core import choices
from research_log.T062A.core import losses
from research_log.T069B.core import weighted_gradients
from research_log.T069BN.core import comparison,bookkeeping,summarize
HERE=Path('research_log/T069BN')
MODEL=Path('research_log/T066A/evidence/model.json')
DEV=Path('research_log/T066A/evidence/development_features.json')
ENDS=Path('research_log/T068B/evidence/candidate_freeze.json')

def inputs():
    raw,items,recs=load_inputs(False)
    data=json.loads(DEV.read_bytes())['rows'];end=json.loads(ENDS.read_bytes())
    assert end['reference_reads']==0 and len(data)==len(end['rows'])==100
    return raw,items,recs,data,end['rows']

def tensor_record(t):
    t=t.detach().cpu().reshape(-1)
    assert t.numel()==12 and torch.isfinite(t).all()
    return dict(values=t.double().tolist(),sha256=thash(t),norm=float(np.linalg.norm(t.double().numpy())),dtype=str(t.dtype))

def compute(low,trace,k,dtype):
    x=low.cuda().to(dtype);renderer=CommonRegion2(trace['active']).to(x).eval()
    with torch.no_grad():renderer.raw.copy_(trace['states'][k].to(x))
    y=renderer(x);output_hash=thash(y.detach().cpu());parts=losses(x,y)
    gs=weighted_gradients(parts,renderer.raw)
    direct=torch.autograd.grad(parts @ parts.new_tensor([1.,10.,5.]),renderer.raw)[0]
    records=[tensor_record(g) for g in gs];dr=tensor_record(direct)
    total=torch.stack([g.detach().cpu().double().reshape(-1) for g in gs]).sum(0)
    return dict(output_hash=output_hash,components=records,direct=dr,sum=tensor_record(total))

def values(rec):return [g['values'] for g in rec['components']],rec['direct']['values']

def main(out):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    binding=json.loads((HERE/'binding.json').read_bytes())
    for p,h in binding.items():assert sha(p)==h,p
    out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    write(out/'config.json',dict(task='T069-BN',source_commit=os.environ['TTIE_SOURCE_COMMIT'],source_binding=binding,started_utc=utc(),reference_reads=0,optimizer_runs=0,model_fits=0,device=torch.cuda.get_device_name(),torch_version=torch.__version__))
    raw,items,recs,data,ends=inputs();model=json.loads(MODEL.read_bytes());rows=[]
    for rec,item,d,end in zip(recs,items,data,ends):
        assert rec['index']==d['index']==end['index'] and rec['low']==item['low']==d['low']==end['low']
        low,trace=state_input(raw,rec,False)
        np.testing.assert_allclose(d['totals'],trace['values'][:28],rtol=0,atol=1e-7)
        choice=next(c for c in choices(d['totals'],predict(d['features'],model).tolist(),d['base_step']) if c['lambda_value']==.875)
        fs=choice['k_FS'];k=choice['selected_step']
        assert end['k_FS']==fs and end['k_lambda']==k and end['low_sha256']==rec['low_sha256'] and end['trace_sha256']==rec['files']['trace.pt']
        assert end['interval_output_hashes']==d['hashes'][fs:k+1] and end['model_sha256']==sha(MODEL)
        stored=tensor_record(trace['gradients'][k]);first=compute(low,trace,k,torch.float32);second=compute(low,trace,k,torch.float32)
        assert first['output_hash']==second['output_hash']==d['hashes'][k]==rec['rendered_hashes'][k]
        a=bookkeeping(*values(first),stored['values'])
        rep=dict(direct_equal=first['direct']['sha256']==second['direct']['sha256'],components_equal=[g['sha256']==h['sha256'] for g,h in zip(first['components'],second['components'])],direct_comparison=comparison(second['direct']['values'],first['direct']['values']))
        try:
            double=compute(low,trace,k,torch.float64)
            control=dict(status='supported',gradients=double,audit=bookkeeping(*values(double)))
        except (NotImplementedError,RuntimeError) as exc:
            if 'not implemented' not in str(exc).lower() or 'double' not in str(exc).lower():raise
            control=dict(status='double_control_unsupported',reason=str(exc))
        row=dict(index=rec['index'],low=rec['low'],low_sha256=rec['low_sha256'],trace_sha256=rec['files']['trace.pt'],k_FS=fs,k_lambda=k,state_hash=thash(trace['states'][k]),output_hash=first['output_hash'],model_sha256=sha(MODEL),rule_sha256=sha(HERE/'core.py'),interpolation_rule_sha256=sha('research_log/T067B/core.py'),stored=stored,first=first,second=second,float32=a,repeat=rep,float64=control,reference_reads=0)
        rows.append(row)
    assert len(rows)==100
    write(out/'development_freeze.json',dict(rows=rows,reference_reads=0,frozen_utc=utc(),source_commit=os.environ['TTIE_SOURCE_COMMIT']))
    result=summarize(rows);result.update(completed_utc=utc(),seconds=time.perf_counter()-start,development_freeze_sha256=sha(out/'development_freeze.json'))
    write(out/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
