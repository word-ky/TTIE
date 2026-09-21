import argparse,json,math
from pathlib import Path
import torch
from research_log.T063A.common import sha,thash,write,utc
from research_log.T066A.run import state_input,CommonRegion2
from research_log.T066A.verify import independent_predict
from research_log.T067B.verify import independent_choices
from research_log.T062A.core import losses
from research_log.T069BN.run import HERE,MODEL,inputs

norm=lambda v:math.sqrt(math.fsum(float(x)**2 for x in v))
def record(t):
    t=t.detach().cpu().reshape(-1);v=t.double().tolist()
    assert len(v)==12 and all(math.isfinite(x) for x in v)
    return dict(values=v,sha256=thash(t),norm=norm(v),dtype=str(t.dtype))
def compute(low,trace,k,dtype):
    gs=[];hashes=[]
    for c in range(4):
        x=low.cuda().to(dtype);m=CommonRegion2(trace['active']).to(x).eval()
        with torch.no_grad():m.raw.copy_(trace['states'][k].to(x))
        y=m(x);hashes.append(thash(y.detach().cpu()));ls=losses(x,y)
        value=ls[c]*[1.,10.,5.][c] if c<3 else ls @ ls.new_tensor([1.,10.,5.])
        gs.append(torch.autograd.grad(value,m.raw)[0].detach().cpu().reshape(-1))
    assert len(set(hashes))==1
    total=torch.stack([g.double() for g in gs[:3]]).sum(0)
    return dict(output_hash=hashes[0],components=[record(g) for g in gs[:3]],direct=record(gs[3]),sum=record(total))
def compare(a,b):
    diff=[abs(x-y) for x,y in zip(a,b)];fails=[d>2e-7+2e-5*abs(y) for d,y in zip(diff,b)]
    return dict(abs_residual=diff,max_abs=max(diff),l2_relative=norm(diff)/max(norm(b),1e-12),failed_coordinates=sum(fails),passes=not any(fails))
def stats(g,stored=None):
    ns=[norm(c['values']) for c in g['components']];den=math.fsum(ns)
    total=[math.fsum(c['values'][j] for c in g['components']) for j in range(12)]
    rc=0. if den<=1e-12 else 1-norm(total)/max(den,1e-12)
    rd=1-norm(g['direct']['values'])/max(den,1e-12)
    result=dict(sum_vs_direct=compare(total,g['direct']['values']),R_cancel=rc,R_directnorm=rd,score_sensitivity=abs(rc-rd))
    if stored is not None:result['direct_vs_trace']=compare(g['direct']['values'],stored['values'])
    return result

def equal(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    elif isinstance(a,float):assert abs(a-b)<=1e-12+1e-10*abs(a),(a,b)
    else:assert a==b,(a,b)

def distribution(values):
    v=sorted(values)
    def pct(p):
        pos=(len(v)-1)*p;i=math.floor(pos);j=math.ceil(pos)
        return v[i]+(v[j]-v[i])*(pos-i)
    return dict(max=v[-1],median=pct(.5),p95=pct(.95))
def summary(rows):
    result=dict(status='GRADIENT_NUMERICS_CHARACTERIZED' if all(r['float32']['direct_vs_trace']['passes'] and r['repeat']['direct_comparison']['passes'] for r in rows) else 'GRADIENT_PATH_MISMATCH',rows=len(rows),reference_reads=0,optimizer_runs=0,model_fits=0)
    for name in ['direct_vs_trace','sum_vs_direct']:
        cs=[r['float32'][name] for r in rows]
        result[name]=dict(absolute_coordinates=distribution([v for c in cs for v in c['abs_residual']]),absolute_row_max=distribution([c['max_abs'] for c in cs]),l2_relative=distribution([c['l2_relative'] for c in cs]),failed_rows=sum(not c['passes'] for c in cs),failed_coordinates=sum(c['failed_coordinates'] for c in cs))
    result['score_sensitivity']=distribution([r['float32']['score_sensitivity'] for r in rows])
    result['repeat']=dict(exact_direct_rows=sum(r['repeat']['direct_equal'] for r in rows),exact_component_rows=sum(all(r['repeat']['components_equal']) for r in rows),direct_tolerance_failed_rows=sum(not r['repeat']['direct_comparison']['passes'] for r in rows))
    ds=[r['float64']['audit'] for r in rows if r['float64']['status']=='supported']
    result['float64']=dict(supported_rows=len(ds),unsupported_rows=len(rows)-len(ds))
    if ds:result['float64'].update(sum_vs_direct_abs=distribution([x for d in ds for x in d['sum_vs_direct']['abs_residual']]),sum_vs_direct_l2_relative=distribution([d['sum_vs_direct']['l2_relative'] for d in ds]),score_sensitivity=distribution([d['score_sensitivity'] for d in ds]))
    return result

def main(out):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    get=lambda n:json.loads((out/n).read_bytes());cfg=get('config.json');frozen=get('development_freeze.json');result=get('result.json')
    for p,h in cfg['source_binding'].items():assert sha(p)==h,p
    assert frozen['reference_reads']==0 and sha(out/'development_freeze.json')==result['development_freeze_sha256'] and len(frozen['rows'])==100
    raw,items,recs,data,ends=inputs();model=json.loads(MODEL.read_bytes());rows=[];renders=0
    for rec,item,d,end,row in zip(recs,items,data,ends,frozen['rows']):
        probs=independent_predict(d['features'],model);fs=min(k for k in range(d['base_step']+1) if probs[k]>=.5);k=dict(independent_choices(d['totals'],probs,d['base_step']))[.875]
        low,trace=state_input(raw,rec,False)
        assert row['index']==rec['index']==d['index']==end['index'] and row['low']==rec['low']==item['low']==d['low']==end['low']
        assert row['k_FS']==fs==end['k_FS'] and row['k_lambda']==k==end['k_lambda']
        assert row['low_sha256']==rec['low_sha256']==end['low_sha256'] and row['trace_sha256']==rec['files']['trace.pt']==end['trace_sha256'] and row['state_hash']==thash(trace['states'][k])
        assert row['model_sha256']==sha(MODEL)==end['model_sha256'] and row['rule_sha256']==sha(HERE/'core.py') and row['interpolation_rule_sha256']==sha('research_log/T067B/core.py')
        assert end['interval_output_hashes']==d['hashes'][fs:k+1] and row['reference_reads']==0
        stored=record(trace['gradients'][k]);first=compute(low,trace,k,torch.float32);second=compute(low,trace,k,torch.float32);renders+=8
        equal(stored,row['stored']);equal(first,row['first']);equal(second,row['second'])
        assert first['output_hash']==second['output_hash']==row['output_hash']==d['hashes'][k]==rec['rendered_hashes'][k]
        audit=stats(first,stored);equal(audit,row['float32'])
        rep=dict(direct_equal=first['direct']['sha256']==second['direct']['sha256'],components_equal=[g['sha256']==h['sha256'] for g,h in zip(first['components'],second['components'])],direct_comparison=compare(second['direct']['values'],first['direct']['values']))
        equal(rep,row['repeat'])
        try:
            dg=compute(low,trace,k,torch.float64);renders+=4;control=dict(status='supported',gradients=dg,audit=stats(dg))
        except (NotImplementedError,RuntimeError) as exc:
            if 'not implemented' not in str(exc).lower() or 'double' not in str(exc).lower():raise
            control=dict(status='double_control_unsupported',reason=str(exc))
        equal(control,row['float64']);rows.append(dict(float32=audit,repeat=rep,float64=control))
    expected=summary(rows)
    for key,val in expected.items():equal(val,result[key])
    receipt=dict(status='PASS',audit_status=expected['status'],development_endpoints=100,independent_gpu_forwards=renders,gradient_and_output_hashes_exact=True,reference_reads=0,optimizer_runs=0,model_fits=0,verified_utc=utc())
    write(out/'verification.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
