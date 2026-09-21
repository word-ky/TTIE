import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from research_log.T063A.common import sha,thash,write,utc
from research_log.T066A.run import state_input
from research_log.T066A.run import CommonRegion2
from research_log.T062A.core import losses
from research_log.T069BN.verify import compute,record,equal
from research_log.T066A.verify import independent_predict
from research_log.T067B.verify import independent_choices
from research_log.T069BR.run import HERE,MODEL,inputs,INDEPENDENT_LABELS

def recompute(low,trace,k,expected):
    x=low.cuda();m=CommonRegion2(trace['active']).to(x).eval()
    with torch.no_grad():m.raw.copy_(trace['states'][k].to(x))
    y=m(x);assert thash(y.detach().cpu())==expected
    parts=losses(x,y);value=parts @ parts.new_tensor([1.,10.,5.])
    direct32=record(torch.autograd.grad(value,m.raw)[0]);stored=record(trace['gradients'][k])
    dif32=[abs(a-b) for a,b in zip(direct32['values'],stored['values'])]
    pc=dict(abs_residual=dif32,max_abs=max(dif32),passes=all(e<=2e-7+2e-5*abs(d) for e,d in zip(dif32,stored['values'])))
    assert pc['passes']
    double=compute(low,trace,k,torch.float64);gs=[g['values'] for g in double['components']];direct=double['direct']['values']
    norm=lambda v:math.sqrt(math.fsum(x*x for x in v))
    ns=[norm(g) for g in gs];total=[math.fsum(g[j] for g in gs) for j in range(12)];den=math.fsum(ns);tn=norm(total)
    diff=[abs(a-b) for a,b in zip(total,direct)]
    dc=dict(abs_residual=diff,max_abs=max(diff),passes=all(e<=1e-12+1e-10*abs(d) for e,d in zip(diff,direct)))
    assert dc['passes']
    return dict(gradients=gs,gradient_norms=ns,summed_gradient=total,summed_gradient_norm=tn,sum_component_norms=den,R_cancel=0. if den<=1e-12 else 1-tn/max(den,1e-12),reason='zero_component_gradient' if den<=1e-12 else 'formula',gradient_hashes=[g['sha256'] for g in double['components']],direct_total_gradient=direct,direct_total_gradient_hash=double['direct']['sha256'],direct_total_gradient_norm=norm(direct),total_consistency_max_error=max(diff),double_check=dc,output64_hash=double['output_hash'],path32=dict(direct=direct32,stored=stored,check=pc))

def main(out):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    get=lambda n:json.loads((out/n).read_bytes());cfg=get('config.json')
    for p,h in cfg['source_binding'].items():assert sha(p)==h,p
    dev=get('development_freeze.json');trans=get('transfer_freeze.json');opened=get('reference_open.json')
    assert dev['reference_reads']==trans['reference_reads']==0 and len(dev['rows'])==len(trans['rows'])==100
    assert dev['frozen_utc']<trans['frozen_utc']<opened['first_transfer_quality_read_utc'] and sha(out/'transfer_freeze.json')==opened['freeze_sha256'] and sha(out/'development_freeze.json')==trans['development_freeze_sha256']
    model=json.loads(MODEL.read_bytes());error=0.;renders=0;gradient_error=0.;total_error=0.;path_error=0.;computed=[]
    for transfer,frozen in [(False,dev),(True,trans)]:
        raw,recs,data,ends=inputs(transfer);scores=[]
        for rec,d,end,row in zip(recs,data,ends,frozen['rows']):
            probs=independent_predict(d['features'],model);fs=min(k for k in range(d['base_step']+1) if probs[k]>=.5);kl=dict(independent_choices(d['totals'],probs,d['base_step']))[.875]
            assert end['k_FS']==fs and (end['selected_step'] if transfer else end['k_lambda'])==kl
            low,trace=state_input(raw,rec,transfer);np.testing.assert_allclose(d['totals'],trace['values'][:28],rtol=0,atol=1e-7)
            expected=recompute(low,trace,kl,d['hashes'][kl]);renders+=5
            assert row['k_FS']==fs and row['k_lambda']==kl
            for key,val in expected.items():equal(val,row[key])
            error=max(error,abs(expected['R_cancel']-row['R_cancel']))
            gradient_error=max(gradient_error,float(np.max(np.abs(np.asarray(expected['gradients'])-np.asarray(row['gradients'])))))
            total_error=max(total_error,expected['total_consistency_max_error']);path_error=max(path_error,expected['path32']['check']['max_abs'])
            assert row['index']==rec['index'] and row['low']==rec['low'] and row['low_sha256']==rec['low_sha256'] and row['reference_reads']==0
            expected_trace=rec['methods']['T063']['trace_sha256'] if transfer else rec['files']['trace.pt'];assert row['trace_sha256']==expected_trace
            assert row['state_hash']==thash(trace['states'][kl]) and row['output_hash']==d['hashes'][kl] and row['interval_output_hashes']==d['hashes'][fs:kl+1]
            assert row['model_sha256']==sha(MODEL) and row['rule_sha256']==sha(HERE/'core.py') and row['interpolation_rule_sha256']==sha('research_log/T067B/core.py')
            scores.append(expected['R_cancel'])
        for row in frozen['rows']:
            assert row['descending_rank']==1+sum(r['R_cancel']>row['R_cancel'] for r in frozen['rows']) and row['empirical_percentile']==sum(r['R_cancel']<=row['R_cancel'] for r in frozen['rows'])/100
        computed.append(scores)
    t99=sorted(r['R_cancel'] for r in dev['rows'])[98];independent_t99=sorted(computed[0])[98]
    assert t99==dev['T99_cancel']==trans['T99_cancel'] and abs(t99-independent_t99)<1e-12
    for row,r in zip(trans['rows'],computed[1]):assert row['T99_cancel']==t99 and row['above_T99_cancel']==(row['R_cancel']>t99)==(r>independent_t99)
    for p,h in json.loads((HERE/'evaluation_binding.json').read_bytes()).items():assert sha(p)==h,p
    labels=json.loads(INDEPENDENT_LABELS.read_bytes());joined=get('joined_endpoints.json');assert len(joined)==100
    unsafe=[];tp=fp=0
    for row,frozen in zip(joined,trans['rows']):
        assert all(row[k]==v for k,v in frozen.items());q=labels[row['index']*28+row['k_lambda']]
        assert q['index']==row['index'] and q['step']==row['k_lambda'] and abs(q['quality_margin']-row['margin'])<1e-10
        assert row['unsafe']==(q['quality_margin']< -5.614)
        if row['unsafe']:unsafe.append(row);tp+=row['above_T99_cancel']
        else:fp+=row['above_T99_cancel']
    assert unsafe;verdict='GRADIENT_CANCELLATION_SIGNAL_PRESENT' if tp==len(unsafe) and fp<=5 else 'GRADIENT_CANCELLATION_SIGNAL_ABSENT';result=get('result.json')
    assert result['classification']==verdict and result['T99_cancel']==t99 and result['unsafe_endpoints']==len(unsafe) and result['unsafe_above_T99']==tp and result['safe_endpoints']==100-len(unsafe) and result['safe_above_T99']==fp and result['unsafe_rows']==unsafe and result['development_above_T99_cancel']==sum(r['R_cancel']>t99 for r in dev['rows'])
    equal(path_error,result['path32_max_abs']);equal(total_error,result['double_sum_max_abs'])
    for name,values in zip(['development','transfer'],computed):
        ordered=sorted(values);pos=99*.95;i=int(pos);j=math.ceil(pos)
        dist=dict(min=ordered[0],median=(ordered[49]+ordered[50])/2,p95=ordered[i]+(ordered[j]-ordered[i])*(pos-i),max=ordered[-1])
        equal(dist,result['score_distributions'][name])
    write(out/'verification.json',dict(status='PASS',classification=verdict,development_endpoints=100,transfer_endpoints=100,gpu_endpoint_renders=renders,path32_max_abs=path_error,float32_trace_checks=200,float64_linearity_checks=200,gradient_reproduction_max_error=gradient_error,total_consistency_max_error=total_error,score_max_error=error,independent_T99_cancel=independent_t99,T99_cancel_error=abs(t99-independent_t99),optimizer_runs=0,model_fits=0,verified_utc=utc()));print(json.dumps(get('verification.json')),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
