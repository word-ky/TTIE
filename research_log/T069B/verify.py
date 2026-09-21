import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from research_log.T063A.common import sha,thash,write,utc
from research_log.T066A.run import state_input
from research_log.T066A.run import CommonRegion2
from research_log.T062A.core import losses
from research_log.T069B.core import TOTAL_ATOL,TOTAL_RTOL
from research_log.T066A.verify import independent_predict
from research_log.T067B.verify import independent_choices
from research_log.T069B.run import HERE,MODEL,inputs,INDEPENDENT_LABELS

def recompute(low,trace,k,expected):
    gradients=[];x=low.cuda()
    for c in range(4):
        renderer=CommonRegion2(trace['active']).to(x).eval()
        with torch.no_grad():renderer.raw.copy_(trace['states'][k].to(x))
        y=renderer(x);assert thash(y.detach().cpu())==expected
        parts=losses(x,y)
        loss=parts[c]*[1.,10.,5.][c] if c<3 else parts @ parts.new_tensor([1.,10.,5.])
        gradients.append(torch.autograd.grad(loss,renderer.raw)[0].detach().cpu().reshape(-1))
    gs=[g.double().tolist() for g in gradients[:3]];direct=gradients[3].double().tolist()
    norm=lambda v:math.sqrt(math.fsum(x*x for x in v))
    ns=[norm(g) for g in gs];total=[math.fsum(g[j] for g in gs) for j in range(12)];den=math.fsum(ns);tn=norm(total)
    diff=[abs(a-b) for a,b in zip(total,direct)]
    assert all(e<=TOTAL_ATOL+TOTAL_RTOL*abs(d) for e,d in zip(diff,direct))
    return dict(gradients=gs,gradient_norms=ns,summed_gradient=total,summed_gradient_norm=tn,sum_component_norms=den,R_cancel=0. if den<=1e-12 else 1-tn/max(den,1e-12),reason='zero_component_gradient' if den<=1e-12 else 'formula',gradient_hashes=[thash(g) for g in gradients[:3]],direct_total_gradient=direct,direct_total_gradient_hash=thash(gradients[3]),direct_total_gradient_norm=norm(direct),total_consistency_max_error=max(diff))

def main(out):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    get=lambda n:json.loads((out/n).read_bytes());cfg=get('config.json')
    for p,h in cfg['source_binding'].items():assert sha(p)==h,p
    dev=get('development_freeze.json');trans=get('transfer_freeze.json');opened=get('reference_open.json')
    assert dev['reference_reads']==trans['reference_reads']==0 and len(dev['rows'])==len(trans['rows'])==100
    assert dev['frozen_utc']<trans['frozen_utc']<opened['first_transfer_quality_read_utc'] and sha(out/'transfer_freeze.json')==opened['freeze_sha256'] and sha(out/'development_freeze.json')==trans['development_freeze_sha256']
    model=json.loads(MODEL.read_bytes());error=0.;renders=0;gradient_error=0.;total_error=0.;computed=[]
    for transfer,frozen in [(False,dev),(True,trans)]:
        raw,recs,data,ends=inputs(transfer);scores=[]
        for rec,d,end,row in zip(recs,data,ends,frozen['rows']):
            probs=independent_predict(d['features'],model);fs=min(k for k in range(d['base_step']+1) if probs[k]>=.5);kl=dict(independent_choices(d['totals'],probs,d['base_step']))[.875]
            assert end['k_FS']==fs and (end['selected_step'] if transfer else end['k_lambda'])==kl
            low,trace=state_input(raw,rec,transfer);np.testing.assert_allclose(d['totals'],trace['values'][:28],rtol=0,atol=1e-7)
            expected=recompute(low,trace,kl,d['hashes'][kl]);renders+=4
            assert row['k_FS']==fs and row['k_lambda']==kl
            for key,val in expected.items():
                if key in ['reason','gradient_hashes','direct_total_gradient_hash']:assert row[key]==val,key
                else:
                    err=float(np.max(np.abs(np.asarray(val)-np.asarray(row[key]))))
                    if key in ['gradients','direct_total_gradient']:
                        gradient_error=max(gradient_error,err);np.testing.assert_allclose(row[key],val,rtol=1e-6,atol=1e-7)
                    else:error=max(error,err);assert err<1e-12,key
            total_error=max(total_error,expected['total_consistency_max_error'])
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
    write(out/'verification.json',dict(status='PASS',classification=verdict,development_endpoints=100,transfer_endpoints=100,gpu_endpoint_renders=renders,gradient_reproduction_max_error=gradient_error,total_consistency_max_error=total_error,score_max_error=error,independent_T99_cancel=independent_t99,T99_cancel_error=abs(t99-independent_t99),optimizer_runs=0,model_fits=0,verified_utc=utc()));print(json.dumps(get('verification.json')),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
