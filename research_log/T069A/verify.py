import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from research_log.T063A.common import sha,thash,write,utc
from research_log.T066A.run import state_input
from ttie.common_gain import CommonBox
from types import SimpleNamespace
from research_log.T066A.verify import independent_predict
from research_log.T067B.verify import independent_choices
from research_log.T069A.run import HERE,MODEL,inputs,render,INDEPENDENT_LABELS

def recompute(trace,fs,k):
    end=trace['states'][k][:,:2].double().numpy()
    if k==0:return dict(k_FS=fs,k_lambda=k,s_prev=None,p=None,s_end=end.tolist(),norm_prop=0.,norm_clip=0.,R_proj=0.,reason='no_transition')
    prev=trace['states'][k-1][:,:2].double().numpy();proposal=trace['pre_box'][k-1][:,:2].double().numpy()
    pn=math.sqrt(math.fsum(float(x)**2 for x in (proposal-prev).ravel()));cn=math.sqrt(math.fsum(float(x)**2 for x in (proposal-end).ravel()))
    return dict(k_FS=fs,k_lambda=k,s_prev=prev.tolist(),p=proposal.tolist(),s_end=end.tolist(),norm_prop=pn,norm_clip=cn,R_proj=cn/max(pn,1e-12) if pn>1e-12 else 0.,reason='formula' if pn>1e-12 else 'zero_proposal')

def main(out):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    get=lambda n:json.loads((out/n).read_bytes());cfg=get('config.json')
    for p,h in cfg['source_binding'].items():assert sha(p)==h,p
    dev=get('development_freeze.json');trans=get('transfer_freeze.json');opened=get('reference_open.json')
    assert dev['reference_reads']==trans['reference_reads']==0 and len(dev['rows'])==len(trans['rows'])==100
    assert dev['frozen_utc']<trans['frozen_utc']<opened['first_transfer_quality_read_utc'] and sha(out/'transfer_freeze.json')==opened['freeze_sha256'] and sha(out/'development_freeze.json')==trans['development_freeze_sha256']
    model=json.loads(MODEL.read_bytes());error=0.;renders=0;projection_checks=0;computed=[]
    for transfer,frozen in [(False,dev),(True,trans)]:
        raw,recs,data,ends=inputs(transfer);scores=[]
        for rec,d,end,row in zip(recs,data,ends,frozen['rows']):
            probs=independent_predict(d['features'],model);fs=min(k for k in range(d['base_step']+1) if probs[k]>=.5);kl=dict(independent_choices(d['totals'],probs,d['base_step']))[.875]
            assert end['k_FS']==fs and (end['selected_step'] if transfer else end['k_lambda'])==kl
            low,trace=state_input(raw,rec,transfer);np.testing.assert_allclose(d['totals'],trace['values'][:28],rtol=0,atol=1e-7)
            images=render(low,trace,kl,kl,d['hashes']);renders+=len(images)
            if kl:
                gate=SimpleNamespace(active=trace['active'].cuda(),winner=trace['winner'].cuda());box=CommonBox(gate,2)
                assert torch.equal(box.lower.cpu(),trace['lower']) and torch.equal(box.upper.cpu(),trace['upper'])
                proposal=trace['pre_box'][kl-1].cuda().clone();box(SimpleNamespace(raw=proposal))
                assert torch.equal(proposal[:,:2].cpu(),trace['states'][kl][:,:2]);projection_checks+=1
            assert row['previous_state_hash']==(thash(trace['states'][kl-1]) if kl else None) and row['pre_box_hash']==(thash(trace['pre_box'][kl-1]) if kl else None)
            assert row['lower_hash']==thash(trace['lower']) and row['upper_hash']==thash(trace['upper'])
            expected=recompute(trace,fs,kl)
            for key,val in expected.items():
                if key in ['reason','k_FS','k_lambda'] or val is None:assert row[key]==val
                else:
                    err=float(np.max(np.abs(np.asarray(val)-np.asarray(row[key]))));error=max(error,err);assert err<1e-12,key
            assert row['index']==rec['index'] and row['low']==rec['low'] and row['low_sha256']==rec['low_sha256'] and row['reference_reads']==0
            expected_trace=rec['methods']['T063']['trace_sha256'] if transfer else rec['files']['trace.pt'];assert row['trace_sha256']==expected_trace
            assert row['state_hash']==thash(trace['states'][kl]) and row['output_hash']==thash(images[kl]) and row['interval_output_hashes']==d['hashes'][fs:kl+1]
            assert row['model_sha256']==sha(MODEL) and row['rule_sha256']==sha(HERE/'core.py') and row['interpolation_rule_sha256']==sha('research_log/T067B/core.py')
            scores.append(expected['R_proj'])
        for row in frozen['rows']:
            assert row['descending_rank']==1+sum(r['R_proj']>row['R_proj'] for r in frozen['rows']) and row['empirical_percentile']==sum(r['R_proj']<=row['R_proj'] for r in frozen['rows'])/100
        computed.append(scores)
    t99=sorted(r['R_proj'] for r in dev['rows'])[98];independent_t99=sorted(computed[0])[98]
    assert t99==dev['T99_proj']==trans['T99_proj'] and abs(t99-independent_t99)<1e-12
    for row,r in zip(trans['rows'],computed[1]):assert row['T99_proj']==t99 and row['above_T99_proj']==(row['R_proj']>t99)==(r>independent_t99)
    for p,h in json.loads((HERE/'evaluation_binding.json').read_bytes()).items():assert sha(p)==h,p
    labels=json.loads(INDEPENDENT_LABELS.read_bytes());joined=get('joined_endpoints.json');assert len(joined)==100
    unsafe=[];tp=fp=0
    for row,frozen in zip(joined,trans['rows']):
        assert all(row[k]==v for k,v in frozen.items());q=labels[row['index']*28+row['k_lambda']]
        assert q['index']==row['index'] and q['step']==row['k_lambda'] and abs(q['quality_margin']-row['margin'])<1e-10
        assert row['unsafe']==(q['quality_margin']< -5.614)
        if row['unsafe']:unsafe.append(row);tp+=row['above_T99_proj']
        else:fp+=row['above_T99_proj']
    assert unsafe;verdict='PROJECTION_PRESSURE_SIGNAL_PRESENT' if tp==len(unsafe) and fp<=5 else 'PROJECTION_PRESSURE_SIGNAL_ABSENT';result=get('result.json')
    assert result['classification']==verdict and result['T99_proj']==t99 and result['unsafe_endpoints']==len(unsafe) and result['unsafe_above_T99']==tp and result['safe_endpoints']==100-len(unsafe) and result['safe_above_T99']==fp and result['unsafe_rows']==unsafe and result['development_above_T99_proj']==sum(r['R_proj']>t99 for r in dev['rows'])
    write(out/'verification.json',dict(status='PASS',classification=verdict,development_endpoints=100,transfer_endpoints=100,gpu_endpoint_renders=renders,exact_CommonBox_projection_checks=projection_checks,score_max_error=error,independent_T99_proj=independent_t99,T99_proj_error=abs(t99-independent_t99),optimizer_runs=0,model_fits=0,verified_utc=utc()));print(json.dumps(get('verification.json')),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
