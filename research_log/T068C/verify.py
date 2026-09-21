import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from research_log.T063A.common import sha,thash,write,utc
from research_log.T066A.run import state_input
from research_log.T066A.verify import independent_predict
from research_log.T067B.verify import independent_choices
from research_log.T068C.run import HERE,MODEL,inputs,render,INDEPENDENT_LABELS

def recompute(values,fs,kl,images):
    k0=max(fs,kl-3);motions=[0.]
    for k in range(fs+1,kl+1):
        d=images[k].numpy().astype(np.float64)-images[k-1].numpy().astype(np.float64);motions.append(math.sqrt(float(d.ravel()@d.ravel())/d.size))
    dt=float(values[fs]-values[kl]);total=sum(motions);tail=sum(motions[k0-fs+1:]);p=min(1.,max(0.,float(values[k0]-values[kl])/max(dt,1e-12)));q=tail/max(total,1e-12)
    ratio=math.log((q+1e-12)/(p+1e-12))
    if fs==kl:p=q=ratio=0.
    return dict(k_FS=fs,k_lambda=kl,k0=k0,D_total=dt,p_tail=p,m_j=motions,M_total=total,M_tail=tail,q_tail=q,R=ratio,reason='singleton' if fs==kl else 'formula')

def main(out):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    get=lambda n:json.loads((out/n).read_bytes());cfg=get('config.json')
    for p,h in cfg['source_binding'].items():assert sha(p)==h,p
    dev=get('development_freeze.json');trans=get('transfer_freeze.json');opened=get('reference_open.json')
    assert dev['reference_reads']==trans['reference_reads']==0 and len(dev['rows'])==len(trans['rows'])==100
    assert dev['frozen_utc']<trans['frozen_utc']<opened['first_transfer_quality_read_utc'] and sha(out/'transfer_freeze.json')==opened['freeze_sha256'] and sha(out/'development_freeze.json')==trans['development_freeze_sha256']
    model=json.loads(MODEL.read_bytes());error=0.;renders=0;computed=[]
    for transfer,frozen in [(False,dev),(True,trans)]:
        raw,recs,data,ends=inputs(transfer);scores=[]
        for rec,d,end,row in zip(recs,data,ends,frozen['rows']):
            probs=independent_predict(d['features'],model);fs=min(k for k in range(d['base_step']+1) if probs[k]>=.5);kl=dict(independent_choices(d['totals'],probs,d['base_step']))[.875]
            assert end['k_FS']==fs and (end['selected_step'] if transfer else end['k_lambda'])==kl
            low,trace=state_input(raw,rec,transfer);np.testing.assert_allclose(d['totals'],trace['values'][:28],rtol=0,atol=1e-7)
            images=render(low,trace,fs,kl,d['hashes']);renders+=len(images);expected=recompute(d['totals'],fs,kl,images)
            for key,val in expected.items():
                if key in ['reason','k_FS','k_lambda','k0']:assert row[key]==val
                else:
                    err=float(np.max(np.abs(np.asarray(val)-np.asarray(row[key]))));error=max(error,err);assert err<1e-12,key
            assert row['index']==rec['index'] and row['low']==rec['low'] and row['low_sha256']==rec['low_sha256'] and row['reference_reads']==0
            expected_trace=rec['methods']['T063']['trace_sha256'] if transfer else rec['files']['trace.pt'];assert row['trace_sha256']==expected_trace
            assert row['state_hash']==thash(trace['states'][kl]) and row['output_hash']==thash(images[kl]) and row['interval_output_hashes']==d['hashes'][fs:kl+1]
            assert row['model_sha256']==sha(MODEL) and row['rule_sha256']==sha(HERE/'core.py') and row['interpolation_rule_sha256']==sha('research_log/T067B/core.py')
            scores.append(expected['R'])
        for row in frozen['rows']:
            assert row['descending_rank']==1+sum(r['R']>row['R'] for r in frozen['rows']) and row['empirical_percentile']==sum(r['R']<=row['R'] for r in frozen['rows'])/100
        computed.append(scores)
    t99=sorted(r['R'] for r in dev['rows'])[98];independent_t99=sorted(computed[0])[98]
    assert t99==dev['T99']==trans['T99'] and abs(t99-independent_t99)<1e-12
    for row,r in zip(trans['rows'],computed[1]):assert row['T99']==t99 and row['above_T99']==(row['R']>t99)==(r>independent_t99)
    for p,h in json.loads((HERE/'evaluation_binding.json').read_bytes()).items():assert sha(p)==h,p
    labels=json.loads(INDEPENDENT_LABELS.read_bytes());joined=get('joined_endpoints.json');assert len(joined)==100
    unsafe=[];tp=fp=0
    for row,frozen in zip(joined,trans['rows']):
        assert all(row[k]==v for k,v in frozen.items());q=labels[row['index']*28+row['k_lambda']]
        assert q['index']==row['index'] and q['step']==row['k_lambda'] and abs(q['quality_margin']-row['margin'])<1e-10
        assert row['unsafe']==(q['quality_margin']< -5.614)
        if row['unsafe']:unsafe.append(row);tp+=row['above_T99']
        else:fp+=row['above_T99']
    assert unsafe;verdict='TAIL_INEFFICIENCY_SIGNAL_PRESENT' if tp==len(unsafe) and fp<=5 else 'TAIL_INEFFICIENCY_SIGNAL_ABSENT';result=get('result.json')
    assert result['classification']==verdict and result['T99']==t99 and result['unsafe_endpoints']==len(unsafe) and result['unsafe_above_T99']==tp and result['safe_endpoints']==100-len(unsafe) and result['safe_above_T99']==fp and result['unsafe_rows']==unsafe and result['development_above_T99']==sum(r['R']>t99 for r in dev['rows'])
    write(out/'verification.json',dict(status='PASS',classification=verdict,development_endpoints=100,transfer_endpoints=100,gpu_interval_renders=renders,score_component_max_error=error,independent_T99=independent_t99,T99_error=abs(t99-independent_t99),optimizer_runs=0,model_fits=0,verified_utc=utc()));print(json.dumps(get('verification.json')),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
