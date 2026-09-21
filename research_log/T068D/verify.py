import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from research_log.T063A.common import sha,thash,write,utc
from research_log.T066A.run import state_input
from research_log.T062A.core import losses
from research_log.T066A.verify import independent_predict
from research_log.T067B.verify import independent_choices
from research_log.T068D.run import HERE,MODEL,inputs,render,INDEPENDENT_LABELS

def recompute(components,fs,kl):
    weights=[1.,10.,5.];z=[[float(row[c])*weights[c] for c in range(3)] for row in components]
    lo=[min(row[c] for row in z) for c in range(3)];hi=[max(row[c] for row in z) for c in range(3)]
    a=[z[-1][c]-lo[c] for c in range(3)];e=[hi[c]-lo[c] for c in range(3)];den=sum(e)
    if fs==kl:r=0.;reason='singleton'
    elif den<=1e-12:r=0.;reason='zero_excursion'
    else:r=sum(a)/max(den,1e-12);reason='formula'
    return dict(k_FS=fs,k_lambda=kl,Z=z,endpoint_Z=z[-1],zmin=lo,zmax=hi,a=a,e=e,sum_a=sum(a),sum_e=den,R_comp=r,reason=reason)

def main(out):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    get=lambda n:json.loads((out/n).read_bytes());cfg=get('config.json')
    for p,h in cfg['source_binding'].items():assert sha(p)==h,p
    dev=get('development_freeze.json');trans=get('transfer_freeze.json');opened=get('reference_open.json')
    assert dev['reference_reads']==trans['reference_reads']==0 and len(dev['rows'])==len(trans['rows'])==100
    assert dev['frozen_utc']<trans['frozen_utc']<opened['first_transfer_quality_read_utc'] and sha(out/'transfer_freeze.json')==opened['freeze_sha256'] and sha(out/'development_freeze.json')==trans['development_freeze_sha256']
    model=json.loads(MODEL.read_bytes());error=0.;renders=0;component_error=0.;computed=[]
    for transfer,frozen in [(False,dev),(True,trans)]:
        raw,recs,data,ends=inputs(transfer);scores=[]
        for rec,d,end,row in zip(recs,data,ends,frozen['rows']):
            probs=independent_predict(d['features'],model);fs=min(k for k in range(d['base_step']+1) if probs[k]>=.5);kl=dict(independent_choices(d['totals'],probs,d['base_step']))[.875]
            assert end['k_FS']==fs and (end['selected_step'] if transfer else end['k_lambda'])==kl
            low,trace=state_input(raw,rec,transfer);np.testing.assert_allclose(d['totals'],trace['values'][:28],rtol=0,atol=1e-7)
            images=render(low,trace,fs,kl,d['hashes']);renders+=len(images)
            with torch.no_grad():parts=[losses(low.cuda(),images[k].cuda()).cpu().double().tolist() for k in range(fs,kl+1)]
            ce=float(np.max(np.abs(np.asarray(parts)-np.asarray(d['components'][fs:kl+1]))));component_error=max(component_error,ce)
            np.testing.assert_array_equal(parts,d['components'][fs:kl+1]);np.testing.assert_array_equal(parts,trace['components'][fs:kl+1].double().numpy())
            expected=recompute(parts,fs,kl)
            for key,val in expected.items():
                if key in ['reason','k_FS','k_lambda','k0']:assert row[key]==val
                else:
                    err=float(np.max(np.abs(np.asarray(val)-np.asarray(row[key]))));error=max(error,err);assert err<1e-12,key
            assert row['index']==rec['index'] and row['low']==rec['low'] and row['low_sha256']==rec['low_sha256'] and row['reference_reads']==0
            expected_trace=rec['methods']['T063']['trace_sha256'] if transfer else rec['files']['trace.pt'];assert row['trace_sha256']==expected_trace
            assert row['state_hash']==thash(trace['states'][kl]) and row['output_hash']==thash(images[kl]) and row['interval_output_hashes']==d['hashes'][fs:kl+1]
            assert row['model_sha256']==sha(MODEL) and row['rule_sha256']==sha(HERE/'core.py') and row['interpolation_rule_sha256']==sha('research_log/T067B/core.py')
            scores.append(expected['R_comp'])
        for row in frozen['rows']:
            assert row['descending_rank']==1+sum(r['R_comp']>row['R_comp'] for r in frozen['rows']) and row['empirical_percentile']==sum(r['R_comp']<=row['R_comp'] for r in frozen['rows'])/100
        computed.append(scores)
    t99=sorted(r['R_comp'] for r in dev['rows'])[98];independent_t99=sorted(computed[0])[98]
    assert t99==dev['T99_comp']==trans['T99_comp'] and abs(t99-independent_t99)<1e-12
    for row,r in zip(trans['rows'],computed[1]):assert row['T99_comp']==t99 and row['above_T99_comp']==(row['R_comp']>t99)==(r>independent_t99)
    for p,h in json.loads((HERE/'evaluation_binding.json').read_bytes()).items():assert sha(p)==h,p
    labels=json.loads(INDEPENDENT_LABELS.read_bytes());joined=get('joined_endpoints.json');assert len(joined)==100
    unsafe=[];tp=fp=0
    for row,frozen in zip(joined,trans['rows']):
        assert all(row[k]==v for k,v in frozen.items());q=labels[row['index']*28+row['k_lambda']]
        assert q['index']==row['index'] and q['step']==row['k_lambda'] and abs(q['quality_margin']-row['margin'])<1e-10
        assert row['unsafe']==(q['quality_margin']< -5.614)
        if row['unsafe']:unsafe.append(row);tp+=row['above_T99_comp']
        else:fp+=row['above_T99_comp']
    assert unsafe;verdict='COMPONENT_REGRET_SIGNAL_PRESENT' if tp==len(unsafe) and fp<=5 else 'COMPONENT_REGRET_SIGNAL_ABSENT';result=get('result.json')
    assert result['classification']==verdict and result['T99_comp']==t99 and result['unsafe_endpoints']==len(unsafe) and result['unsafe_above_T99']==tp and result['safe_endpoints']==100-len(unsafe) and result['safe_above_T99']==fp and result['unsafe_rows']==unsafe and result['development_above_T99_comp']==sum(r['R_comp']>t99 for r in dev['rows'])
    write(out/'verification.json',dict(status='PASS',classification=verdict,development_endpoints=100,transfer_endpoints=100,gpu_interval_renders=renders,score_component_max_error=error,direct_loss_component_max_error=component_error,independent_T99_comp=independent_t99,T99_comp_error=abs(t99-independent_t99),optimizer_runs=0,model_fits=0,verified_utc=utc()));print(json.dumps(get('verification.json')),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
