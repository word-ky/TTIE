import argparse,json
from pathlib import Path
import torch
from research_log.T063A.common import sha,thash,write,utc
from research_log.T066A.run import load_inputs,state_input
from research_log.T066B.run import transfer_labels
from research_log.T067D.run import HERE,PRIOR,CHOICE

def independent_boundary(states):
    flags=[s['safe'] for s in states];first=next((j for j,x in enumerate(flags) if not x),None);end=(first-1 if first is not None else len(states)-1);selected=next(r for r in states if r['selected_flag']);fs=states[0]['k_FS'];after=[r for r in states if r['k']>fs and not r['safe']]
    return dict(index=states[0]['index'],k_FS=fs,k_rho=states[0]['k_rho'],selected_step=selected['k'],any_interval_unsafe=first is not None,first_unsafe_step=states[first]['k'] if first is not None else None,first_unsafe_strictly_after_FS=after[0]['k'] if after else None,first_safe_state_is_reference_safe=bool(flags[0]),contiguous_safe_prefix_end=states[end]['k'] if end>=0 else None,safety_recovers=any(flags[first+1:]) if first is not None else False,q_first_unsafe=states[first]['q_k'] if first is not None else None,q_safe_prefix_end=states[end]['q_k'] if end>=0 else None,selected_boundary_relation=('before' if selected['k']<states[first]['k'] else 'after' if selected['k']>states[first]['k'] else 'on') if first is not None else 'no_unsafe_boundary',selected_margin=selected['margin'],selected_safe=selected['safe'],unsafe_states=flags.count(False))

def main(out):
    import numpy as np
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;read=lambda n:json.loads((out/n).read_bytes());cfg=read('config.json')
    for p,h in cfg['source_binding'].items():assert sha(p)==h,p
    frozen=read('interval_freeze.json');opened=read('reference_open.json');assert frozen['reference_reads']==0 and frozen['frozen_utc']<opened['first_reference_quality_read_utc'] and sha(out/'interval_freeze.json')==opened['freeze_sha256']
    for p,h in frozen['input_hashes'].items():assert sha(p)==h,p
    prior=json.loads(PRIOR.read_bytes())['rows'];choice=json.loads(CHOICE.read_bytes());raw,items,recs=load_inputs(True)
    for i,(p,c,rec) in enumerate(zip(prior,choice['rows'],recs)):
        low,trace=state_input(raw,rec,True);np.testing.assert_allclose(p['totals'],trace['values'][:28],rtol=0,atol=1e-7);values=p['totals'];den=max(values[0]-min(values),1e-12);r=[min(1.,max(0.,(values[0]-v)/den)) for v in values];fs=min(k for k in range(c['k_rho']+1) if p['probabilities'][k]>=.5);assert fs==c['k_FS'];rows=[x for x in frozen['rows'] if x['index']==i];assert [x['k'] for x in rows]==list(range(fs,c['k_rho']+1))
        for row in rows:
            k=row['k'];assert abs(row['r_k']-r[k])<1e-12 and abs(row['q_k']-(r[k]-r[fs])/max(.9857470621423519-r[fs],1e-12))<1e-12
            assert row['k_FS']==fs and row['k_rho']==c['k_rho'] and row['selected_flag']==(k==c['selected_step'])
            assert row['state_hash']==thash(trace['states'][k]) and row['render_hash']==p['hashes'][k] and row['trace_sha256']==rec['methods']['T063']['trace_sha256'] and row['low_sha256']==rec['low_sha256'] and row['model_sha256']==choice['model_sha256'] and row['rule_sha256']==choice['rule_sha256']
    for p,h in json.loads((HERE/'evaluation_binding.json').read_bytes()).items():assert sha(p)==h,p
    recomputed=transfer_labels(prior,True);saved=read('interval_labels.json');error=0.
    for original,row in zip(frozen['rows'],saved):
        assert {k:row[k] for k in original}==original;q=recomputed[row['index']*28+row['k']];error=max(error,abs(row['margin']-q['quality_margin']));assert row['safe']==(q['quality_margin']>=-5.614)
    assert error<1e-10
    boundaries=[independent_boundary([r for r in saved if r['index']==i]) for i in range(100)];assert boundaries==read('boundaries.json');assert read('tail_rows.json')==[r for r in boundaries if r['index'] in [16,86]]
    bins={k:0 for k in ['[0,.50)','[.50,.75)','[.75,.875)','[.875,1.0001]','outside_fixed_bins']};outliers=[]
    for row in saved:
        if row['safe']:continue
        q=row['q_k'];matched=False
        for lo,hi,name in [(0,.5,'[0,.50)'),(.5,.75,'[.50,.75)'),(.75,.875,'[.75,.875)'),(.875,1.0001,'[.875,1.0001]')]:
            if q>=lo and (q<hi or (name=='[.875,1.0001]' and q==hi)):bins[name]+=1;matched=True;break
        if not matched:bins['outside_fixed_bins']+=1;outliers.append(dict(index=row['index'],k=row['k'],q_k=q))
    expected=dict(images=100,interval_states=len(saved),intervals_with_unsafe=sum(b['any_interval_unsafe'] for b in boundaries),intervals_with_recovery=sum(b['safety_recovers'] for b in boundaries),selected_unsafe=sum(not b['selected_safe'] for b in boundaries),unsafe_interval_states=sum(not row['safe'] for row in saved),fixed_q_bins=bins,outside_fixed_bin_rows=outliers)
    result=read('result.json')
    for k,v in expected.items():assert result[k]==v
    assert result['classification']=='INTERVAL_BOUNDARY_DIAGNOSIS_COMPLETE';write(out/'verification.json',dict(status='PASS',classification=result['classification'],interval_states_verified=len(saved),all_transfer_states_rerendered=2800,metric_max_error=error,optimizer_runs=0,model_fits=0,verified_utc=utc()));print(json.dumps(read('verification.json')),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
