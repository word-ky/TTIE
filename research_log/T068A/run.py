import argparse,json,os,time
from pathlib import Path
from collections import Counter
import numpy as np
from research_log.T063A.common import sha,thash,write,utc
from research_log.T066A.run import load_inputs,state_input,development_rows
from research_log.T066A.core import predict
from research_log.T063B.core import summarize
from research_log.T067B.core import choices,RHO
from research_log.T067B.run import DEV_FEATURES,DEV_TABLE,MODEL
from research_log.T068A.core import capped_step,select,CAPS,LAMBDA
HERE=Path('research_log/T068A');PRIOR=Path('research_log/T067B/evidence/candidate_freeze.json');PRIOR_METRICS=Path('research_log/T067B/evidence/candidate_metrics.json')

def main(out):
    binding=json.loads((HERE/'binding.json').read_bytes())
    for p,h in binding.items():assert sha(p)==h,p
    out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    write(out/'config.json',dict(task='T068-A',source_commit=os.environ['TTIE_SOURCE_COMMIT'],source_binding=binding,started_utc=utc(),optimizer_runs=0,model_fits=0))
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
        for cap in CAPS:
            k=capped_step(interior['k_FS'],interior['selected_step'],cap)
            row=dict(index=i,low=rec['low'],low_sha256=rec['low_sha256'],trace_sha256=rec['files']['trace.pt'],images_sha256=rec['files']['images.pt'],K=cap,k_FS=interior['k_FS'],k_lambda=interior['selected_step'],k_K=k,state_hash=thash(trace['states'][k]),output_hash=rec['rendered_hashes'][k],model_sha256=model_hash,rule_sha256=rule_hash,interpolation_rule_sha256=interpolation_hash)
            if cap==27:assert k==old['selected_step'] and row['state_hash']==old['state_hash'] and row['output_hash']==old['output_hash']
            rows.append(row)
    write(out/'candidate_freeze.json',dict(rows=rows,source_commit=os.environ['TTIE_SOURCE_COMMIT'],input_hashes={str(p):sha(p) for p in [DEV_FEATURES,DEV_TABLE,MODEL,PRIOR]},reference_reads=0,frozen_utc=utc()))
    write(out/'reference_open.json',dict(first_development_quality_read_utc=utc(),freeze_sha256=sha(out/'candidate_freeze.json')))
    for p,h in json.loads((HERE/'evaluation_binding.json').read_bytes()).items():assert sha(p)==h,p
    quality=development_rows();table=[]
    for q,d in zip(quality,dev):assert q['state_hashes'][:28]==d['hashes']
    for cap in CAPS:
        selected=[r['k_K'] for r in rows if r['K']==cap];m=summarize(quality,selected);m.pop('eligible')
        table.append(dict(K=cap,selected_steps=selected,**m))
    control=next(r for r in json.loads(PRIOR_METRICS.read_bytes()) if r['lambda_value']==LAMBDA)
    assert {k:v for k,v in control.items() if k!='lambda_value'}=={k:v for k,v in table[27].items() if k!='K'}
    best,verdict,order=select(table);write(out/'candidate_metrics.json',table)
    selected_rows=[r for r in rows if r['K']==best['K']];write(out/'selected_choices.json',selected_rows)
    result=dict(classification=verdict,selected_K=best['K'],selected_metrics=best,passing_candidates_ranked=order,changed_choices_vs_uncapped=sum(r['k_K']!=r['k_lambda'] for r in selected_rows),selected_step_histogram=dict(sorted(Counter(r['k_K'] for r in selected_rows).items())),control_exact=True,optimizer_runs=0,model_fits=0,completed_utc=utc(),seconds=time.perf_counter()-start)
    write(out/'rule_manifest.json',dict(K=best['K'],lambda_value=LAMBDA,rho=RHO,probability_threshold=.5,caps=CAPS,formula='max(k_FS,min(k_lambda,K))',ranking='maximize worst_T026, mean_T036, median_T036, K; passing gates only',probability_model_sha256=model_hash,rule_sha256=rule_hash,interpolation_rule_sha256=interpolation_hash,candidate_freeze_sha256=sha(out/'candidate_freeze.json'),candidate_metrics_sha256=sha(out/'candidate_metrics.json'),selected_choices_sha256=sha(out/'selected_choices.json'),classification=verdict,transfer_authorized=False,frozen_utc=utc()))
    write(out/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
