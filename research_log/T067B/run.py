import argparse,json,os,time
from pathlib import Path
from research_log.T063A.common import sha,thash,write,utc
from research_log.T066A.run import load_inputs,state_input,development_rows
from research_log.T066A.core import predict
from research_log.T063B.core import summarize
from research_log.T067B.core import choices,select,GRID,RHO
HERE=Path('research_log/T067B');DEV_FEATURES=Path('research_log/T066A/evidence/development_features.json');DEV_TABLE=Path('research_log/T066B/evidence/development_target_free.json');MODEL=Path('research_log/T066A/evidence/model.json')
def main(out):
    import numpy as np
    binding=json.loads((HERE/'binding.json').read_bytes())
    for p,h in binding.items():assert sha(p)==h,p
    out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();write(out/'config.json',dict(task='T067-B',source_commit=os.environ['TTIE_SOURCE_COMMIT'],source_binding=binding,started_utc=utc(),probability_policy='frozen T066-A all-development model, no refit; matching previously frozen development target-free table',optimizer_runs=0,model_fits=0))
    dev=json.loads(DEV_FEATURES.read_bytes())['rows'];pt=json.loads(DEV_TABLE.read_bytes());model=json.loads(MODEL.read_bytes());raw,items,recs=load_inputs(False);rows=[]
    for i,(d,rec) in enumerate(zip(dev,recs)):
        low,trace=state_input(raw,rec,False);probs=predict(d['features'],model).tolist();prior=pt[i*28:(i+1)*28];np.testing.assert_array_equal(probs,[r['p_safe'] for r in prior]);assert d['hashes']==rec['rendered_hashes'][:28];np.testing.assert_allclose(d['totals'],trace['values'][:28],rtol=0,atol=1e-7)
        assert all(r['base_step']==d['base_step'] and r['index']==i and r['features']==d['features'][r['step']] for r in prior)
        for row in choices(d['totals'],probs,d['base_step']):
            k=row['selected_step'];row.update(index=i,low=rec['low'],low_sha256=rec['low_sha256'],trace_sha256=rec['files']['trace.pt'],images_sha256=rec['files']['images.pt'],state_hash=thash(trace['states'][k]),output_hash=rec['rendered_hashes'][k]);rows.append(row)
    write(out/'candidate_freeze.json',dict(rows=rows,source_commit=os.environ['TTIE_SOURCE_COMMIT'],input_hashes={str(p):sha(p) for p in [DEV_FEATURES,DEV_TABLE,MODEL]},reference_reads=0,frozen_utc=utc()))
    write(out/'reference_open.json',dict(first_development_quality_read_utc=utc(),freeze_sha256=sha(out/'candidate_freeze.json')))
    for p,h in json.loads((HERE/'evaluation_binding.json').read_bytes()).items():assert sha(p)==h,p
    quality=development_rows();table=[]
    for lam in GRID:
        selected=[r['selected_step'] for r in rows if r['lambda_value']==lam]
        for q,d in zip(quality,dev):assert q['state_hashes'][:28]==d['hashes']
        m=summarize(quality,selected);m.pop('eligible');table.append(dict(lambda_value=lam,selected_steps=selected,**m))
    best,verdict,order=select(table);write(out/'candidate_metrics.json',table)
    result=dict(classification=verdict,selected_lambda=best['lambda_value'] if best else None,selected_metrics=best,passing_candidates_ranked=order,selection_key='maximize worst_delta_t026; exact tie maximize mean_delta_psnr; exact tie minimize lambda',optimizer_runs=0,model_fits=0,completed_utc=utc(),seconds=time.perf_counter()-start)
    write(out/'rule_manifest.json',dict(lambda_value=result['selected_lambda'],rho=RHO,probability_threshold=.5,grid=GRID,probability_model_sha256=sha(MODEL),candidate_freeze_sha256=sha(out/'candidate_freeze.json'),candidate_metrics_sha256=sha(out/'candidate_metrics.json'),classification=verdict,transfer_authorized=False,frozen_utc=utc()))
    write(out/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
