import argparse,json,os,time
from pathlib import Path
from research_log.T063A.common import sha,thash,write,utc
from research_log.T066A.run import load_inputs,state_input
from research_log.T067D.core import interval,boundaries
HERE=Path('research_log/T067D');PRIOR=Path('research_log/T066A/evidence/transfer_freeze.json');CHOICE=Path('research_log/T067C/evidence/choice_freeze.json');LABELS=Path('research_log/T066B/evidence/transfer_labels.json')
def main(out):
    import numpy as np
    binding=json.loads((HERE/'binding.json').read_bytes())
    for p,h in binding.items():assert sha(p)==h,p
    out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();write(out/'config.json',dict(task='T067-D',source_commit=os.environ['TTIE_SOURCE_COMMIT'],source_binding=binding,started_utc=utc(),optimizer_runs=0,model_fits=0))
    prior=json.loads(PRIOR.read_bytes())['rows'];choice=json.loads(CHOICE.read_bytes());raw,items,recs=load_inputs(True);rows=[]
    for i,(rec,c,p) in enumerate(zip(recs,choice['rows'],prior)):
        low,trace=state_input(raw,rec,True);np.testing.assert_allclose(p['totals'],trace['values'][:28],rtol=0,atol=1e-7);base=rec['methods']['T063']['selected_step'];assert base==c['k_rho']==p['base_step'];fs=next(k for k in range(base+1) if p['probabilities'][k]>=.5);assert fs==c['k_FS']
        for row in interval(p['totals'],fs,base,c['selected_step']):
            k=row['k'];row.update(index=i,low=rec['low'],low_sha256=rec['low_sha256'],state_hash=thash(trace['states'][k]),render_hash=p['hashes'][k],trace_sha256=rec['methods']['T063']['trace_sha256'],model_sha256=choice['model_sha256'],rule_sha256=choice['rule_sha256']);rows.append(row)
    write(out/'interval_freeze.json',dict(rows=rows,input_hashes={str(p):sha(p) for p in [PRIOR,CHOICE]},reference_reads=0,frozen_utc=utc(),source_commit=os.environ['TTIE_SOURCE_COMMIT']))
    write(out/'reference_open.json',dict(first_reference_quality_read_utc=utc(),freeze_sha256=sha(out/'interval_freeze.json')))
    for p,h in json.loads((HERE/'evaluation_binding.json').read_bytes()).items():assert sha(p)==h,p
    labels=json.loads(LABELS.read_bytes());joined=[]
    for row in rows:
        q=labels[row['index']*28+row['k']];assert q['index']==row['index'] and q['step']==row['k'];joined.append(dict(row,margin=q['quality_margin'],safe=q['quality_margin']>=-5.614))
    per_image,aggregate=boundaries(joined);write(out/'interval_labels.json',joined);write(out/'boundaries.json',per_image);write(out/'tail_rows.json',[r for r in per_image if r['index'] in [16,86]])
    result=dict(classification='INTERVAL_BOUNDARY_DIAGNOSIS_COMPLETE',**aggregate,optimizer_runs=0,model_fits=0,completed_utc=utc(),seconds=time.perf_counter()-start);write(out/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
