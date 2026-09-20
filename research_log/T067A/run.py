import argparse,json,os,time,statistics
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import torch
from research_log.T063A.common import sha,thash,utc,write
from research_log.T066A.run import load_inputs,state_input,ReadScope,TARGET_LOW,score_transfer,metric_rows,CommonRegion2
from research_log.T063B.core import summarize
from research_log.T067A.core import first_safe
HERE=Path('research_log/T067A')
EVENTS=Path('research_log/T066C/evidence/events.json')
TABLE=Path('research_log/T066B/evidence/transfer_target_free.json')
def main(out):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    binding=json.loads((HERE/'binding.json').read_bytes())
    for p,h in binding.items():assert sha(p)==h,p
    out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();write(out/'config.json',dict(task='T067-A',source_commit=os.environ['TTIE_SOURCE_COMMIT'],source_binding=binding,started_utc=utc(),gpu=torch.cuda.get_device_name(),physical_gpu=1,optimizer_runs=0,model_fits=0))
    events=json.loads(EVENTS.read_bytes());table=json.loads(TABLE.read_bytes());raw,items,recs=load_inputs(True);rows=[]
    scope=ReadScope([raw/f'{i:03d}'/'T063_trace.pt' for i in range(100)]+[TARGET_LOW/r['low'] for r in recs],out)
    with scope,torch.no_grad():
        for i,(e,item,rec) in enumerate(zip(events,items,recs)):
            seq=table[i*28:(i+1)*28];base=rec['methods']['T063']['selected_step'];assert e['index']==i and e['base_step']==base and all(r['base_step']==base for r in seq)
            probs=[r['p_safe'] for r in seq];k=first_safe(probs,base);assert k==e['first_safe'] and probs[:base+1]==e['prefix_probabilities']
            low,trace=state_input(raw,rec,True);state=trace['states'][k];x=low.cuda();model=CommonRegion2(trace['active']).to(x).eval().requires_grad_(False);model.raw.copy_(state.to(x));y=model(x).cpu();h=thash(y);assert h==seq[k]['render_hash']
            path=out/f'{i:03d}.pt'
            with path.open('xb') as f:torch.save(y,f)
            rows.append(dict(index=i,low=item['low'],low_sha256=rec['low_sha256'],base_step=base,selected_step=k,p_safe=probs[k],selected_state_hash=thash(state),trace_sha256=rec['methods']['T063']['trace_sha256'],output_hash=h,output_file_sha256=sha(path)))
    write(out/'choice_freeze.json',dict(rows=rows,data_reads=scope.reads,events_sha256=sha(EVENTS),table_sha256=sha(TABLE),source_commit=os.environ['TTIE_SOURCE_COMMIT'],frozen_utc=utc(),reference_reads=0))
    first=utc();write(out/'reference_open.json',dict(first_reference_quality_read_utc=first,freeze_sha256=sha(out/'choice_freeze.json')))
    for p,h in json.loads((HERE/'evaluation_binding.json').read_bytes()).items():assert sha(p)==h,p
    refs=json.loads(Path('research_log/T063D/reference_inputs.json').read_bytes())
    with ProcessPoolExecutor(max_workers=8) as pool:scored=list(pool.map(score_transfer,[(str(out),r,i,n,False) for r,i,n in zip(rows,items,refs)]))
    metrics=summarize(metric_rows(scored),[0]*100);metrics.pop('eligible');passed=all(metrics['gates'].values())
    result=dict(classification='FIRST_SAFE_TRANSFER_PASS' if passed else 'FIRST_SAFE_TRANSFER_NEGATIVE',label='exposed-cohort fixed-rule audit only; not fresh qualification',**metrics,absolute_psnr=statistics.fmean(r['metrics']['selected']['psnr'] for r in scored),absolute_ssim=statistics.fmean(r['metrics']['selected']['ssim'] for r in scored),selected_step_histogram={str(k):sum(r['selected_step']==k for r in rows) for k in range(28)},choices_changed=sum(r['selected_step']!=r['base_step'] for r in rows),completed_utc=utc(),seconds=time.perf_counter()-start,optimizer_runs=0,model_fits=0)
    for row in scored:row['delta_t026']=row['metrics']['selected']['psnr']-row['metrics']['T026']['psnr'];row['delta_t036']=row['metrics']['selected']['psnr']-row['metrics']['T036']['psnr']
    write(out/'per_image.json',scored);write(out/'tail_rows.json',[r for r in scored if r['index'] in [16,86]]);write(out/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
