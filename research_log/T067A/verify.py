import argparse,json,math,statistics
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import torch
from research_log.T063A.common import sha,thash,utc,write
from research_log.T066A.run import load_inputs,state_input,TARGET_LOW,score_transfer,CommonRegion2
from research_log.T067A.run import EVENTS,TABLE,HERE

def independent_first(probabilities,base):
    candidates=[k for k,p in enumerate(probabilities[:base+1]) if p>=.5];assert candidates
    return min(candidates)

def main(out):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;read=lambda n:json.loads((out/n).read_bytes())
    config=read('config.json')
    for p,h in config['source_binding'].items():assert sha(p)==h,p
    freeze=read('choice_freeze.json');opened=read('reference_open.json');rows=freeze['rows'];events=json.loads(EVENTS.read_bytes());table=json.loads(TABLE.read_bytes())
    assert freeze['reference_reads']==0 and freeze['frozen_utc']<opened['first_reference_quality_read_utc'];assert sha(out/'choice_freeze.json')==opened['freeze_sha256'] and sha(EVENTS)==freeze['events_sha256'] and sha(TABLE)==freeze['table_sha256']
    raw,items,recs=load_inputs(True);assert len(rows)==len(events)==100
    with torch.no_grad():
        for i,(row,rec,e) in enumerate(zip(rows,recs,events)):
            seq=[r for r in table if r['index']==i];assert [r['step'] for r in seq]==list(range(28));base=rec['methods']['T063']['selected_step'];k=independent_first([r['p_safe'] for r in seq],base)
            assert k==row['selected_step']==e['first_safe'] and row['base_step']==base==e['base_step'] and row['index']==i and row['low']==rec['low']
            assert row['low_sha256']==rec['low_sha256'] and row['trace_sha256']==rec['methods']['T063']['trace_sha256'] and row['p_safe']==seq[k]['p_safe']
            low,trace=state_input(raw,rec,True);state=trace['states'][k];assert thash(state)==row['selected_state_hash'];x=low.cuda();model=CommonRegion2(trace['active']).to(x).eval().requires_grad_(False);model.raw.copy_(state.to(x));image=model(x).cpu();assert thash(image)==row['output_hash']==seq[k]['render_hash']
            path=out/f'{i:03d}.pt';assert sha(path)==row['output_file_sha256'];saved=torch.load(path,weights_only=True,map_location='cpu');assert torch.equal(image,saved)
    allowed={str((raw/f'{i:03d}'/'T063_trace.pt').resolve()) for i in range(100)}|{str((TARGET_LOW/r['low']).resolve()) for r in recs};assert set(freeze['data_reads'])==allowed
    for p,h in json.loads((HERE/'evaluation_binding.json').read_bytes()).items():assert sha(p)==h,p
    refs=json.loads(Path('research_log/T063D/reference_inputs.json').read_bytes())
    with ProcessPoolExecutor(max_workers=8) as pool:scored=list(pool.map(score_transfer,[(str(out),r,i,n,True) for r,i,n in zip(rows,items,refs)]))
    primary=read('per_image.json');error=0.
    for a,b in zip(scored,primary):
        assert b['reference_read_utc']>=opened['first_reference_quality_read_utc']
        for method in ['selected','T026','T036']:
            for key in ['psnr','ssim']:error=max(error,abs(a['metrics'][method][key]-b['metrics'][method][key]))
        for baseline,key in [('T026','delta_t026'),('T036','delta_t036')]:assert abs(a['metrics']['selected']['psnr']-a['metrics'][baseline]['psnr']-b[key])<1e-10
    assert error<1e-10
    d=[r['metrics']['selected']['psnr']-r['metrics']['T036']['psnr'] for r in scored];b=[r['metrics']['selected']['psnr']-r['metrics']['T026']['psnr'] for r in scored];s=[r['metrics']['selected']['ssim']-r['metrics']['T036']['ssim'] for r in scored]
    m=dict(mean_delta_psnr=math.fsum(d)/100,median_delta_psnr=statistics.median(d),regressions_t026=sum(x<0 for x in b),worst_delta_t026=min(b),mean_delta_ssim=math.fsum(s)/100,absolute_psnr=math.fsum(r['metrics']['selected']['psnr'] for r in scored)/100,absolute_ssim=math.fsum(r['metrics']['selected']['ssim'] for r in scored)/100)
    result=read('result.json')
    for k,v in m.items():assert abs(v-result[k])<1e-10
    gates=dict(mean_psnr=m['mean_delta_psnr']>=2,median_psnr=m['median_delta_psnr']>0,regressions=m['regressions_t026']<=29,worst=m['worst_delta_t026']>=-5.614,mean_ssim=m['mean_delta_ssim']>=-.001);assert gates==result['gates']
    verdict='FIRST_SAFE_TRANSFER_PASS' if all(gates.values()) else 'FIRST_SAFE_TRANSFER_NEGATIVE';assert verdict==result['classification']
    assert read('tail_rows.json')==[r for r in primary if r['index'] in [16,86]]
    assert result['selected_step_histogram']=={str(k):sum(r['selected_step']==k for r in rows) for k in range(28)} and result['choices_changed']==sum(r['selected_step']!=r['base_step'] for r in rows)
    write(out/'verification.json',dict(status='PASS',classification=verdict,selected_states_verified=100,metric_max_error=error,gates=gates,optimizer_runs=0,model_fits=0,verified_utc=utc()));print(json.dumps(read('verification.json')),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
