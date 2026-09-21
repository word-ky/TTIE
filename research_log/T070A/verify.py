import argparse,json
from pathlib import Path
import numpy as np
import torch
from research_log.T063A.common import sha,thash,write,utc
from research_log.T066A.run import state_input,ReadScope,DEV_LOW,TARGET_LOW,CommonRegion2
from research_log.T066A.core import features
from research_log.T066A.verify import independent_predict
from research_log.T067B.verify import independent_choices
from research_log.T070A.audit import inputs,DEV_ANCHOR,TRANSFER_ANCHOR
from research_log.T070A.manifest import HERE,build

def main(out,manifest):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    get=lambda n:json.loads((out/n).read_bytes());cfg=get('config.json');frozen=get('replay_freeze.json');result=get('result.json')
    for p,h in cfg['source_binding'].items():assert sha(p)==h,p
    mf=json.loads(manifest.read_bytes());assert mf==build();mh=sha(manifest)
    assert mh==cfg['manifest_sha256']==frozen['manifest_sha256']==result['manifest_sha256']
    assert sha(out/'replay_freeze.json')==result['replay_sha256'] and frozen['reference_reads']==0 and len(frozen['rows'])==200
    model=json.loads(Path(mf['assets']['probability_model']['path']).read_bytes());renders=0;checked=[];reads=[]
    for transfer in [False,True]:
        raw,recs,anchors,tables=inputs(transfer);ah=sha(TRANSFER_ANCHOR if transfer else DEV_ANCHOR)
        cohort='transfer' if transfer else 'development';reports=[r for r in frozen['rows'] if r['cohort']==cohort];assert len(reports)==100
        allowed=[(TARGET_LOW if transfer else DEV_LOW)/r['low'] for r in recs]+[raw/f"{r['index']:03d}"/('T063_trace.pt' if transfer else 'trace.pt') for r in recs]
        scope=ReadScope(allowed,out)
        with scope:
            for rec,anchor,d,row in zip(recs,anchors,tables,reports):
                low,trace=state_input(raw,rec,transfer);x=low.cuda();renderer=CommonRegion2(trace['active']).to(x).eval().requires_grad_(False);images=[]
                with torch.no_grad():
                    for state in trace['states'][:28]:
                        renderer.raw.copy_(state.to(x));images.append(renderer(x).cpu())
                hashes=[thash(y) for y in images];renders+=len(images);assert hashes==row['image_hashes']==d['hashes']
                ts={}
                for key,count in [('states',28),('components',28),('gradients',27),('pre_box',27)]:ts[key]=thash(trace[key][:count])
                for key in ['active','winner','lower','upper']:ts[key]=thash(trace[key])
                ts['values']=trace['values'][:28];assert ts==row['trajectory']
                ft=features(x,images,trace['states'][:28],trace['values'][:28]);np.testing.assert_allclose(ft,d['features'],rtol=1e-10,atol=1e-10)
                prob=independent_predict(ft,model);k=dict(independent_choices(trace['values'][:28],prob,d['base_step']))[.875]
                fs=min(j for j in range(d['base_step']+1) if prob[j]>=.5)
                assert row['index']==rec['index']==anchor['index'] and row['low']==rec['low']==anchor['low'] and row['low_sha256']==rec['low_sha256']
                assert row['trace_sha256']==anchor['trace_sha256'] and row['anchor_sha256']==ah and row['reference_reads']==0
                assert row['selected_step']==k==anchor['selected_step'] and row['k_FS']==fs==anchor['k_FS'] and row['k_rho']==d['base_step']==anchor['k_rho']
                assert row['selected_state_hash']==thash(trace['states'][k])==anchor['selected_state_hash' if transfer else 'state_hash'] and row['output_hash']==hashes[k]==anchor['output_hash']
                assert row['feature_shape']==[28,19] and row['optimizer_runs']==1 and row['optimizer_updates']==27 and row['model_fits']==0
                checked.append(row)
        reads.extend(scope.reads)
    assert len(checked)==200 and len(frozen['repeats'])==10
    assert [(r['cohort'],r['index']) for r in frozen['repeats']]==[(c,i) for c in ['development','transfer'] for i in range(5)]
    for rep in frozen['repeats']:
        row=next(r for r in checked if r['cohort']==rep['cohort'] and r['index']==rep['index'])
        assert rep['exact'] and all(row[k]==v for k,v in rep['receipt'].items()) and rep['optimizer_runs']==1 and rep['optimizer_updates']==27
    assert result['classification']=='FINAL_OURS_CANDIDATE_FROZEN' and result['optimizer_runs']==210 and result['optimizer_updates']==5670 and result['model_fits']==0 and result['reference_reads']==0
    for k in ['selection_mismatches','state_mismatches','output_mismatches','trajectory_mismatches','repeat_mismatches']:assert result[k]==0
    assert sha(manifest)==mh
    receipt=dict(status='PASS',classification=result['classification'],images=200,independent_gpu_renders=renders,independent_selected_mismatches=0,repeat_rows_verified=10,manifest_sha256=mh,reference_reads=0,data_reads=reads,optimizer_runs=0,model_fits=0,verified_utc=utc())
    write(out/'verification.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='data_reads'}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args();main(a.out.resolve(),a.manifest.resolve())
