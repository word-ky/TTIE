import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from research_log.T063A.common import sha,thash,write,utc
from research_log.T066A.run import load_inputs,state_input,ReadScope,DEV_LOW,TARGET_LOW
from research_log.T070A.infer import FinalOurs
from research_log.T070A.manifest import HERE,build
DEV_ANCHOR=Path('research_log/T067B/evidence/candidate_freeze.json')
TRANSFER_ANCHOR=Path('research_log/T067C/evidence/choice_freeze.json')
DEV_FEATURES=Path('research_log/T066A/evidence/development_features.json')
TRANSFER_FEATURES=Path('research_log/T066A/evidence/transfer_freeze.json')

def inputs(transfer):
    raw,items,recs=load_inputs(transfer)
    anchor=json.loads((TRANSFER_ANCHOR if transfer else DEV_ANCHOR).read_bytes());assert anchor['reference_reads']==0
    anchors=anchor['rows'] if transfer else [r for r in anchor['rows'] if r['lambda_value']==.875]
    features=json.loads((TRANSFER_FEATURES if transfer else DEV_FEATURES).read_bytes())['rows']
    assert len(recs)==len(anchors)==len(features)==100
    return raw,recs,anchors,features

def trace_hashes(trace):
    result={k:thash(trace[k][:28 if k in ['states','components'] else 27]) for k in ['states','components','gradients','pre_box']}
    result.update({k:thash(trace[k]) for k in ['active','winner','lower','upper']})
    result['values']=trace['values'][:28]
    return result

def receipt(result):
    t=result['trace'];k=result['decision']['selected_step']
    return dict(selected_step=k,k_FS=result['decision']['k_FS'],k_rho=result['decision']['k_rho'],selected_state_hash=thash(result['state']),output_hash=thash(result['image']),trajectory=trace_hashes(t),image_hashes=[thash(im) for im in t['images']],feature_shape=list(result['features'].shape))

def main(out,manifest):
    binding=json.loads((HERE/'binding.json').read_bytes())
    for path,h in binding.items():assert sha(path)==h,path
    manifest_hash=sha(manifest);assert json.loads(manifest.read_bytes())==build()
    out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    model=FinalOurs(manifest)
    # Same lazy optimizer import initialization as accepted T062/T063 runners; no step.
    torch.optim.Adam([torch.nn.Parameter(torch.zeros(1,device='cuda'))],lr=.03)
    write(out/'config.json',dict(task='T070-A',source_commit=os.environ['TTIE_SOURCE_COMMIT'],source_binding=binding,manifest_sha256=manifest_hash,started_utc=utc(),repeat_subset='first5development+first5transfer',reference_reads=0,model_fits=0))
    rows=[];repeats=[];reads=[]
    for transfer in [False,True]:
        raw,recs,anchors,features=inputs(transfer)
        allowed=[(TARGET_LOW if transfer else DEV_LOW)/r['low'] for r in recs]+[raw/f"{r['index']:03d}"/('T063_trace.pt' if transfer else 'trace.pt') for r in recs]
        scope=ReadScope(allowed,out)
        with scope:
            for rec,anchor,feature in zip(recs,anchors,features):
                low,trace=state_input(raw,rec,transfer)
                result=model(low);row=receipt(result)
                assert row['trajectory']==trace_hashes(trace),(transfer,rec['index'],'trajectory')
                assert row['image_hashes']==feature['hashes'],(transfer,rec['index'],'renders')
                assert anchor['index']==rec['index'] and anchor['low']==rec['low'] and anchor['trace_sha256']==(rec['methods']['T063']['trace_sha256'] if transfer else rec['files']['trace.pt'])
                assert row['selected_step']==anchor['selected_step'] and row['k_FS']==anchor['k_FS'] and row['k_rho']==anchor['k_rho']
                assert row['selected_state_hash']==anchor['selected_state_hash' if transfer else 'state_hash'] and row['output_hash']==anchor['output_hash']
                row.update(cohort='transfer' if transfer else 'development',index=rec['index'],low=rec['low'],low_sha256=rec['low_sha256'],trace_sha256=anchor['trace_sha256'],reference_reads=0,optimizer_runs=1,optimizer_updates=27,model_fits=0)
                rows.append(row);del result
                if rec['index']<5:
                    repeated=model(low);again=receipt(repeated)
                    assert all(again[k]==row[k] for k in again),(transfer,rec['index'],'repeat')
                    repeats.append(dict(cohort=row['cohort'],index=rec['index'],receipt=again,exact=True,optimizer_runs=1,optimizer_updates=27));del repeated
                print(json.dumps(dict(cohort=row['cohort'],completed=rec['index']+1)),flush=True)
        reads.extend(scope.reads)
        anchor_hash=sha(TRANSFER_ANCHOR if transfer else DEV_ANCHOR)
        for row in rows:
            if row['cohort']==('transfer' if transfer else 'development'):row['anchor_sha256']=anchor_hash
    assert len(rows)==200 and len(repeats)==10 and sha(manifest)==manifest_hash
    write(out/'replay_freeze.json',dict(rows=rows,repeats=repeats,data_reads=reads,reference_reads=0,manifest_sha256=manifest_hash,frozen_utc=utc()))
    result=dict(classification='FINAL_OURS_CANDIDATE_FROZEN',images=200,repeat_images=10,selection_mismatches=0,state_mismatches=0,output_mismatches=0,trajectory_mismatches=0,repeat_mismatches=0,manifest_sha256=manifest_hash,replay_sha256=sha(out/'replay_freeze.json'),reference_reads=0,optimizer_runs=210,optimizer_updates=5670,model_fits=0,completed_utc=utc(),seconds=time.perf_counter()-start)
    write(out/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args();main(a.out.resolve(),a.manifest.resolve())
