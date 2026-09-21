"""One native smoke image; stop at the first frozen-method failure."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
import numpy as np,torch,cv2
from PIL import Image
from research_log.T063A.common import sha,thash,utc,write
HERE=Path('research_log/T072B')
MANIFEST=Path('research_log/T070A/evidence/FINAL_OURS_MANIFEST.json')
MANIFEST_SHA='e7f129d931d27531e5f6a14cd72e8c94734c474c40c3c403959cc8f5764e5ab9'

def main(method,low,out):
    binding=json.loads((HERE/'binding.json').read_bytes())
    for p,h in binding.items():assert sha(p)==h,p
    specs=json.loads(Path('research_log/T071B/baseline_bindings.json').read_bytes())
    for spec in specs.values():
        for p,h in spec['files'].items():assert sha(p)==h,p
    assert sha(MANIFEST)==MANIFEST_SHA
    mf=json.loads(MANIFEST.read_bytes())
    for p,h in mf['source_binding'].items():assert sha(p)==h,p
    for a in mf['assets'].values():assert sha(a['path'])==a['sha256'],a['path']
    receipt=json.loads((HERE/'input_receipt.json').read_bytes());assert sha(low)==receipt['sha256']
    assert low.name==json.loads((HERE/'smoke_selection.json').read_bytes())['name']
    out.mkdir(parents=True,exist_ok=False)
    record=dict(method=method,source_commit=os.environ['TTIE_SOURCE_COMMIT'],source_binding=binding,started_utc=utc(),low=str(low),input_sha256=sha(low),geometry=receipt['geometry'],reference_reads=0,model_fits=0,attempts=0,gpu=torch.cuda.get_device_name(),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],cuda_memory_free_before=torch.cuda.mem_get_info()[0],cuda_memory_total=torch.cuda.mem_get_info()[1])
    write(out/'receipt.json',record);begin=time.perf_counter();opened=[]
    try:
        if method=='ours':
            from research_log.T070A.infer import FinalOurs,native_rgb
            from research_log.T066A.run import ReadScope
            model=FinalOurs(MANIFEST)
            torch.optim.Adam([torch.nn.Parameter(torch.zeros(1,device='cuda'))],lr=.03)
            scope=ReadScope([low],out)
            with scope:
                x=native_rgb(low);assert tuple(x.shape[-2:])==(2160,3840)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();record['attempts']=1
                result=model(x);torch.cuda.synchronize();elapsed=time.perf_counter()-start
                image=result['image'].detach().cpu().clone();torch.save(image,out/'output.pt')
                write(out/'decision.json',dict(**result['decision'],output_hash=thash(image),state_hash=thash(result['state']),features=result['features'].tolist(),probabilities=result['probabilities'].tolist(),values=result['trace']['values'],reference_reads=0,optimizer_updates=27))
                record.update(output='output.pt',output_file_sha256=sha(out/'output.pt'),output_hash=thash(image),output_shape=list(image.shape),finite=bool(torch.isfinite(image).all()),seconds=elapsed,decision_sha256=sha(out/'decision.json'),optimizer_updates=27)
            opened=scope.reads
        else:
            if method=='retinexformer':from ttie import retinex_exporter as exporter
            else:from ttie import snr_exporter as exporter
            spec=specs[method];original=cv2.imread;pil_open=Image.open;argv=sys.argv
            def only_low(path,*args,**kwargs):
                assert Path(path).resolve()==low;opened.append(str(low));return original(path,*args,**kwargs)
            def deny(*args,**kwargs):raise AssertionError('PIL decoding forbidden in baseline exporter')
            cv2.imread=only_low;Image.open=deny
            try:
                sys.argv=['exporter','--low',str(low),'--checkpoint',spec['checkpoint'],'--config',spec['config'],'--out',str(out/'outputs')]
                if method=='snr_aware':sys.argv+=['--source',spec['source']]
                record['attempts']=1;exporter.main()
            finally:cv2.imread=original;Image.open=pil_open;sys.argv=argv
            er=json.loads((out/'outputs/receipt.json').read_bytes());assert len(er['rows'])==1 and er['mode']==spec['mode']
            if method=='snr_aware':assert er['parameter_hash_before']==er['parameter_hash_after']==spec['accepted_parameter_hash']
            else:assert er['GT_mean'] is False and er['self_ensemble'] is False
            file=out/'outputs'/(low.stem+'.npy');y=np.load(file);assert y.shape==(2160,3840,3) and y.dtype==np.float32
            record.update(output=str(file.relative_to(out)),output_file_sha256=sha(file),output_hash=er['rows'][0]['output_sha256'],output_shape=list(y.shape),finite=bool(np.isfinite(y).all()),seconds=er['rows'][0]['runtime_s'],optimizer_updates=0)
        assert record['finite'] and opened==[str(low)]
        record['status']='PASS'
    except Exception as exc:
        record.update(status='BLOCKED',error_type=type(exc).__name__,error=str(exc))
        (out/'traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        if method=='ours' and 'scope' in locals():opened=scope.reads
        record.update(opened=opened,completed_utc=utc(),elapsed_including_load_seconds=time.perf_counter()-begin,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        for p,h in binding.items():assert sha(p)==h,p
        for spec in specs.values():
            for p,h in spec['files'].items():assert sha(p)==h,p
        for a in mf['assets'].values():assert sha(a['path'])==a['sha256'],a['path']
        record['post_binding_unchanged']=True;write(out/'receipt.json',record);print(json.dumps(record),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['ours','retinexformer','snr_aware'],required=True);p.add_argument('--low',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.method,a.low.resolve(),a.out.resolve())
