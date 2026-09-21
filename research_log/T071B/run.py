"""Invoke the accepted exporter unchanged, replacing only the cohort list."""
import argparse,sys,os,platform
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from research_log.T071B.common import HERE,PAIRS,METHODS,read,write,sha,utc

def main(method,out):
    spec=read(HERE/'baseline_bindings.json')[method]
    binding=read(HERE/'binding.json')
    for p,h in binding.items():assert sha(p)==h,p
    for p,h in spec['files'].items():assert sha(p)==h,p
    cohort=read(PAIRS);paths=[Path(r['staged_low']) for r in cohort['rows']]
    assert len(paths)==100
    for p,r in zip(paths,cohort['rows']):assert sha(p)==r['low_sha256']
    out.mkdir(parents=True,exist_ok=False)
    write(out/'config.json',dict(task='T071-B',method=method,started_utc=utc(),source_commit=os.environ['TTIE_SOURCE_COMMIT'],source_binding=binding,baseline=spec,pairs_manifest_sha256=sha(PAIRS),torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),numpy=np.__version__,opencv=cv2.__version__,python=platform.python_version(),gpu=torch.cuda.get_device_name(),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES']))
    allowed={str(p.resolve()) for p in paths};opened=[];original=cv2.imread;pil_open=Image.open;argv=sys.argv
    def low_only(path,*args,**kwargs):
        name=str(Path(path).resolve());assert name in allowed,'non-cohort image access'
        opened.append(dict(path=name,utc=utc()));return original(path,*args,**kwargs)
    def deny(*args,**kwargs):raise AssertionError('PIL decode forbidden during exporter inference')
    cv2.imread=low_only;Image.open=deny
    try:
        if method=='retinexformer':from ttie import retinex_exporter as exporter
        else:from ttie import snr_exporter as exporter
        sys.argv=['exporter','--low',*[str(p) for p in paths],'--checkpoint',spec['checkpoint'],'--config',spec['config'],'--out',str(out/'outputs')]
        if method=='snr_aware':sys.argv+=['--source',spec['source']]
        exporter.main()
    finally:cv2.imread=original;Image.open=pil_open;sys.argv=argv
    receipt=read(out/'outputs/receipt.json');assert len(receipt['rows'])==100
    assert receipt['mode']==spec['mode']
    if method=='snr_aware':assert receipt['parameter_hash_before']==receipt['parameter_hash_after']==spec['accepted_parameter_hash']
    else:assert receipt['GT_mean'] is False and receipt['self_ensemble'] is False
    assert [r['path'] for r in opened]==[str(p.resolve()) for p in paths]
    rows=[]
    for pair,r,p in zip(cohort['rows'],receipt['rows'],paths):
        file=out/'outputs'/(p.stem+'.npy');y=np.load(file)
        assert r['low']==str(p) and sha(p)==r['low_sha256']==pair['low_sha256']
        assert y.dtype==np.float32 and y.shape==(400,600,3) and np.isfinite(y).all() and 0<=y.min()<=y.max()<=1
        import hashlib
        assert hashlib.sha256(y.tobytes()).hexdigest()==r['output_sha256']
        rows.append(dict(index=pair['index'],low=pair['low'],normal=pair['normal'],low_sha256=pair['low_sha256'],output=str(file.relative_to(out)),file_sha256=sha(file),**{k:r[k] for k in ['output_sha256','shape','runtime_s','peak_memory_bytes']}))
    for p,h in spec['files'].items():assert sha(p)==h,p
    for p,h in binding.items():assert sha(p)==h,p
    write(out/'freeze.json',dict(method=method,completed_utc=utc(),rows=rows,pairs_manifest_sha256=sha(PAIRS),config_sha256=sha(out/'config.json'),receipt_sha256=sha(out/'outputs/receipt.json'),opened=opened,reference_reads=0,optimizer_runs=0,model_fits=0,inference_seconds=sum(r['runtime_s'] for r in rows),source_checkpoint_unchanged=True))
    print(method,'FROZEN100',sha(out/'freeze.json'),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.method,a.out.resolve())
