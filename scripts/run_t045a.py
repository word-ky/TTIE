"""Audit wrapper around the unchanged accepted T027-B exporter main."""
import argparse,json,hashlib,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch,cv2
from PIL import Image
from ttie import snr_exporter

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()

def main():
    p=argparse.ArgumentParser()
    for k in ['low-root','split','binding','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();b=json.loads(a.binding.read_bytes());split=json.loads(a.split.read_bytes())['selected'];assert len(split)==100
    assert sha(a.split)==b['split_sha256']=='b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b'
    for path,h in b['files'].items():assert sha(path)==h,path
    assert Path(b['checkpoint']).stat().st_size==156523164
    paths=[a.low_root/r['low'] for r in split]
    for p,r in zip(paths,split):assert sha(p)==r['low_sha256']
    allowed={str(p.resolve()) for p in paths};opened=[];original=cv2.imread
    def low_only(path,*args,**kwargs):
        name=str(Path(path).resolve());assert name in allowed,'non-cohort image access'
        opened.append(dict(path=name,utc=utc()));return original(path,*args,**kwargs)
    def deny(*args,**kwargs):raise AssertionError('PIL decoding disabled in low-only exporter')
    cv2.imread=low_only;Image.open=deny
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'config.json').write_text(json.dumps(dict(task='T045-A',started_utc=utc(),binding=b,binding_sha256=sha(a.binding),
        upstream_commit='1113144c82adc8bcc4a9ec27749ed75f196a4e4d',mode='ttie_native_pad16',torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()),indent=2)+'\n')
    sys.argv=['snr_exporter','--low',*[str(p) for p in paths],'--checkpoint',b['checkpoint'],'--config',b['config'],'--source',b['source'],'--out',str(a.out/'outputs')]
    snr_exporter.main()
    receipt=json.loads((a.out/'outputs/receipt.json').read_bytes());assert receipt['mode']=='ttie_native_pad16' and receipt['parameter_hash_before']==receipt['parameter_hash_after']==b['accepted_parameter_hash']
    assert [r['path'] for r in opened]==[str(p.resolve()) for p in paths]
    assert len(receipt['rows'])==100;rows=[]
    for i,(r,s,p) in enumerate(zip(receipt['rows'],split,paths)):
        output=a.out/'outputs'/(p.stem+'.npy');y=np.load(output)
        assert r['low']==str(p) and r['low_sha256']==s['low_sha256']
        assert y.dtype==np.float32 and y.shape==(400,600,3) and np.isfinite(y).all() and 0<=y.min()<=y.max()<=1
        assert hashlib.sha256(y.tobytes()).hexdigest()==r['output_sha256']
        rows.append(dict(index=i,low=s['low'],input_sha256=s['low_sha256'],output=str(output.relative_to(a.out)),file_sha256=sha(output),**{k:r[k] for k in ['output_sha256','shape','runtime_s','peak_memory_bytes']}))
    for path,h in b['files'].items():assert sha(path)==h,path
    (a.out/'freeze.json').write_text(json.dumps(dict(completed_utc=utc(),rows=rows,split_sha256=sha(a.split),binding_sha256=sha(a.binding),config_sha256=sha(a.out/'config.json'),
        exporter_receipt_sha256=sha(a.out/'outputs/receipt.json'),opened=opened,normal_decodes=0,postrun_source_checkpoint_unchanged=True),indent=2)+'\n')
    print('FROZEN100 native float outputs; normal decodes0',flush=True)

if __name__=='__main__':main()
