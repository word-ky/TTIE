"""Execute the unchanged T071-B exporter.main in a fresh single-GPU process."""
import hashlib,importlib,json,sys
from pathlib import Path
from launcher import load_contract,invocation,sha,write

def main():
    method,out,root=sys.argv[1:];out=Path(out);root=Path(root)
    spec,bindings=load_contract();b=bindings[method]
    sys.path.insert(0,str(root))
    import cv2,numpy as np,torch
    from PIL import Image
    exporter=importlib.import_module(spec['entrypoints'][method])
    expected=root/('ttie/retinex_exporter.py' if method=='retinexformer' else 'ttie/snr_exporter.py')
    if Path(exporter.__file__).resolve()!=expected.resolve():raise ValueError('unexpected exporter import')
    allowed=Path(spec['smoke']['path']).resolve();original=cv2.imread;opens=[]
    def only_low(path,*args,**kwargs):
        if Path(path).resolve()!=allowed:raise PermissionError('only canonical low decode permitted')
        opens.append(str(path));return original(path,*args,**kwargs)
    def deny(*args,**kwargs):raise PermissionError('PIL decode forbidden in exporter')
    cv2.imread=only_low;Image.open=deny
    sys.argv=['exporter',*invocation(method,spec,b,out)]
    exporter.main()
    if len(opens)!=1:raise ValueError('not exactly one low decode')
    exported=json.loads((out/'outputs/receipt.json').read_bytes())
    if exported['mode']!=b['mode'] or len(exported['rows'])!=1:raise ValueError('exporter receipt mismatch')
    if method=='retinexformer':
        if exported['GT_mean'] or exported['self_ensemble']:raise ValueError('changed options')
    elif exported['parameter_hash_before']!=exported['parameter_hash_after'] or exported['parameter_hash_before']!=b['accepted_parameter_hash']:raise ValueError('model parameter changed')
    row=exported['rows'][0];file=out/'outputs'/(allowed.stem+'.npy');y=np.load(file,allow_pickle=False)
    if row['low_sha256']!=spec['smoke']['sha256']:raise ValueError('input identity mismatch')
    digest=hashlib.sha256(y.tobytes()).hexdigest()
    if digest!=row['output_sha256']:raise ValueError('output hash mismatch')
    torch.cuda.synchronize()
    result={'method':method,'shape':list(y.shape),'dtype':str(y.dtype),'finite':bool(np.isfinite(y).all()),'output_sha256':digest,'file_sha256':sha(file),'runtime_s':row['runtime_s'],'peak_allocated_bytes':row['peak_memory_bytes'],'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    write(out/'output_freeze.json',result)

if __name__=='__main__':main()
