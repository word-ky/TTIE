"""Extract immutable invocation metadata from Git; no target or host access."""
import hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
COMMIT='579c3691a80f5b7cadfd706a2fe6750876c53aa0'
paths=['research_log/T071B/run.py','research_log/T071B/baseline_bindings.json','ttie/retinex_exporter.py','ttie/snr_exporter.py']
sources={}
for path in paths:
    raw=subprocess.check_output(['git','show',COMMIT+':'+path],cwd=HERE)
    sources[path]={'commit':COMMIT,'blob':hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),'sha256':hashlib.sha256(raw).hexdigest()}
    if path.endswith('baseline_bindings.json'):
        (HERE/'baseline_bindings.json').write_bytes(raw)
spec={'task':'T072-O','gate':{'name':'NVIDIA RTX A6000','free_mib':40960,'process_max_mib':1024},'smoke':{'path':'/media/wenchang/F/wjq/TTIE/shared/t072i/uhdll/input/1003_UHD_LL.JPG','sha256':'cb89bcd019ac26a34665f07189483a0138e76e9b00714337c5e2919bae9062ca','shape':[2160,3840,3],'mode':'RGB'},'bindings':{'retinexformer':'a9a61665602618adf20c5fb6b05af22da5906c293876fe27342c05a5da833d00','snr_aware':'03ce8bb7f608051ec5c5fd92b7a3e3baea315a2d3978f4c62cc114d226cee875'},'entrypoints':{'retinexformer':'ttie.retinex_exporter','snr_aware':'ttie.snr_exporter'},'argv_template':['--low','SMOKE','--checkpoint','ACCEPTED_CHECKPOINT','--config','ACCEPTED_CONFIG','--out','METHOD_OUTPUT'],'snr_extra':['--source','ACCEPTED_SOURCE'],'options':{'resize':False,'crop':False,'tiling':False,'precision_override':False,'normalization_override':False,'GT_mean':False,'self_ensemble':False},'sources':sources,'reference_reads':0,'metrics':0}
(HERE/'spec.json').write_bytes((json.dumps(spec,indent=2)+'\n').encode())
print(hashlib.sha256((HERE/'spec.json').read_bytes()).hexdigest())
