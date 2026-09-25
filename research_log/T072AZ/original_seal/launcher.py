"""Future one-shot smoke launcher. This task executes it with mocks only."""
import argparse,csv,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
SPEC_SHA='4a6f2c39bc4426327bf9a20d1c00bf02cdd514b01d46e44ece072fd9a1645895'
METHODS=('retinexformer','snr_aware')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8388608),b''):h.update(block)
    return h.hexdigest()
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
def load_contract(raw=None):
    raw=(HERE/'spec.json').read_bytes() if raw is None else raw
    if hashlib.sha256(raw).hexdigest()!=SPEC_SHA:raise ValueError('sealed specification mismatch')
    spec=json.loads(raw)
    if sha(HERE/'baseline_bindings.json')!=spec['sources']['research_log/T071B/baseline_bindings.json']['sha256']:raise ValueError('binding metadata changed')
    return spec,json.loads((HERE/'baseline_bindings.json').read_bytes())
def choose(snapshot):
    eligible=[]
    for row in snapshot['gpus']:
        apps=[p for p in snapshot['processes'] if p['uuid']==row['uuid']]
        if row['name']=='NVIDIA RTX A6000' and row['free_mib']>=40960 and all(p['memory_mib']<=1024 for p in apps):eligible.append(row['index'])
    return min(eligible) if eligible else None
def invocation(method,spec,binding,out):
    args=['--low',spec['smoke']['path'],'--checkpoint',binding['checkpoint'],'--config',binding['config'],'--out',str(out/'outputs')]
    if method=='snr_aware':args+=['--source',binding['source']]
    return args

class RealBackend:
    def __init__(self,root):self.root=Path(root).resolve()
    def snapshot(self):
        queries=[['nvidia-smi','--query-gpu=index,uuid,name,memory.free','--format=csv,noheader,nounits'],['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits']]
        raw=[subprocess.check_output(q,text=True) for q in queries]
        gpus=[{'index':int(r[0]),'uuid':r[1],'name':r[2],'free_mib':int(r[3])} for r in csv.reader(raw[0].splitlines(),skipinitialspace=True)]
        processes=[{'uuid':r[0],'pid':int(r[1]),'name':r[2],'memory_mib':int(r[3])} for r in csv.reader(raw[1].splitlines(),skipinitialspace=True) if r]
        return {'gpus':gpus,'processes':processes,'commands':queries,'raw':raw}
    def bindings(self,spec,bindings):
        for method,b in bindings.items():
            if b['accepted_binding_sha256']!=spec['bindings'][method]:raise ValueError('accepted binding changed')
            if sha(self.root/b['accepted_binding_file'])!=spec['bindings'][method]:raise ValueError('binding file changed')
            for path,digest in b['files'].items():
                if sha(self.root/path)!=digest:raise ValueError('frozen source/config/checkpoint changed: '+path)
    def input(self,smoke):
        from PIL import Image
        if sha(smoke['path'])!=smoke['sha256']:raise ValueError('smoke hash changed')
        with Image.open(smoke['path']) as im:
            if im.mode!='RGB' or list(im.size)!=[3840,2160]:raise ValueError('smoke geometry changed')
    def run(self,method,device,spec,binding,out):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(device),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        command=[sys.executable,str(HERE/'worker.py'),method,str(out),str(self.root)]
        p=subprocess.run(command,cwd=self.root,env=env,text=True,capture_output=True)
        write(out/'process.json',{'command':command,'CUDA_VISIBLE_DEVICES':str(device),'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
        if p.returncode:raise RuntimeError(p.stderr or p.stdout)
        return json.loads((out/'output_freeze.json').read_bytes())

def launch(backend,out,raw_spec=None):
    spec,bindings=load_contract(raw_spec)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    receipt={'task':'T072-O-future-smoke','runs':[],'reference_reads':0,'metrics':0}
    method=None
    try:
        snapshot=backend.snapshot();write(out/'gpu_snapshot.json',snapshot)
        device=choose(snapshot)
        if device is None:
            receipt['classification']='BLOCKED_GPU_GATE'
        else:
            receipt['device']=device
            backend.bindings(spec,bindings)
            backend.input(spec['smoke'])
            for method in METHODS:
                method_out=out/method;method_out.mkdir()
                attempt={'method':method,'run_count':1,'status':'STARTED'}
                receipt['runs'].append(attempt)
                write(method_out/'attempt.json',attempt)
                result=backend.run(method,device,spec,bindings[method],method_out)
                if result['method']!=method or result['shape']!=[2160,3840,3] or result['dtype']!='float32' or result['finite'] is not True:raise ValueError('invalid native output')
                required={'method','shape','dtype','finite','output_sha256','file_sha256','runtime_s','peak_allocated_bytes','peak_reserved_bytes'}
                if set(result)!=required:raise ValueError('receipt fields differ')
                if result['runtime_s']<0 or result['peak_allocated_bytes']<0 or result['peak_reserved_bytes']<result['peak_allocated_bytes']:raise ValueError('invalid telemetry')
                for key in ('output_sha256','file_sha256'):
                    if len(result[key])!=64 or any(c not in '0123456789abcdef' for c in result[key]):raise ValueError('invalid output hash')
                attempt.update(status='FROZEN',output=result)
                write(method_out/'run_receipt.json',attempt)
            receipt['classification']='UHDLL_NATIVE_BASELINES_SMOKE_PASS'
    except Exception as e:
        receipt.update(classification='BLOCKED_NATIVE4K_'+method.upper() if method else 'BLOCKED_PREFLIGHT',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
        if receipt['runs']:receipt['runs'][-1]['status']='FAILED'
    write(out/'receipt.json',receipt)
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime-root',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();print(json.dumps(launch(RealBackend(args.runtime_root),args.out)))
