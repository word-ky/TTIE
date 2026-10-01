"""Read only the explicit sealed asset paths; no GPU/model/image imports."""
import hashlib,importlib.util,json,os,platform,stat,sys
from pathlib import Path

def audit_one(path,expected):
    row={'path':str(path),'expected_sha256':expected,'exists':False,'regular':False,'readable':False,'size':None,'observed_sha256':None}
    try:
        s=path.stat();row.update(exists=True,regular=stat.S_ISREG(s.st_mode),size=s.st_size)
        if not row['regular']:raise ValueError('not a regular file')
        h=hashlib.sha256()
        with path.open('rb') as f:
            row['readable']=True
            for b in iter(lambda:f.read(8388608),b''):h.update(b)
        row['observed_sha256']=h.hexdigest();row['match']=h.hexdigest()==expected
        if not row['match']:row['error']='digest mismatch'
    except (OSError,ValueError) as e:row.update(match=False,error_type=type(e).__name__,error=str(e))
    return row

def validate_manifest(manifest,expected_sha):
    raw=json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()
    if hashlib.sha256(raw).hexdigest()!=expected_sha:raise ValueError('sealed allowlist changed before file access')

def run(manifest,expected_sha):
    validate_manifest(manifest,expected_sha)
    environment={'executable':sys.executable,'python':platform.python_version(),'packages':{}}
    for name in ('torch','numpy','cv2','PIL','einops'):
        spec=importlib.util.find_spec(name)
        environment['packages'][name]=None if spec is None else spec.origin
    result={'task':'T072-P','environment':environment,'runtime_root':manifest['runtime_root'],'manifest_sha256':expected_sha,'rows':[],'accounting':dict(gpu_queries=0,cuda_initializations=0,inference_runs=0,input_payload_reads=0,reference_reads=0,metrics=0,process_interventions=0),'model_import_execution':'NOT_EXECUTED_BY_DESIGN'}
    for name,expected in manifest['files'].items():
        row=audit_one(Path(manifest['runtime_root'])/name,expected);row['declared_path']=name
        result['rows'].append(row)
        if not row['match']:
            result.update(classification='BLOCKED_RUNTIME_ASSET_PROVENANCE',remaining_not_audited=len(manifest['files'])-len(result['rows']))
            return result
    result['classification']='NATIVE4K_RUNTIME_ASSETS_VERIFIED' if all(environment['packages'].values()) else 'BLOCKED_RUNTIME_ASSET_PROVENANCE'
    result['remaining_not_audited']=0
    return result
