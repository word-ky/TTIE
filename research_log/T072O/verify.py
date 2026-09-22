"""Independent source/digest and mock receipt checks. No host or target reads."""
import hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
raw=(HERE/'spec.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='4a6f2c39bc4426327bf9a20d1c00bf02cdd514b01d46e44ece072fd9a1645895'
spec=json.loads(raw)
for path,identity in spec['sources'].items():
    data=subprocess.check_output(['git','show','579c3691a80f5b7cadfd706a2fe6750876c53aa0:'+path],cwd=HERE)
    assert hashlib.sha256(data).hexdigest()==identity['sha256']
    assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==identity['blob']
    if path.endswith('baseline_bindings.json'):assert data==(HERE/'baseline_bindings.json').read_bytes()
seal=json.loads((HERE/'seal.json').read_bytes())
assert hashlib.sha256(json.dumps(seal['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==seal['root_sha256']
for path,digest in seal['files'].items():assert hashlib.sha256((HERE/path).read_bytes()).hexdigest()==digest,path
fixtures=json.loads((HERE/'mock_evidence.json').read_bytes())
assert fixtures['blocked']['events']==['snapshot']
assert fixtures['bad_binding']['events']==['snapshot','bindings']
success=fixtures['success'];assert success['events']==['snapshot','bindings','input','retinexformer','snr_aware']
assert success['receipt']['classification']=='UHDLL_NATIVE_BASELINES_SMOKE_PASS'
for method,row in zip(('retinexformer','snr_aware'),success['receipt']['runs']):
    assert row['method']==method and row['run_count']==1 and row['status']=='FROZEN'
    result=row['output']
    assert set(result)=={'method','shape','dtype','finite','output_sha256','file_sha256','runtime_s','peak_allocated_bytes','peak_reserved_bytes'}
    assert result['shape']==[2160,3840,3] and result['dtype']=='float32' and result['finite'] is True
    assert result['output_sha256']==hashlib.sha256(method.encode()).hexdigest()
    assert result['file_sha256']==hashlib.sha256(b'synthetic').hexdigest()
for method in ('retinexformer','snr_aware'):
    case=fixtures['failure_'+method]
    assert case['events'].count(method)==1 and case['events'][-1]==method
    assert case['receipt']['classification']=='BLOCKED_NATIVE4K_'+method.upper()
    assert 'injected CUDA out of memory' in case['receipt']['traceback']
for case in fixtures.values():assert case['receipt']['reference_reads']==case['receipt']['metrics']==0
assert seal['accounting']==dict(real_inference_runs=0,real_input_payload_reads=0,reference_reads=0,real_metrics=0,process_interventions=0)
print('NATIVE4K_SMOKE_LAUNCHER_SEALED: independent verification PASS')
