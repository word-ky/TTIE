"""Replay all audited path identities from immutable T072-O Git metadata."""
import hashlib,json,subprocess
from pathlib import Path,PurePosixPath
HERE=Path(__file__).resolve().parent
def source(name):return subprocess.check_output(['git','show','4eaa37e7fce5dba59a5bb769363d061cc340bac3:research_log/T072O/'+name],cwd=HERE)
raw=source('spec.json');assert hashlib.sha256(raw).hexdigest()=='4a6f2c39bc4426327bf9a20d1c00bf02cdd514b01d46e44ece072fd9a1645895'
spec=json.loads(raw);seal=json.loads(source('seal.json'))
assert seal['root_sha256']=='c000543fc2623f76d1270d7c479114ea325ad3971d12821bbf46bcd8629e8c9d'
assert hashlib.sha256(json.dumps(seal['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==seal['root_sha256']
bindings_raw=source('baseline_bindings.json')
assert hashlib.sha256(bindings_raw).hexdigest()==spec['sources']['research_log/T071B/baseline_bindings.json']['sha256']
expected={}
for method,b in json.loads(bindings_raw).items():
    expected[b['accepted_binding_file']]=spec['bindings'][method];expected.update(b['files'])
expected.update({p:v['sha256'] for p,v in spec['sources'].items()})
manifest=json.loads((HERE/'manifest.json').read_bytes());receipt=json.loads((HERE/'receipt.json').read_bytes())
assert manifest['files']==expected
assert receipt['manifest_sha256']==hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert receipt['runtime_root']==manifest['runtime_root']=='/media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t071b-official-baselines'
assert len(receipt['rows'])==len(expected)==70
assert {r['declared_path'] for r in receipt['rows']}==set(expected)
for row in receipt['rows']:
    assert row['expected_sha256']==row['observed_sha256']==expected[row['declared_path']]
    assert row['path']==str(PurePosixPath(receipt['runtime_root'])/row['declared_path'])
    assert row['exists'] and row['regular'] and row['readable'] and row['match'] and row['size']>=0
assert receipt['remaining_not_audited']==0
assert set(receipt['environment']['packages'])=={'torch','numpy','cv2','PIL','einops'}
assert all(receipt['environment']['packages'].values())
assert receipt['accounting']==dict(gpu_queries=0,cuda_initializations=0,inference_runs=0,input_payload_reads=0,reference_reads=0,metrics=0,process_interventions=0)
assert receipt['classification']=='NATIVE4K_RUNTIME_ASSETS_VERIFIED'
print('NATIVE4K_RUNTIME_ASSETS_VERIFIED: independent replay PASS; 70/70 files')
