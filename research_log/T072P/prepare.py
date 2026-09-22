import hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
COMMIT='4eaa37e7fce5dba59a5bb769363d061cc340bac3'
def blob(name):return subprocess.check_output(['git','show',COMMIT+':research_log/T072O/'+name],cwd=HERE)
raw=blob('spec.json');assert hashlib.sha256(raw).hexdigest()=='4a6f2c39bc4426327bf9a20d1c00bf02cdd514b01d46e44ece072fd9a1645895'
spec=json.loads(raw);seal=json.loads(blob('seal.json'))
assert seal['root_sha256']=='c000543fc2623f76d1270d7c479114ea325ad3971d12821bbf46bcd8629e8c9d'
assert hashlib.sha256(json.dumps(seal['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==seal['root_sha256']
bindings_raw=blob('baseline_bindings.json');assert hashlib.sha256(bindings_raw).hexdigest()==spec['sources']['research_log/T071B/baseline_bindings.json']['sha256']
files={}
for method,b in json.loads(bindings_raw).items():
    files[b['accepted_binding_file']]=spec['bindings'][method]
    files.update(b['files'])
for path,identity in spec['sources'].items():files[path]=identity['sha256']
manifest={'task':'T072-P','runtime_root':'/media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t071b-official-baselines','runtime_root_provenance':'accepted T071-B release_path in project research_log/T071B_state.json; explicit runtime-root for future launcher','t072o_commit':COMMIT,'spec_sha256':hashlib.sha256(raw).hexdigest(),'seal_root':seal['root_sha256'],'files':files}
(HERE/'manifest.json').write_bytes((json.dumps(manifest,indent=2)+'\n').encode())
print(len(files),'sealed paths')
