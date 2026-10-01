"""Independent offline replay of the sealed launcher's observed gate stop."""
import csv,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
receipt=json.loads((HERE/'receipt.json').read_bytes());snapshot=json.loads((HERE/'gpu_snapshot.json').read_bytes())
assert receipt=={'task':'T072-O-future-smoke','runs':[],'reference_reads':0,'metrics':0,'classification':'BLOCKED_GPU_GATE'}
gpus=[{'index':int(r[0]),'uuid':r[1],'name':r[2],'free_mib':int(r[3])} for r in csv.reader(snapshot['raw'][0].splitlines(),skipinitialspace=True)]
apps=[{'uuid':r[0],'pid':int(r[1]),'name':r[2],'memory_mib':int(r[3])} for r in csv.reader(snapshot['raw'][1].splitlines(),skipinitialspace=True)]
assert gpus==snapshot['gpus'] and apps==snapshot['processes']
assert len(snapshot['commands'])==2
assert not [g for g in gpus if g['name']=='NVIDIA RTX A6000' and g['free_mib']>=40960 and all(p['memory_mib']<=1024 for p in apps if p['uuid']==g['uuid'])]
log=(HERE/'workflow_run/train.log').read_text()
assert log.count('[autodl] started_at=')==1 and log.count('[autodl] exit_code=0')==1
meta=json.loads((HERE/'workflow_run/meta.json').read_bytes())
assert meta['runId']=='20260924-010502-ttie-t072aj-sealed-smoke'
assert meta['command'].count('launcher.py')==1
assert '--runtime-root /media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t071b-official-baselines' in meta['command']
commit='4eaa37e7fce5dba59a5bb769363d061cc340bac3'
def source(name):return subprocess.check_output(['git','show',commit+':research_log/T072O/'+name],cwd=HERE)
seal=json.loads(source('seal.json'));assert seal['root_sha256']=='c000543fc2623f76d1270d7c479114ea325ad3971d12821bbf46bcd8629e8c9d'
assert hashlib.sha256(json.dumps(seal['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==seal['root_sha256']
for name,h in seal['files'].items():assert hashlib.sha256(source(name)).hexdigest()==h
assert not (HERE/'retinexformer').exists() and not (HERE/'snr_aware').exists()
result={'task':'T072-AJ','classification':'BLOCKED_GPU_GATE','verification':'PASS','launcher_invocations':1,'gate_snapshots':1,'inference_runs':0,'input_payload_reads':0,'reference_reads':0,'metrics':0,'process_interventions':0,'accounting_basis':'empty run receipts and gate exit in immutable launcher control flow; not independent syscall tracing','source_commit':commit}
(HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
