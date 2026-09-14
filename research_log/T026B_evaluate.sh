set -e
cd /home/wenchang/asdasdsad/wjq/TTIE/releases/20260914-130657-ttie-t026b-80steps
/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python - <<'PY'
from pathlib import Path
import json,hashlib,datetime,shutil
base=Path('/home/wenchang/asdasdsad/wjq/TTIE')
audit=base/'runs/20260914-130732-ttie-t026b-80steps/artifacts/audit'
f=audit/'freeze.json'
freeze=json.loads(f.read_bytes())
assert len(freeze['rows'])==100
assert 'exit_code=0' in (audit.parent.parent/'train.log').read_text()
shared=base/'shared/t026b'
shutil.copyfile(f,shared/'T026B_pre_reference_freeze.json')
receipt=dict(reference_deployment_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),freeze_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),freeze_completed_utc=freeze['completed_utc'],source='accepted validation reference bytes, no official test',reference_scope='task-specific shared/t026b/normal, other-task references existed outside inference allowlist')
assert not (shared/'normal').exists()
shutil.copytree(base/'shared/t026a/normal',shared/'normal')
receipt['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(shared/'T026B_reference_deployment.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
PY
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python scripts/evaluate_t026b.py --audit /home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-130732-ttie-t026b-80steps/artifacts/audit --low-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t022a/low --normal-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t026b/normal --split research_log/T022A_data/split.json --deployment /home/wenchang/asdasdsad/wjq/TTIE/shared/t026b/T026B_reference_deployment.json > /home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-130732-ttie-t026b-80steps/evaluation.log 2>&1
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python scripts/compare_t026b.py /home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-130732-ttie-t026b-80steps/artifacts/audit /home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-113853-ttie-t026a-gamma05/artifacts/audit > /home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-130732-ttie-t026b-80steps/comparison.log 2>&1
