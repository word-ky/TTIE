set -e
cd /home/wenchang/asdasdsad/wjq/TTIE/shared/t023a/source
/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python - <<'PY'
from pathlib import Path
import json,hashlib,datetime,shutil
base=Path('/home/wenchang/asdasdsad/wjq/TTIE')
audit=base/'runs/20260914-061849-ttie-t023a-validation/artifacts/audit'
f=audit/'freeze.json'
freeze=json.loads(f.read_bytes())
assert len(freeze['rows'])==100
assert 'exit_code=0' in (audit.parent.parent/'train.log').read_text()
shared=base/'shared/t023a'
shutil.copyfile(f,shared/'T023A_pre_reference_freeze.json')
receipt=dict(reference_deployment_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),freeze_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),freeze_completed_utc=freeze['completed_utc'],source='accepted validation reference bytes, no official test')
assert not (shared/'normal').exists()
shutil.copytree(base/'shared/t022c/normal',shared/'normal')
receipt['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(shared/'T023A_reference_deployment.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
PY
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python scripts/evaluate_t022a.py --audit /home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-061849-ttie-t023a-validation/artifacts/audit --low-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t022a/low --normal-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t023a/normal --split research_log/T022A_data/split.json --deployment /home/wenchang/asdasdsad/wjq/TTIE/shared/t023a/T023A_reference_deployment.json > /home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-061849-ttie-t023a-validation/evaluation.log 2>&1
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python scripts/compare_t023a.py /home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-061849-ttie-t023a-validation/artifacts/audit /home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-042925-ttie-t022c-ev2/artifacts/audit > /home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-061849-ttie-t023a-validation/comparison.log 2>&1
