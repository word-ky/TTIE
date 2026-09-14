#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260914-194203-ttie-t031a-support'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-194203-ttie-t031a-support'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-194203-ttie-t031a-support/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD" CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python
$PY -m pytest tests/test_t031a_support.py -q && $PY research_log/T031A_support/prepare.py --bank-root /home/wenchang/asdasdsad/wjq/TTIE/runs/20260912-181047-ttie-t014-stage-a-repaired/artifacts/audit --receipt research_log/T014_energy_receipt.json --accepted /home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-183308-ttie-t030a-guard/artifacts/audit --manifest research_log/T030A_cohort/manifest.json --checkpoint /home/wenchang/asdasdsad/wjq/TTIE/research_log/T014_energy.pt --out "$AUTODL_ARTIFACTS_DIR/support"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
