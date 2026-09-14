#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260914-232650-ttie-t032a-radius'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-232650-ttie-t032a-radius'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-232650-ttie-t032a-radius/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export CUDA_VISIBLE_DEVICES=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD"
PY=/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python
$PY -m pytest -q tests/test_t032a_support.py && $PY scripts/prepare_t032a_radius.py --bank-root /home/wenchang/asdasdsad/wjq/TTIE/runs/20260912-181047-ttie-t014-stage-a-repaired/artifacts/audit --receipt research_log/T014_energy_receipt.json --checkpoint /home/wenchang/asdasdsad/wjq/TTIE/research_log/T014_energy.pt --out "$AUTODL_ARTIFACTS_DIR/radius"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
