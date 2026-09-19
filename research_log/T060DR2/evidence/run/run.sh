#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260919-180757-ttie-t060dr2-selector'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260919-180757-ttie-t060dr2-selector'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260919-180757-ttie-t060dr2-selector/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/wenchang/asdasdsad/wjq/TTIE/releases/20260919-ttie-t060dr2-selector
export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$PWD:$PWD/tests" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python
"$PY" -m research_log.T060DR2.infer --out "$AUTODL_ARTIFACTS_DIR/T060DR2" && "$PY" -m research_log.T060DR2.diagnose --out "$AUTODL_ARTIFACTS_DIR/T060DR2" && "$PY" -m research_log.T060DR2.verify --out "$AUTODL_ARTIFACTS_DIR/T060DR2"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
