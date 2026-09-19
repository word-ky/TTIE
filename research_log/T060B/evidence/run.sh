#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260919-114102-ttie-t060b-common'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260919-114102-ttie-t060b-common'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260919-114102-ttie-t060b-common/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/wenchang/asdasdsad/wjq/TTIE/releases/20260919-ttie-t060b-common && export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -m research_log.T060B.predict --out "$AUTODL_ARTIFACTS_DIR/T060B" && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -m research_log.T060B.diagnose --out "$AUTODL_ARTIFACTS_DIR/T060B" && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -m research_log.T060B.verify --out "$AUTODL_ARTIFACTS_DIR/T060B"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
